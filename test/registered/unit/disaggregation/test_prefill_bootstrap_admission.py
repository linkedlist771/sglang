import threading
import time
import types
import unittest
from collections import defaultdict, deque
from types import SimpleNamespace

from sglang.srt.disaggregation.base.conn import KVPoll
from sglang.srt.disaggregation.mooncake.conn import (
    MooncakeKVManager,
    MooncakeKVReceiver,
    MooncakeKVSender,
    TransferKVChunk,
)
from sglang.srt.disaggregation.prefill import PrefillBootstrapQueue
from sglang.srt.disaggregation.utils import ReqToMetadataIdxAllocator
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="stage-a-test-cpu")


def _make_req(rid: str):
    return SimpleNamespace(
        rid=rid,
        origin_input_ids=[1, 2, 3],
        sampling_params=SimpleNamespace(max_new_tokens=8),
        finished_reason=None,
        return_logprob=False,
        disagg_kv_sender=None,
    )


def _make_queue(capacity: int):
    queue = PrefillBootstrapQueue.__new__(PrefillBootstrapQueue)
    queue.queue = []
    queue.pending_queue = deque()
    queue.req_to_metadata_buffer_idx_allocator = ReqToMetadataIdxAllocator(capacity)
    queue.max_total_num_tokens = 1024
    queue.scheduler = SimpleNamespace(
        waiting_queue=[],
        disagg_prefill_inflight_queue=[],
        cur_batch=None,
        last_batch=None,
        running_batch=SimpleNamespace(reqs=[]),
        model_config=SimpleNamespace(num_key_value_heads=1),
    )

    def fake_create_kv_sender(self, req, num_kv_heads):
        req.disagg_kv_sender = object()
        self.queue.append(req)

    queue._create_kv_sender = types.MethodType(fake_create_kv_sender, queue)
    return queue


class TestPrefillBootstrapAdmission(unittest.TestCase):
    def test_admission_defers_requests_without_sender(self):
        queue = _make_queue(capacity=2)
        reqs = [_make_req(f"req-{i}") for i in range(3)]

        for req in reqs:
            queue.add(req, num_kv_heads=1)

        self.assertEqual([req.rid for req in queue.queue], ["req-0", "req-1"])
        self.assertEqual([req.rid for req in queue.pending_queue], ["req-2"])
        self.assertIsNotNone(reqs[0].disagg_kv_sender)
        self.assertIsNotNone(reqs[1].disagg_kv_sender)
        self.assertIsNone(reqs[2].disagg_kv_sender)

    def test_pending_request_admits_when_active_sender_finishes(self):
        queue = _make_queue(capacity=2)
        reqs = [_make_req(f"req-{i}") for i in range(3)]
        for req in reqs:
            queue.add(req, num_kv_heads=1)

        queue.queue.pop(0)
        queue._admit_pending(num_kv_heads=1)

        self.assertEqual([req.rid for req in queue.queue], ["req-1", "req-2"])
        self.assertEqual(list(queue.pending_queue), [])
        self.assertIsNotNone(reqs[2].disagg_kv_sender)

    def test_waiting_and_inflight_reqs_count_against_admission(self):
        queue = _make_queue(capacity=2)
        active_waiting = _make_req("waiting")
        active_waiting.disagg_kv_sender = object()
        queue.scheduler.waiting_queue = [active_waiting]

        queue.add(_make_req("req-0"), num_kv_heads=1)
        queue.add(_make_req("req-1"), num_kv_heads=1)

        self.assertEqual([req.rid for req in queue.queue], ["req-0"])
        self.assertEqual([req.rid for req in queue.pending_queue], ["req-1"])

    def test_running_batch_reqs_count_against_admission(self):
        queue = _make_queue(capacity=2)
        active_running = _make_req("running")
        active_running.disagg_kv_sender = object()
        queue.scheduler.running_batch = SimpleNamespace(reqs=[active_running])

        queue.add(_make_req("req-0"), num_kv_heads=1)
        queue.add(_make_req("req-1"), num_kv_heads=1)

        self.assertEqual([req.rid for req in queue.queue], ["req-0"])
        self.assertEqual([req.rid for req in queue.pending_queue], ["req-1"])

    def test_mooncake_sender_bootstrap_timeout_starts_after_metadata_arrives(self):
        class FakeKVManager:
            bootstrap_timeout = 60.0
            transfer_infos = {}

            def check_status(self, bootstrap_room):
                return KVPoll.Bootstrapping

        sender = MooncakeKVSender.__new__(MooncakeKVSender)
        sender.conclude_state = None
        sender.kv_mgr = FakeKVManager()
        sender.bootstrap_room = 7
        sender.init_time = None

        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        self.assertIsNone(sender.init_time)

        sender.kv_mgr.transfer_infos[sender.bootstrap_room] = {"session": object()}
        before_poll = time.time()

        self.assertEqual(sender.poll(), KVPoll.Bootstrapping)
        self.assertIsNotNone(sender.init_time)
        self.assertGreaterEqual(sender.init_time, before_poll)

    def test_mooncake_receiver_waiting_timeout_starts_after_transfer_begins(self):
        class FakeKVManager:
            waiting_timeout = 60.0

            def __init__(self):
                self.status = KVPoll.WaitingForInput
                self.failures = []

            def check_status(self, bootstrap_room):
                return self.status

            def record_failure(self, bootstrap_room, reason):
                self.failures.append((bootstrap_room, reason))

        receiver = MooncakeKVReceiver.__new__(MooncakeKVReceiver)
        receiver.conclude_state = None
        receiver.kv_mgr = FakeKVManager()
        receiver.bootstrap_room = 11
        receiver.init_time = None

        self.assertEqual(receiver.poll(), KVPoll.WaitingForInput)
        self.assertIsNone(receiver.init_time)
        self.assertEqual(receiver.kv_mgr.failures, [])

        receiver.kv_mgr.status = KVPoll.Transferring
        before_poll = time.time()

        self.assertEqual(receiver.poll(), KVPoll.Transferring)
        self.assertIsNotNone(receiver.init_time)
        self.assertGreaterEqual(receiver.init_time, before_poll)

        receiver.init_time = time.time() - receiver.kv_mgr.waiting_timeout - 1

        self.assertEqual(receiver.poll(), KVPoll.Failed)
        self.assertIn("KVPoll.Transferring", receiver.kv_mgr.failures[0][1])

    def test_mooncake_transfer_worker_drops_failed_room_without_transfer(self):
        class StopLoop(BaseException):
            pass

        class OneChunkQueue:
            def __init__(self, chunk):
                self.chunk = chunk
                self.calls = 0

            def get(self):
                self.calls += 1
                if self.calls == 1:
                    return self.chunk
                raise StopLoop()

        manager = SimpleNamespace()
        manager.enable_staging = False
        manager.transfer_infos = {3: {"session": object()}}
        manager.req_to_decode_prefix_len = {3: 0}
        manager.request_status = {3: KVPoll.Failed}

        chunk = TransferKVChunk(
            room=3,
            prefill_kv_indices=[],
            index_slice=slice(0, 0),
            is_last_chunk=True,
            prefill_aux_index=0,
            state_indices=None,
        )

        with self.assertRaises(StopLoop):
            MooncakeKVManager.transfer_worker(manager, OneChunkQueue(chunk), None)

        self.assertEqual(manager.transfer_infos, {})
        self.assertEqual(manager.req_to_decode_prefix_len, {})

    def test_decode_status_success_ignores_cleared_room(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.status_lock = threading.RLock()
        manager.request_status = {}
        manager.required_prefill_response_num_table = {7: 1}
        manager.prefill_response_tracker = defaultdict(set)
        manager.enable_staging = False

        MooncakeKVManager._handle_decode_status(manager, 7, KVPoll.Success, 0)

        self.assertNotIn(7, manager.request_status)
        self.assertNotIn(7, manager.prefill_response_tracker)

    def test_decode_status_success_ignores_missing_response_count(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.status_lock = threading.RLock()
        manager.request_status = {7: KVPoll.Transferring}
        manager.required_prefill_response_num_table = {}
        manager.prefill_response_tracker = defaultdict(set)
        manager.enable_staging = False

        MooncakeKVManager._handle_decode_status(manager, 7, KVPoll.Success, 0)

        self.assertEqual(manager.request_status[7], KVPoll.Transferring)
        self.assertNotIn(7, manager.prefill_response_tracker)

    def test_decode_status_success_completes_after_required_responses(self):
        manager = MooncakeKVManager.__new__(MooncakeKVManager)
        manager.status_lock = threading.RLock()
        manager.request_status = {7: KVPoll.Transferring}
        manager.required_prefill_response_num_table = {7: 2}
        manager.prefill_response_tracker = defaultdict(set)
        manager.enable_staging = False

        MooncakeKVManager._handle_decode_status(manager, 7, KVPoll.Success, 0)
        self.assertEqual(manager.request_status[7], KVPoll.Transferring)

        MooncakeKVManager._handle_decode_status(manager, 7, KVPoll.Success, 1)
        self.assertEqual(manager.request_status[7], KVPoll.Success)


if __name__ == "__main__":
    unittest.main()
