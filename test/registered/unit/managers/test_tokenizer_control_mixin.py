import unittest

from pydantic import TypeAdapter

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.io_struct import UpdateWeightsFromTensorReqInput  # noqa: E402
from sglang.srt.managers.tokenizer_control_mixin import (  # noqa: E402
    _decode_serialized_named_tensor_payloads,
)
from sglang.srt.utils import MultiprocessingSerializer  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestUpdateWeightsFromTensorPayloadDecoding(CustomTestCase):
    def test_base64_bytes_from_json_are_decoded(self):
        payload = [("weight", [1, 2, 3])]
        encoded = MultiprocessingSerializer.serialize(payload, output_str=True)
        req = TypeAdapter(UpdateWeightsFromTensorReqInput).validate_python(
            {"serialized_named_tensors": [encoded]}
        )

        self.assertIsInstance(req.serialized_named_tensors[0], bytes)
        decoded = _decode_serialized_named_tensor_payloads(
            req.serialized_named_tensors
        )

        self.assertEqual(MultiprocessingSerializer.deserialize(decoded[0]), payload)

    def test_raw_pickle_bytes_are_preserved(self):
        payload = [("weight", [1, 2, 3])]
        raw = MultiprocessingSerializer.serialize(payload)

        decoded = _decode_serialized_named_tensor_payloads([raw])

        self.assertEqual(decoded, [raw])
        self.assertEqual(MultiprocessingSerializer.deserialize(decoded[0]), payload)


if __name__ == "__main__":
    unittest.main()
