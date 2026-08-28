import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.dsa_backend import (
    DeepseekSparseAttnBackend,
    DSAFlashMLAMetadata,
    _flashmla_kv_physical_topk,
    _pad_flashmla_page_table,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDSAFlashMLAKPool(CustomTestCase):
    def _mock_flashmla(self, **functions):
        package = ModuleType("sgl_kernel")
        package.__path__ = []
        module = ModuleType("sgl_kernel.flash_mla")
        for name, function in functions.items():
            setattr(module, name, function)
        package.flash_mla = module
        return patch.dict(
            sys.modules,
            {"sgl_kernel": package, "sgl_kernel.flash_mla": module},
        )

    def test_kpool_gate_accepts_both_flashmla_backends(self):
        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.dsa_index_kpool = 4
        indices = torch.zeros((1, 2051), dtype=torch.int32)

        for implementation in ("flashmla_sparse", "flashmla_kv"):
            with self.subTest(implementation=implementation):
                backend._check_kpool_tail_backend(indices, implementation, "decode")

        with self.assertRaises(NotImplementedError):
            backend._check_kpool_tail_backend(indices, "flashmla_sparse_q8", "decode")

    def test_flashmla_page_table_padding_uses_backend_alignment(self):
        page_table = torch.arange(2 * 2051, dtype=torch.int32).view(2, 2051)

        sparse = _pad_flashmla_page_table(page_table, alignment=128)
        decode = _pad_flashmla_page_table(page_table, alignment=64)

        self.assertEqual(sparse.shape, (2, 2176))
        self.assertEqual(decode.shape, (2, 2112))
        torch.testing.assert_close(sparse[:, :2051], page_table)
        torch.testing.assert_close(decode[:, :2051], page_table)
        self.assertTrue(torch.all(sparse[:, 2051:] == -1))
        self.assertTrue(torch.all(decode[:, 2051:] == -1))

        aligned = torch.zeros((1, 2048), dtype=torch.int32)
        self.assertIs(_pad_flashmla_page_table(aligned, alignment=128), aligned)
        self.assertEqual(_flashmla_kv_physical_topk(2048, 1), 2048)
        self.assertEqual(_flashmla_kv_physical_topk(2048, 4), 2112)

    def test_sparse_forward_passes_padded_indices_and_valid_lengths(self):
        captured = {}

        def fake_sparse_forward(**kwargs):
            captured.update(kwargs)
            q = kwargs["q"]
            output = q.new_zeros((q.shape[0], q.shape[1], kwargs["d_v"]))
            return output, None, None

        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.device_sm_major = 9
        backend.dsa_index_kpool = 4
        q = torch.zeros((2, 64, 576), dtype=torch.bfloat16)
        kv = torch.zeros((64, 1, 576), dtype=torch.bfloat16)
        page_table = torch.arange(2 * 2051, dtype=torch.int32).view(2, 2051)
        topk_length = torch.tensor([2051, 2049], dtype=torch.int32)

        with self._mock_flashmla(flash_mla_sparse_fwd=fake_sparse_forward):
            output = backend._forward_flashmla_sparse(
                q_all=q,
                kv_cache=kv,
                v_head_dim=512,
                page_table_1=page_table,
                sm_scale=0.125,
                topk_length=topk_length,
            )

        indices = captured["indices"]
        self.assertEqual(indices.shape, (2, 1, 2176))
        torch.testing.assert_close(indices[:, 0, :2051], page_table)
        self.assertTrue(torch.all(indices[:, :, 2051:] == -1))
        self.assertIs(captured["topk_length"], topk_length)
        self.assertEqual(output.shape, (2, 64, 512))

    def test_with_kvcache_uses_decode_aligned_indices(self):
        captured = {}

        def fake_with_kvcache(**kwargs):
            captured.update(kwargs)
            q = kwargs["q"]
            return q.new_zeros((*q.shape[:-1], kwargs["head_dim_v"])), None

        backend = object.__new__(DeepseekSparseAttnBackend)
        backend.real_page_size = 64
        backend.kv_cache_dim = 8
        backend.dsa_kv_cache_store_fp8 = True
        backend.flashmla_kv_num_q_heads = 2
        backend.dsa_index_topk = 2048
        backend.dsa_index_kpool = 4

        scheduler = torch.zeros((1, 1), dtype=torch.int32)
        num_splits = torch.zeros((2,), dtype=torch.int32)
        metadata = SimpleNamespace(
            dsa_cache_seqlens_int32=torch.tensor([2051], dtype=torch.int32),
            flashmla_metadata=DSAFlashMLAMetadata(scheduler, num_splits),
        )
        layer = SimpleNamespace(tp_q_head_num=2, head_dim=4)
        q = torch.zeros((1, 2, 4), dtype=torch.bfloat16)
        kv = torch.zeros((64, 8), dtype=torch.uint8)
        page_table = torch.arange(2051, dtype=torch.int32).view(1, 2051)

        with self._mock_flashmla(flash_mla_with_kvcache=fake_with_kvcache):
            output = backend._forward_flashmla_kv(
                q_all=q,
                kv_cache=kv,
                v_head_dim=4,
                sm_scale=0.125,
                layer=layer,
                metadata=metadata,
                page_table_1=page_table,
            )

        indices = captured["indices"]
        self.assertEqual(indices.shape, (1, 1, 2112))
        torch.testing.assert_close(indices[:, 0, :2051], page_table)
        self.assertTrue(torch.all(indices[:, :, 2051:] == -1))
        self.assertNotIn("topk_length", captured)
        self.assertIs(captured["tile_scheduler_metadata"], scheduler)
        self.assertIs(captured["num_splits"], num_splits)
        self.assertEqual(output.shape, (1, 1, 2, 4))

    def test_with_kvcache_metadata_uses_physical_topk(self):
        cache_seqlens = torch.tensor([2051, 2048], dtype=torch.int32)

        for index_kpool, expected_topk in ((1, 2048), (4, 2112)):
            with self.subTest(index_kpool=index_kpool):
                get_mla_metadata = Mock(
                    return_value=(
                        torch.zeros((1, 1), dtype=torch.int32),
                        torch.zeros((3,), dtype=torch.int32),
                    )
                )
                backend = object.__new__(DeepseekSparseAttnBackend)
                backend.flashmla_kv_num_q_heads = 64
                backend.dsa_index_topk = 2048
                backend.dsa_index_kpool = index_kpool

                with self._mock_flashmla(get_mla_metadata=get_mla_metadata):
                    backend._compute_flashmla_metadata(cache_seqlens, seq_len_q=1)

                get_mla_metadata.assert_called_once_with(
                    cache_seqlens=cache_seqlens,
                    num_q_tokens_per_head_k=64,
                    num_heads_k=1,
                    num_heads_q=64,
                    is_fp8_kvcache=True,
                    topk=expected_topk,
                )


if __name__ == "__main__":
    unittest.main()
