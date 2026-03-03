import unittest

try:
    import torch
    from plugin.models.transformer_utils.MapTransformer import SlotwiseTemporalGate
    from plugin.models.mapers.vector_memory import VectorInstanceMemory
    TORCH_OK = True
except Exception:
    TORCH_OK = False


@unittest.skipUnless(TORCH_OK, 'torch/runtime deps unavailable')
class TestTemporalGateMVP(unittest.TestCase):

    def test_dimension_smoke(self):
        gate = SlotwiseTemporalGate(embed_dims=16, enabled=True)
        q = torch.randn(4, 2, 16)
        mem = torch.randn(3, 2, 16)
        values, alpha = gate(q, mem)
        self.assertEqual(values.shape, mem.shape)
        self.assertEqual(alpha.shape, (4, 2, 3))

    def test_no_history(self):
        gate = SlotwiseTemporalGate(embed_dims=8, enabled=True)
        q = torch.randn(2, 1, 8)
        mem = torch.randn(0, 1, 8)
        values, alpha = gate(q, mem)
        self.assertEqual(values.shape[0], 0)
        self.assertEqual(alpha.shape, (2, 1, 0))

    def test_one_alpha_parity(self):
        gate = SlotwiseTemporalGate(embed_dims=8, enabled=False)
        q = torch.randn(3, 1, 8)
        mem = torch.randn(4, 1, 8)
        values, alpha = gate(q, mem)
        self.assertTrue(torch.allclose(values, mem))
        self.assertTrue(torch.allclose(alpha, torch.ones_like(alpha)))

    def test_zero_alpha_suppression(self):
        gate = SlotwiseTemporalGate(embed_dims=8, enabled=True)
        with torch.no_grad():
            gate.gate_mlp[-1].weight.zero_()
            gate.gate_mlp[-1].bias.fill_(-100.0)
        q = torch.randn(3, 1, 8)
        mem = torch.randn(4, 1, 8)
        values, alpha = gate(q, mem)
        self.assertLess(alpha.max().item(), 1e-4)
        self.assertLess(values.abs().max().item(), 1e-4)

    def test_corrupted_read_isolation(self):
        clean = torch.randn(4, 2, 8)
        canonical = clean.clone()
        key_padding = torch.zeros((2, 4), dtype=torch.bool)
        select_indices = [torch.arange(4), torch.arange(4)]
        corrupt, _, _ = VectorInstanceMemory._build_local_corrupted_read_view(
            clean, canonical, select_indices, key_padding,
            corruption_mode='c_full', stale_offset=2, apply_corruption=True)
        self.assertTrue(torch.allclose(canonical, clean))
        self.assertEqual(corrupt.shape, clean.shape)

    def test_c_tail_selectivity(self):
        clean = torch.arange(5 * 1 * 1).view(5, 1, 1).float()
        key_padding = torch.zeros((1, 5), dtype=torch.bool)
        select_indices = [torch.arange(5)]
        corrupt, mask, eligible = VectorInstanceMemory._build_local_corrupted_read_view(
            clean.clone(), clean.clone(), select_indices, key_padding,
            corruption_mode='c_tail', stale_offset=1, tail_keep_recent=2, apply_corruption=True)
        self.assertTrue(mask[0, -1].item() is False)
        self.assertTrue(mask[0, -2].item() is False)
        self.assertTrue((eligible[0, :-2] == mask[0, :-2]).all().item())
        self.assertTrue(torch.allclose(corrupt[-1], clean[-1]))


if __name__ == '__main__':
    unittest.main()
