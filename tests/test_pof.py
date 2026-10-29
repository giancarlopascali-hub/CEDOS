import unittest

import torch
from botorch.acquisition.acquisition import AcquisitionFunction

from src.core.feasibility import (
    ProbabilityOfFeasibilityAcquisition,
    constraint_probability,
)


class _LogAcquisition(AcquisitionFunction):
    _log = True

    def __init__(self, pending):
        super().__init__(torch.nn.Identity())
        self.set_X_pending(pending)

    def forward(self, x_value):
        return torch.zeros(x_value.shape[:-2])


class ProbabilityOfFeasibilityTests(unittest.TestCase):
    def test_probability_at_boundary_is_half(self):
        probability = constraint_probability(
            torch.tensor(10.0), torch.tensor(2.0), '>=', 10.0
        )
        self.assertAlmostEqual(float(probability), 0.5, places=6)

    def test_probability_moves_in_expected_direction(self):
        lower_good = constraint_probability(
            torch.tensor(12.0), torch.tensor(1.0), '>=', 10.0
        )
        lower_bad = constraint_probability(
            torch.tensor(8.0), torch.tensor(1.0), '>=', 10.0
        )
        upper_good = constraint_probability(
            torch.tensor(8.0), torch.tensor(1.0), '<=', 10.0
        )
        self.assertGreater(float(lower_good), float(lower_bad))
        self.assertAlmostEqual(float(lower_good), float(upper_good), places=6)

    def test_zero_standard_deviation_is_numerically_safe(self):
        probability = constraint_probability(
            torch.tensor(11.0), torch.tensor(0.0), '>=', 10.0
        )
        self.assertTrue(torch.isfinite(probability))
        self.assertGreater(float(probability), 0.999)

    def test_wrapper_infers_log_space_and_forwards_pending_points(self):
        pending = torch.tensor([[0.25]])
        base = _LogAcquisition(pending)
        wrapped = ProbabilityOfFeasibilityAcquisition(base, {}, [])
        self.assertTrue(wrapped.logarithmic)
        self.assertTrue(torch.equal(wrapped.X_pending, pending))

        replacement = torch.tensor([[0.75]])
        wrapped.set_X_pending(replacement)
        self.assertTrue(torch.equal(wrapped.X_pending, replacement))
        self.assertTrue(torch.equal(base.X_pending, replacement))


if __name__ == '__main__':
    unittest.main()
