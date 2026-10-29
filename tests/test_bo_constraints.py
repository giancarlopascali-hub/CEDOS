import threading

import pytest
import torch

from src.core.bo import (
    compile_botorch_feature_constraints,
    generate_feasible_initial_conditions,
)
from src.core.bo_policy import FeasibleInitializationError


FEATURES = [
    {"name": "x", "type": "continuous", "range": "[0, 1]"},
    {"name": "y", "type": "continuous", "range": "[0, 1]"},
]


def test_feature_constraints_are_intra_point_and_use_nonnegative_margins():
    constraints = compile_botorch_feature_constraints(
        ["x <= y", "x = 0.5"], FEATURES
    )
    assert all(is_intra_point is True for _, is_intra_point in constraints)
    points = torch.tensor([[[0.5, 0.8]], [[0.7, 0.2]]])
    assert constraints[0][0](points)[0].item() >= 0
    assert constraints[0][0](points)[1].item() < 0
    assert constraints[1][0](points)[0].item() >= 0
    assert constraints[1][0](points)[1].item() < 0


def test_feasible_initialization_has_a_hard_draw_cap_and_no_invalid_fallback():
    bounds = torch.tensor([[0.0], [1.0]])
    impossible = [(lambda value: -torch.ones(value.shape[:-1]), True)]
    with pytest.raises(FeasibleInitializationError, match="after 32 draws"):
        generate_feasible_initial_conditions(
            bounds,
            q=1,
            num_restarts=2,
            p_constraints=impossible,
            max_draws=32,
            chunk_size=16,
        )


def test_feasible_initialization_honors_cancellation():
    bounds = torch.tensor([[0.0], [1.0]])
    cancelled = threading.Event()
    cancelled.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        generate_feasible_initial_conditions(
            bounds,
            q=1,
            num_restarts=2,
            p_constraints=[],
            cancel_event=cancelled,
        )
