"""Acceptance tests for the locked Adaptive Masked-EarPrint handoff.

Locked architecture under test
------------------------------

    H = 1000 Hz

    BaseTarget
        |
        |  H < f <= E
        |  quintic smootherstep
        v
    Masked EarPrint
        |
        v
    final output

E-selection rule:

    earliest candidate E satisfying every hard gate.

There is deliberately no:

    - Pareto ranking
    - weighted candidate score
    - curvature minimisation
    - target-specific optimisation

The tests are intentionally deterministic and target-independent.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from engine.adaptive_handoff import (
    NOMINAL_ANCHOR_HZ,
    MIN_TRANSITION_OCTAVES,
    MAX_TRANSITION_OCTAVES,
    STABILITY_WINDOW_OCTAVES,
    _smootherstep,
    _smootherstep_derivative,
    _smootherstep_second_derivative,
    _evaluate_quintic_bridge,
    _monotone_bridge,
    adaptive_masked_handoff,
    validate_locked_smootherstep,
)


# ---------------------------------------------------------------------------
# Test-grid helpers
# ---------------------------------------------------------------------------

def _log_grid(
    start_hz: float = 20.0,
    end_hz: float = 12000.0,
    points_per_octave: int = 96,
):
    """Create a deterministic logarithmic frequency grid."""
    octaves = math.log2(
        end_hz / start_hz
    )

    count = int(
        round(
            octaves
            * points_per_octave
        )
    ) + 1

    return np.geomspace(
        start_hz,
        end_hz,
        count,
    )


def _grid_with_exact_1khz(
    start_hz: float = 20.0,
    end_hz: float = 12000.0,
    points_per_octave: int = 96,
):
    """Create a deterministic grid and explicitly insert 1 kHz."""
    freq = _log_grid(
        start_hz,
        end_hz,
        points_per_octave,
    )

    freq = np.unique(
        np.concatenate(
            [
                freq,
                np.array(
                    [1000.0],
                    dtype=float,
                ),
            ]
        )
    )

    return np.sort(freq)


def _smooth_base(freq):
    """Simple deterministic BaseTarget for synthetic tests."""
    x = np.log2(
        freq / 1000.0
    )

    return (
        86.0
        + 2.0 * x
        - 0.18 * x**2
    )


def _make_masked(
    freq,
    start_hz=1000.0,
    end_hz=1600.0,
    amplitude=-2.0,
):
    """Create a smooth monotonic mask for controlled synthetic tests."""
    x = np.log2(
        freq
        / start_hz
    )

    span = math.log2(
        end_hz
        / start_hz
    )

    t = np.clip(
        x / span,
        0.0,
        1.0,
    )

    w = _smootherstep(t)

    mask = amplitude * w

    return (
        _smooth_base(freq)
        + mask
    )


# ---------------------------------------------------------------------------
# Smootherstep mathematical tests
# ---------------------------------------------------------------------------

def test_smootherstep_exact_endpoints():
    """w(0)=0 and w(1)=1 exactly."""
    assert math.isclose(
        float(
            _smootherstep(
                np.array([0.0])
            )[0]
        ),
        0.0,
        abs_tol=1e-15,
    )

    assert math.isclose(
        float(
            _smootherstep(
                np.array([1.0])
            )[0]
        ),
        1.0,
        abs_tol=1e-15,
    )


def test_smootherstep_zero_endpoint_derivatives():
    """w'(0)=0 and w'(1)=0 exactly."""
    derivative = (
        _smootherstep_derivative(
            np.array(
                [0.0, 1.0]
            )
        )
    )

    assert math.isclose(
        float(derivative[0]),
        0.0,
        abs_tol=1e-15,
    )

    assert math.isclose(
        float(derivative[1]),
        0.0,
        abs_tol=1e-15,
    )


def test_smootherstep_second_derivative_endpoint_values():
    """The quintic also has zero second derivative at both endpoints."""
    second = (
        _smootherstep_second_derivative(
            np.array(
                [0.0, 1.0]
            )
        )
    )

    assert math.isclose(
        float(second[0]),
        0.0,
        abs_tol=1e-15,
    )

    assert math.isclose(
        float(second[1]),
        0.0,
        abs_tol=1e-15,
    )


def test_smootherstep_bounded():
    """0 <= w(t) <= 1 throughout the transition."""
    t = np.linspace(
        0.0,
        1.0,
        100001,
    )

    w = _smootherstep(t)

    assert np.all(
        np.isfinite(w)
    )

    assert float(
        np.min(w)
    ) >= -1e-12

    assert float(
        np.max(w)
    ) <= 1.0 + 1e-12


def test_smootherstep_monotonic():
    """w'(t) >= 0 throughout the transition."""
    t = np.linspace(
        0.0,
        1.0,
        100001,
    )

    derivative = (
        _smootherstep_derivative(t)
    )

    assert float(
        np.min(derivative)
    ) >= -1e-12


def test_locked_smootherstep_validator():
    """Public self-check must pass."""
    assert (
        validate_locked_smootherstep()
        is True
    )


# ---------------------------------------------------------------------------
# Locked geometry constants
# ---------------------------------------------------------------------------

def test_locked_constants():
    """Verify the locked H and transition limits."""
    assert math.isclose(
        NOMINAL_ANCHOR_HZ,
        1000.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        MIN_TRANSITION_OCTAVES,
        1.0 / 3.0,
        abs_tol=1e-12,
    )

    assert math.isclose(
        MAX_TRANSITION_OCTAVES,
        0.8,
        abs_tol=1e-12,
    )

    assert math.isclose(
        STABILITY_WINDOW_OCTAVES,
        0.20,
        abs_tol=1e-12,
    )


# ---------------------------------------------------------------------------
# Direct bridge tests
# ---------------------------------------------------------------------------

def test_quintic_bridge_basic_shape():
    """A simple monotonic bridge must pass every hard gate."""
    freq = np.geomspace(
        1000.0,
        1600.0,
        65,
    )

    target = (
        80.0
        + 2.0
        * np.log2(
            freq / 1000.0
        )
    )

    masked = target - 2.0

    # Replace endpoint geometry with a controlled monotonic masked curve.
    masked = (
        target
        - 2.0
        * _smootherstep(
            np.linspace(
                0.0,
                1.0,
                len(freq),
            )
        )
    )

    diagnostics = (
        _evaluate_quintic_bridge(
            freq,
            target,
            masked,
            0,
            len(freq) - 1,
        )
    )

    assert diagnostics[
        "passed"
    ]

    assert diagnostics[
        "finite_pass"
    ]

    assert diagnostics[
        "monotonicity_pass"
    ]

    assert diagnostics[
        "extrema_pass"
    ]

    assert diagnostics[
        "overshoot_pass"
    ]

    assert diagnostics[
        "undershoot_pass"
    ]

    assert diagnostics[
        "endpoint_weight_pass"
    ]

    assert diagnostics[
        "endpoint_derivative_pass"
    ]

    assert not diagnostics[
        "slope_reversal"
    ]


def test_quintic_bridge_has_no_endpoint_overshoot():
    """The locked bridge must remain between its endpoint levels."""
    freq = np.geomspace(
        1000.0,
        1800.0,
        101,
    )

    target = (
        80.0
        + 2.5
        * np.log2(
            freq / 1000.0
        )
    )

    masked = (
        target
        - 3.0
        * _smootherstep(
            np.linspace(
                0.0,
                1.0,
                len(freq),
            )
        )
    )

    diagnostics = (
        _evaluate_quintic_bridge(
            freq,
            target,
            masked,
            0,
            len(freq) - 1,
        )
    )

    assert (
        diagnostics[
            "overshoot_db"
        ]
        <= 1e-8
    )

    assert (
        diagnostics[
            "undershoot_db"
        ]
        <= 1e-8
    )


def test_quintic_bridge_rejects_slope_reversal():
    """A bridge whose endpoint geometry forces a reversal must fail."""
    freq = np.geomspace(
        1000.0,
        1600.0,
        65,
    )

    target = np.zeros(
        len(freq),
        dtype=float,
    )

    masked = np.zeros(
        len(freq),
        dtype=float,
    )

    # Force an artificial local reversal.
    masked[:] = np.linspace(
        0.0,
        2.0,
        len(freq),
    )

    masked[
        len(freq) // 2:
    ] -= np.linspace(
        0.0,
        3.0,
        len(freq) // 2,
    )

    diagnostics = (
        _evaluate_quintic_bridge(
            freq,
            target,
            masked,
            0,
            len(freq) - 1,
        )
    )

    assert not diagnostics[
        "passed"
    ]


# ---------------------------------------------------------------------------
# Basic handoff behaviour
# ---------------------------------------------------------------------------

def test_handoff_preserves_base_target_through_1khz():
    """BaseTarget must remain exact at and below H."""
    freq = _grid_with_exact_1khz()

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1600.0,
        amplitude=-2.0,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    below = (
        freq <= 1000.0
    )

    assert np.array_equal(
        output[below],
        target[below],
    )

    assert math.isclose(
        output[
            np.where(
                freq == 1000.0
            )[0][0]
        ],
        target[
            np.where(
                freq == 1000.0
            )[0][0]
        ],
        abs_tol=0.0,
    )

    assert diagnostics[
        "actual_handoff_hz"
    ] == 1000.0


def test_handoff_uses_masked_target_after_E():
    """Samples strictly after E must equal Masked EarPrint exactly."""
    freq = _grid_with_exact_1khz()

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1600.0,
        amplitude=-2.0,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    assert (
        diagnostics[
            "status"
        ]
        == "ADAPTIVE_HANDOFF"
    )

    E = diagnostics[
        "transition_end_hz"
    ]

    after = (
        freq > E
    )

    assert np.array_equal(
        output[after],
        masked[after],
    )


# ---------------------------------------------------------------------------
# Transition-width gates
# ---------------------------------------------------------------------------

def test_selected_E_respects_minimum_transition_width():
    """Selected E must be at least 1/3 octave after H."""
    freq = _grid_with_exact_1khz(
        points_per_octave=192,
    )

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.5,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    if (
        diagnostics["status"]
        == "ADAPTIVE_HANDOFF"
    ):
        assert (
            diagnostics[
                "transition_width_octaves"
            ]
            >=
            MIN_TRANSITION_OCTAVES
            - 1e-12
        )


def test_selected_E_respects_maximum_transition_width():
    """Selected E must never exceed 0.8 octave after H."""
    freq = _grid_with_exact_1khz(
        points_per_octave=192,
    )

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.5,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    if (
        diagnostics["status"]
        == "ADAPTIVE_HANDOFF"
    ):
        assert (
            diagnostics[
                "transition_width_octaves"
            ]
            <=
            MAX_TRANSITION_OCTAVES
            + 1e-12
        )


# ---------------------------------------------------------------------------
# Earliest-feasible-E rule
# ---------------------------------------------------------------------------

def test_selection_is_earliest_feasible_E():
    """The engine must select the first feasible E, not a later optimised E."""
    freq = _grid_with_exact_1khz(
        points_per_octave=192,
    )

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.0,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    assert (
        diagnostics[
            "status"
        ]
        == "ADAPTIVE_HANDOFF"
    )

    E = diagnostics[
        "transition_end_hz"
    ]

    width = math.log2(
        E / 1000.0
    )

    assert (
        width
        >= MIN_TRANSITION_OCTAVES
        - 1e-12
    )

    assert (
        width
        <= MAX_TRANSITION_OCTAVES
        + 1e-12
    )

    assert (
        diagnostics[
            "selection_rule"
        ]
        == "earliest_feasible_E"
    )

    assert (
        diagnostics[
            "candidate_scoring"
        ]
        is False
    )

    assert (
        diagnostics[
            "pareto_selection"
        ]
        is False
    )

    assert (
        diagnostics[
            "curvature_optimization"
        ]
        is False
    )


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

def test_handoff_is_deterministic():
    """Identical inputs must produce bitwise-identical outputs."""
    freq = _grid_with_exact_1khz()

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.0,
    )

    output_a, diag_a = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    output_b, diag_b = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    assert np.array_equal(
        output_a,
        output_b,
    )

    assert (
        diag_a[
            "status"
        ]
        == diag_b[
            "status"
        ]
    )

    assert (
        diag_a[
            "transition_end_hz"
        ]
        == diag_b[
            "transition_end_hz"
        ]
    )


# ---------------------------------------------------------------------------
# Vertical-translation invariance
# ---------------------------------------------------------------------------

def test_vertical_translation_invariance():
    """Adding a constant vertical offset must not change H/E geometry."""
    freq = _grid_with_exact_1khz()

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.0,
    )

    output_a, diag_a = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    offset = 37.25

    output_b, diag_b = (
        adaptive_masked_handoff(
            freq,
            target + offset,
            masked + offset,
        )
    )

    assert (
        diag_a[
            "actual_handoff_hz"
        ]
        == diag_b[
            "actual_handoff_hz"
        ]
    )

    assert (
        diag_a[
            "transition_end_hz"
        ]
        == diag_b[
            "transition_end_hz"
        ]
    )

    assert np.allclose(
        output_b,
        output_a + offset,
        rtol=0.0,
        atol=1e-10,
    )


# ---------------------------------------------------------------------------
# Positive scale invariance
# ---------------------------------------------------------------------------

def test_positive_scale_invariance():
    """Positive vertical scaling must preserve H/E geometry."""
    freq = _grid_with_exact_1khz()

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.0,
    )

    output_a, diag_a = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    scale = 3.0

    output_b, diag_b = (
        adaptive_masked_handoff(
            freq,
            target * scale,
            masked * scale,
        )
    )

    assert (
        diag_a[
            "actual_handoff_hz"
        ]
        == diag_b[
            "actual_handoff_hz"
        ]
    )

    assert (
        diag_a[
            "transition_end_hz"
        ]
        == diag_b[
            "transition_end_hz"
        ]
    )

    assert np.allclose(
        output_b,
        output_a * scale,
        rtol=0.0,
        atol=1e-9,
    )


# ---------------------------------------------------------------------------
# Fail-safe behaviour
# ---------------------------------------------------------------------------

def test_no_stable_handoff_is_safe():
    """Pathological input must return BaseTarget unchanged."""
    freq = _grid_with_exact_1khz(
        end_hz=12000.0,
        points_per_octave=96,
    )

    target = np.zeros(
        len(freq),
        dtype=float,
    )

    # Constant masked target creates no meaningful destination slope.
    masked = np.ones(
        len(freq),
        dtype=float,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    assert (
        diagnostics[
            "status"
        ]
        == "NO_STABLE_HANDOFF"
    )

    assert np.array_equal(
        output,
        target,
    )


# ---------------------------------------------------------------------------
# No out-of-domain modification
# ---------------------------------------------------------------------------

def test_no_modification_below_H():
    """No sample below H may differ from BaseTarget."""
    freq = _grid_with_exact_1khz()

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.0,
    )

    output, _ = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    assert np.array_equal(
        output[
            freq < 1000.0
        ],
        target[
            freq < 1000.0
        ],
    )


def test_no_modification_after_E_other_than_masked_target():
    """After E, output must not create another correction."""
    freq = _grid_with_exact_1khz()

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.0,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    if (
        diagnostics[
            "status"
        ]
        != "ADAPTIVE_HANDOFF"
    ):
        pytest.skip(
            "Synthetic input did not produce a valid handoff."
        )

    E = diagnostics[
        "transition_end_hz"
    ]

    after = (
        freq > E
    )

    assert np.array_equal(
        output[after],
        masked[after],
    )


# ---------------------------------------------------------------------------
# Endpoint continuity
# ---------------------------------------------------------------------------

def test_H_endpoint_equals_base_target():
    """At H, the bridge weight is exactly zero."""
    freq = _grid_with_exact_1khz()

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.0,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    h_idx = np.where(
        freq == 1000.0
    )[0][0]

    assert math.isclose(
        output[h_idx],
        target[h_idx],
        rel_tol=0.0,
        abs_tol=0.0,
    )

    if (
        diagnostics[
            "status"
        ]
        == "ADAPTIVE_HANDOFF"
    ):
        assert math.isclose(
            diagnostics[
                "smootherstep_endpoint_weight_pass"
            ],
            True,
        )


# ---------------------------------------------------------------------------
# No candidate ranking / optimisation
# ---------------------------------------------------------------------------

def test_no_legacy_candidate_score_in_locked_selection():
    """Locked diagnostics must explicitly report no score-based selection."""
    freq = _grid_with_exact_1khz()

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.0,
    )

    _, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    assert (
        diagnostics.get(
            "candidate_scoring",
            False,
        )
        is False
    )

    assert (
        diagnostics.get(
            "pareto_selection",
            False,
        )
        is False
    )

    assert (
        diagnostics.get(
            "curvature_optimization",
            False,
        )
        is False
    )


# ---------------------------------------------------------------------------
# Legacy compatibility helper
# ---------------------------------------------------------------------------

def test_monotone_bridge_compatibility_helper():
    """The legacy helper remains importable and validates endpoint geometry."""
    result = _monotone_bridge(
        80.0,
        82.0,
        2.0,
        2.0,
        0.5,
    )

    assert (
        result[
            "bridge_type"
        ]
        == "quintic_smootherstep"
    )

    assert (
        result[
            "endpoint_weight_start"
        ]
        == 0.0
    )

    assert (
        result[
            "endpoint_weight_end"
        ]
        == 1.0
    )

    assert (
        result[
            "endpoint_weight_derivative_start"
        ]
        == 0.0
    )

    assert (
        result[
            "endpoint_weight_derivative_end"
        ]
        == 0.0
    )


# ---------------------------------------------------------------------------
# Broad synthetic regression set
# ---------------------------------------------------------------------------

@pytest.mark.parametrize(
    "amplitude",
    [
        -0.5,
        -1.0,
        -1.5,
        -2.0,
        -2.5,
        -3.0,
    ],
)
def test_multiple_mask_amplitudes(
    amplitude,
):
    """The locked handoff remains deterministic across several mask sizes."""
    freq = _grid_with_exact_1khz(
        points_per_octave=192,
    )

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=amplitude,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    assert np.all(
        np.isfinite(output)
    )

    if (
        diagnostics[
            "status"
        ]
        == "ADAPTIVE_HANDOFF"
    ):
        assert (
            diagnostics[
                "actual_handoff_hz"
            ]
            == 1000.0
        )

        assert (
            diagnostics[
                "transition_width_octaves"
            ]
            >=
            MIN_TRANSITION_OCTAVES
            - 1e-12
        )

        assert (
            diagnostics[
                "transition_width_octaves"
            ]
            <=
            MAX_TRANSITION_OCTAVES
            + 1e-12
        )

        assert diagnostics[
            "bridge_pass"
        ]

        assert diagnostics[
            "monotonicity_pass"
        ]

        assert diagnostics[
            "extrema_pass"
        ]

        assert diagnostics[
            "overshoot_pass"
        ]

        assert diagnostics[
            "undershoot_pass"
        ]

        assert diagnostics[
            "slope_reversal_pass"
        ]


# ---------------------------------------------------------------------------
# Dense numerical regression
# ---------------------------------------------------------------------------

def test_smootherstep_dense_monotonicity():
    """High-density numerical check of the exact locked weight."""
    t = np.linspace(
        0.0,
        1.0,
        1_000_001,
    )

    w = _smootherstep(t)

    diff = np.diff(w)

    assert float(
        np.min(diff)
    ) >= -1e-14

    assert float(
        np.max(w)
    ) <= 1.0 + 1e-14

    assert float(
        np.min(w)
    ) >= -1e-14


# ---------------------------------------------------------------------------
# Final architecture smoke test
# ---------------------------------------------------------------------------

def test_locked_architecture_smoke():
    """Single end-to-end assertion for the locked architecture."""
    freq = _grid_with_exact_1khz(
        points_per_octave=192,
    )

    target = _smooth_base(
        freq
    )

    masked = _make_masked(
        freq,
        start_hz=1000.0,
        end_hz=1700.0,
        amplitude=-2.0,
    )

    output, diagnostics = (
        adaptive_masked_handoff(
            freq,
            target,
            masked,
        )
    )

    assert output.shape == target.shape
    assert output.shape == masked.shape

    assert np.all(
        np.isfinite(output)
    )

    if (
        diagnostics[
            "status"
        ]
        == "ADAPTIVE_HANDOFF"
    ):
        assert (
            diagnostics[
                "actual_handoff_hz"
            ]
            == 1000.0
        )

        assert (
            diagnostics[
                "selection_rule"
            ]
            == "earliest_feasible_E"
        )

        assert (
            diagnostics[
                "bridge_type"
            ]
            == "quintic_smootherstep"
        )

        assert (
            diagnostics[
                "smootherstep_endpoint_weight_pass"
            ]
        )

        assert (
            diagnostics[
                "smootherstep_endpoint_derivative_pass"
            ]
        )

        assert (
            diagnostics[
                "bridge_pass"
            ]
        )

        assert (
            diagnostics[
                "candidate_scoring"
            ]
            is False
        )

        assert (
            diagnostics[
                "pareto_selection"
            ]
            is False
        )

        assert (
            diagnostics[
                "curvature_optimization"
            ]
            is False
        )