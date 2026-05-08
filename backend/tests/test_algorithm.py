"""
Algorithm validation tests.

These are skipped until the real algorithm module is connected.
To activate, remove the @pytest.mark.skip decorator and import your module.
"""
import math

import pytest

# from algorithm.subag import run as subag_run


@pytest.mark.skip(reason="Algorithm not yet connected — remove skip when ready")
def test_pure_energy_convergence() -> None:
    """Final energy density should approach -E_inf as N grows."""
    from algorithm.subag import run as subag_run  # type: ignore[import]

    for N in (20, 40, 80):
        request_model = {"type": "pure_p_spin", "p": 3, "N": N, "seed": 42}
        batches = list(subag_run(request_model, n_steps=200))
        final_energy = batches[-1].snapshots[-1].energy_density
        E_inf = 2 * math.sqrt(2 / 3)  # p=3
        assert abs(final_energy + E_inf) < 0.2, (
            f"N={N}: final energy {final_energy:.4f} far from -E_inf={-E_inf:.4f}"
        )


@pytest.mark.skip(reason="Algorithm not yet connected — remove skip when ready")
def test_spectral_edge_satisfied() -> None:
    """lambda_min_tangent <= spectral_edge_target + eps at every step."""
    from algorithm.subag import run as subag_run  # type: ignore[import]

    request_model = {"type": "pure_p_spin", "p": 3, "N": 40, "seed": 1}
    batches = list(subag_run(request_model, n_steps=100))
    for batch in batches:
        for snap in batch.snapshots:
            assert snap.edge_target_satisfied, (
                f"Step {snap.j}: lambda_min={snap.lambda_min_tangent:.4f} "
                f"> edge={snap.spectral_edge_target:.4f}"
            )


@pytest.mark.skip(reason="Algorithm not yet connected — remove skip when ready")
def test_sphere_constraint() -> None:
    """||sigma||^2 / N should equal q at every step."""
    from algorithm.subag import run as subag_run  # type: ignore[import]

    request_model = {"type": "pure_p_spin", "p": 3, "N": 20, "seed": 5}
    batches = list(subag_run(request_model, n_steps=50))
    for batch in batches:
        for snap in batch.snapshots:
            norm_sq_over_N = sum(x * x for x in snap.sigma) / len(snap.sigma)
            assert abs(norm_sq_over_N - snap.q) < 1e-4, (
                f"Step {snap.j}: ||σ||²/N={norm_sq_over_N:.6f} != q={snap.q:.6f}"
            )
