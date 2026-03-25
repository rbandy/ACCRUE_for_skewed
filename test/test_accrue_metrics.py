import math
import pytest

torch = pytest.importorskip("torch")

from src.asymmLaplace_accrue_torch import (
    CRPS_asymmLaplace_ana_torch,
    analytical_RS_torch as al_analytical_rs,
    calc_eta_torch as al_calc_eta,
    get_avg_CRPS_torch as al_avg_crps,
)
from src.twoPieceGauss_accrue_torch import (
    analytical_RS_torch as tpg_analytical_rs,
    calc_eta_torch as tpg_calc_eta,
    get_avg_CRPS_torch as tpg_avg_crps,
    norm_CDF,
    norm_PDF,
    CRPS_TPG_ana_torch,
)


def _manual_rs(sorted_eta: torch.Tensor) -> torch.Tensor:
    n = len(sorted_eta)
    temp = 0.0
    for i in range(n - 1):
        temp += (i + 1) ** 2 * (sorted_eta[i + 1] - sorted_eta[i])
    temp += n**2 * (1 - sorted_eta[-1])
    return torch.sum(sorted_eta**2) / n + temp / (n**2)


def test_normal_helpers_match_known_values() -> None:
    x0 = torch.tensor(0.0)
    sd = torch.tensor(1.0)

    assert torch.isclose(norm_CDF(x0, sd), torch.tensor(0.5), atol=1e-6)
    assert torch.isclose(norm_PDF(x0, sd), torch.tensor(1 / math.sqrt(2 * math.pi)), atol=1e-6)


def test_tpg_average_crps_matches_elementwise_mean() -> None:
    sigma1 = torch.tensor([1.0, 0.9, 1.2])
    sigma2 = torch.tensor([1.1, 1.3, 0.8])
    error = torch.tensor([-0.2, 0.3, -0.1])

    expected = torch.mean(
        torch.stack([CRPS_TPG_ana_torch(sigma1[i], sigma2[i], error[i]) for i in range(len(error))])
    )
    actual = tpg_avg_crps(sigma1, sigma2, error)

    assert torch.isclose(actual, expected, atol=1e-6)


def test_tpg_eta_sorted_and_rs_consistent() -> None:
    sigma1 = torch.tensor([0.8, 1.0, 1.4, 0.7])
    sigma2 = torch.tensor([1.2, 0.9, 1.1, 1.6])
    error = torch.tensor([-0.5, 0.2, 0.0, 1.0])

    eta = tpg_calc_eta(error, sigma1, sigma2)

    assert torch.all(eta[:-1] <= eta[1:])
    assert torch.all(eta >= 0)
    assert torch.all(eta <= 1)
    assert torch.isclose(tpg_analytical_rs(error, sigma1, sigma2), _manual_rs(eta), atol=1e-6)


def test_asymm_laplace_crps_is_symmetric_when_kappa_is_one() -> None:
    kappa = torch.tensor(1.0)
    pos = CRPS_asymmLaplace_ana_torch(kappa, torch.tensor(0.75), lam=torch.tensor(1.5))
    neg = CRPS_asymmLaplace_ana_torch(kappa, torch.tensor(-0.75), lam=torch.tensor(1.5))

    assert torch.isclose(pos, neg, atol=1e-6)


def test_asymm_laplace_average_and_rs_consistency() -> None:
    kappa = torch.tensor([0.8, 1.0, 1.5, 1.2])
    error = torch.tensor([-0.3, 0.1, 0.7, -0.4])
    lam = torch.tensor([1.2, 1.0, 1.3, 0.9])

    expected = torch.mean(
        torch.stack(
            [CRPS_asymmLaplace_ana_torch(kappa[i], error[i], lam=lam[i]) for i in range(len(error))]
        )
    )
    assert torch.isclose(al_avg_crps(kappa, error, lam), expected, atol=1e-6)

    eta_sorted = al_calc_eta(error, kappa, lam).values
    assert torch.all(eta_sorted[:-1] <= eta_sorted[1:])
    assert torch.all(eta_sorted >= 0)
    assert torch.all(eta_sorted <= 1)
    assert torch.isclose(al_analytical_rs(error, kappa, lam), _manual_rs(eta_sorted), atol=1e-6)
