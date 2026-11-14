"""Action-conditioned noise around the fitted normal drift ("fixed A").

fit_act.py refits an unconstrained A together with the action term. Here A is
the normal drift (fit.py, unchanged); the noise is the empirical
covariance of its residual once the action term is removed:

    r     = z' - A z                         residual of A
    xi    = a - K z,   K = C_az C_zz^-1      action not predictable from z
    B     = Cov(r, xi) Cov(xi)^-1
    Sigma = Cov(r) - B Cov(xi) B^T = Cov(r - B xi)

With a = K z + xi and z' = A_c z + B a + eps, the action-free transition is
z' = (A_c + B K) z + B xi + eps, so the action term removes the
action-induced variance B Cov(xi) B^T and leaves the rest. Sigma is a
residual covariance of the data (a Schur complement), so it is positive
semidefinite, unlike the pinned I - A A^T - B Cov(xi) B^T, which assumes
Cov(z) = I and goes indefinite. xi is uncorrelated with z by construction,
so B does not depend on A and equals fit_act.py's B. Basis = eigenvectors of
Sigma, least noise first (ACTBasis, as fit_act.py).
"""

import hydra
import torch
from stable_worldmodel.wm.utils import load_pretrained

from fit_act import accumulate, load_pairs, save_basis


def residual_cov(cxx, cyx, cyy, A, d):
    """Cov(z' - A z) from the centred covariances of x = [z; a] and y = z'."""
    czz, cyz = cxx[:d, :d], cyx[:, :d]
    cov = cyy - A @ cyz.T - cyz @ A.T + A @ czz @ A.T
    return (cov + cov.T) / 2


def action_term(cxx, cyx, d):
    """B and B Cov(xi) B^T, the action-induced part of Cov(z' - A z)."""
    czz, cza, caa = cxx[:d, :d], cxx[:d, d:], cxx[d:, d:]
    cyz, cya = cyx[:, :d], cyx[:, d:]
    K = torch.linalg.solve(czz, cza).T  # (m, d)
    cov_xi = caa - K @ cza
    cov_xi = (cov_xi + cov_xi.T) / 2
    # Cov(z' - A z, xi) = C_ya - C_yz K^T: the A z part is uncorrelated with xi
    B = torch.linalg.solve(cov_xi, (cya - cyz @ K.T).T).T  # (d, m)
    induced = B @ cov_xi @ B.T
    return B, (induced + induced.T) / 2


def report(sigma, cov_r):
    eig = torch.linalg.eigvalsh(sigma)
    share = (cov_r.trace() - sigma.trace()) / cov_r.trace()
    print(f'act: eig min {eig[:3].tolist()}  negative {(eig < 0).sum().item()} / {len(eig)}'
          f'  share of tr(Cov r): {share:.4f}')


@hydra.main(version_base=None, config_path='./config/train', config_name='act_ours')
def run(cfg):
    cxx, cyx, cyy, d = accumulate(load_pairs(cfg))
    # eval() so a NormalDrift returns A* (static lines zeroed), as cost.LP2GoalMSE reads it
    A = load_pretrained(cfg.drift_ckpt).eval().weight.detach().double()
    cov_r = residual_cov(cxx, cyx, cyy, A, d)
    B, induced = action_term(cxx, cyx, d)
    sigma = cov_r - induced
    report(sigma, cov_r)
    save_basis(sigma, torch.cat([A, B], dim=1), d, cfg.output_model_name)


if __name__ == '__main__':
    run()
