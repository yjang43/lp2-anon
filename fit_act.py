"""Fit an action-conditioned linear transition on a frozen world model's
embeddings, and the planning basis it implies ("free A").

Nothing is trained. Ordinary least squares on (z_t, a_t, z_{t+1}):

    z' ~ A z + B a,        y_hat = A z + B a

a_t is the frameskip raw actions between z_t and z_{t+1}, flattened. The
noise is the residual once the action dependence is modeled,

    Sigma = Cov(z' - y_hat),

and the basis is its eigenvectors, least noise first.
"""

import os

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from stable_worldmodel.wm.utils import save_pretrained

from fit import embedding_cache
from module import ACTBasis


class EmbeddingActionPairs(torch.utils.data.Dataset):
    """(z_t, z_{t+1}) from the embedding cache with a_t from the raw dataset.

    Both are loaded with the same num_steps and frameskip, so index i is the
    same clip in each; this is checked rather than assumed.
    """

    def __init__(self, emb, raw):
        assert len(emb) == len(raw), (len(emb), len(raw))
        assert np.array_equal(np.asarray(emb.lengths), np.asarray(raw.lengths))
        assert list(emb.clip_indices) == list(raw.clip_indices)
        self.emb, self.raw = emb, raw

    def __len__(self):
        return len(self.emb)

    def __getitem__(self, i):
        return {'emb': self.emb[i]['emb'], 'action': self.raw[i]['action']}


def accumulate(loader):
    """Centred covariances of x = [z_t; a_t] and y = z_{t+1}, in float64.

    Streams sums, so the pairs never sit in memory together. Returns
    C_xx, C_yx, C_yy and the embedding dim d (x[:d] is z, x[d:] is a).
    """
    n, sx, sy, sxx, syx, syy = 0, None, None, None, None, None
    for batch in loader:
        emb = batch['emb'].double()  # (B, 2, d)
        act = torch.nan_to_num(batch['action'][:, 0].double(), 0.0)  # (B, m)
        x = torch.cat([emb[:, 0], act], dim=1)
        y = emb[:, 1]
        if sx is None:
            d = y.shape[1]
            sx, sy = x.new_zeros(x.shape[1]), y.new_zeros(d)
            sxx = x.new_zeros(x.shape[1], x.shape[1])
            syx = x.new_zeros(d, x.shape[1])
            syy = x.new_zeros(d, d)
        n += x.shape[0]
        sx += x.sum(dim=0)
        sy += y.sum(dim=0)
        sxx += x.T @ x
        syx += y.T @ x
        syy += y.T @ y
    mx, my = sx / n, sy / n
    cxx = (sxx - n * torch.outer(mx, mx)) / (n - 1)
    cyx = (syx - n * torch.outer(my, mx)) / (n - 1)
    cyy = (syy - n * torch.outer(my, my)) / (n - 1)
    return cxx, cyx, cyy, d


def fit_ols(cxx, cyx):
    """[A B] = C_yx C_xx^{-1}, and Cov(y_hat) = [A B] C_xx [A B]^T."""
    W = torch.linalg.solve(cxx, cyx.T).T  # (d, d + m)
    explained = W @ cxx @ W.T
    return W, (explained + explained.T) / 2


def residual_noise(cyy, explained):
    """Sigma = Cov(z' - y_hat) = Cov(z') - Cov(y_hat) for least squares."""
    return cyy - explained


def basis(sigma):
    """Eigenvectors of Sigma, least noise first (orthonormal columns)."""
    noise, U = torch.linalg.eigh(sigma)  # ascending
    return U, noise


def load_pairs(cfg):
    """Train loader of (z_t, z_{t+1}) with a_t, same split as PCA and SFA."""
    cache_dir = os.environ.get('LOCAL_DATASET_DIR', None)
    cache_name = embedding_cache(cfg)
    frameskip = cfg.data.dataset.frameskip
    common = dict(
        transform=None, cache_dir=cache_dir, num_steps=2, frameskip=frameskip
    )

    print(
        f'Loading embedding cache "{cache_name}" and actions from '
        f'"{cfg.data.dataset.name}" (pairs, frameskip={frameskip})'
    )
    emb = swm.data.load_dataset(
        cache_name, keys_to_load=['emb'], keys_to_cache=['emb'], **common
    )
    raw = swm.data.load_dataset(
        cfg.data.dataset.name,
        keys_to_load=['action'],
        keys_to_cache=['action'],
        **common,
    )
    dataset = EmbeddingActionPairs(emb, raw)

    rnd_gen = torch.Generator().manual_seed(cfg.seed)
    train_set, _ = spt.data.random_split(
        dataset,
        lengths=[cfg.train_split, 1 - cfg.train_split],
        generator=rnd_gen,
    )
    loader_cfg = {**cfg.loader, 'shuffle': False, 'drop_last': False}
    return torch.utils.data.DataLoader(train_set, **loader_cfg)


def save_basis(sigma, W, d, run_name):
    """Save the eigenvectors of Sigma (least noise first) with A and B."""
    U, noise = basis(sigma)
    head = ACTBasis(d, W.shape[1] - d)
    head.weight.copy_(U.float())
    head.noise.copy_(noise.float())
    head.A.copy_(W[:, :d].float())
    head.B.copy_(W[:, d:].float())
    save_pretrained(
        head,
        run_name=run_name,
        config={
            '_target_': 'module.ACTBasis',
            'dim': d,
            'action_dim': head.action_dim,
        },
        filename='basis.pt',
    )
    print(f'noise {noise[:4].tolist()} ... {noise[-2:].tolist()}')
    print(f'wrote {run_name}/basis.pt')


@hydra.main(version_base=None, config_path='./config/train', config_name='act')
def run(cfg):
    cxx, cyx, cyy, d = accumulate(load_pairs(cfg))
    W, explained = fit_ols(cxx, cyx)
    save_basis(residual_noise(cyy, explained), W, d, cfg.output_model_name)


if __name__ == '__main__':
    run()
