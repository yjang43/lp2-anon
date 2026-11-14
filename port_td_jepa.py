"""Split a jointly trained TD-JEPA checkpoint into a LeWM and a head.

Upstream TD-JEPA (HKBU-KnowComp/Temporal-Distance-JEPA @ b4c17ca) saves one
``jepa.JEPA`` state dict: the LeWM modules (encoder, predictor,
action_encoder, projector, pred_proj) plus the quasimetric ``dist_head``. Its
module classes are the LeWM ones with the same parameter names, so the split
is a re-keying, not a conversion:

    <dst>/config.json, weights.pt          stable_worldmodel LeWM  -> policy=
    <dst>/td/config.json, td_epoch_N.pt    module.QuasimetricHead  -> cost.td_ckpt=

Planning then runs unchanged through eval.py, e.g.

    python eval.py --config-name=tworoom policy=<dst>/weights.pt \\
        cost=td cost.td_ckpt=<dst>/td/td_epoch_10.pt cost.alpha=0.0

Paths are relative to $STABLEWM_HOME/checkpoints, as in load_pretrained.

    python port_td_jepa.py --src tworooms/td_jepa_train/seed3072/weights_epoch_10.pt \\
        --dst tworooms/td_jepa_seed3072
"""

import argparse
import json
import re
from pathlib import Path

import torch
from stable_worldmodel.data.utils import get_cache_dir

LEWM_MODULES = ('encoder', 'predictor', 'action_encoder', 'projector', 'pred_proj')

# Upstream class -> stable_worldmodel class. Same code, same parameter names.
TARGETS = {
    'jepa.JEPA': 'stable_worldmodel.wm.lewm.LeWM',
    'module.ARPredictor': 'stable_worldmodel.wm.lewm.module.Predictor',
    'module.Embedder': 'stable_worldmodel.wm.lewm.module.Embedder',
    'module.MLP': 'stable_worldmodel.wm.lewm.module.MLP',
}
HEAD_TARGET = 'module.QuasimetricHead'  # lp2's copy, logic identical


def retarget(cfg):
    """Swap upstream ``_target_``s for stable_worldmodel ones, recursively."""
    if isinstance(cfg, dict):
        out = {k: retarget(v) for k, v in cfg.items()}
        if out.get('_target_') in TARGETS:
            out['_target_'] = TARGETS[out['_target_']]
        return out
    if isinstance(cfg, list):
        return [retarget(v) for v in cfg]
    return cfg


def split(src_cfg, state):
    """Return (lewm_cfg, lewm_state, head_cfg, head_state)."""
    # Refuse anything that is not the canonical td_jepa variant: other heads
    # or a Gaussian predictor have no LeWM / QuasimetricHead counterpart.
    assert src_cfg['_target_'] == 'jepa.JEPA', src_cfg['_target_']
    assert src_cfg.get('reach_head') is None, 'reach_head not supported'
    assert not src_cfg.get('gaussian_predictor', False), 'Gaussian predictor'
    head_cfg = dict(src_cfg['dist_head'])
    assert head_cfg['_target_'] == 'module.QuasimetricHead', head_cfg
    head_cfg['_target_'] = HEAD_TARGET

    lewm_cfg = retarget({k: src_cfg[k] for k in ('_target_', *LEWM_MODULES)})

    lewm_state, head_state = {}, {}
    for k, v in state.items():
        top = k.split('.', 1)[0]
        if top in LEWM_MODULES:
            lewm_state[k] = v
        elif top == 'dist_head':
            head_state[k[len('dist_head.'):]] = v
        else:
            raise KeyError(f'unexpected key {k}')
    return lewm_cfg, lewm_state, head_cfg, head_state


def save(run_dir, filename, cfg, state):
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, run_dir / filename)
    with open(run_dir / 'config.json', 'w') as f:
        json.dump(cfg, f, indent=2)
    print(f'wrote {run_dir / filename}')


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--src', required=True, help='upstream weights_epoch_N.pt')
    p.add_argument('--dst', required=True, help='output run name')
    args = p.parse_args()

    root = get_cache_dir(sub_folder='checkpoints')
    src = Path(root, args.src)
    epoch = int(re.fullmatch(r'weights_epoch_(\d+)\.pt', src.name).group(1))
    with open(src.parent / 'config.json') as f:
        src_cfg = json.load(f)
    state = torch.load(src, map_location='cpu')

    lewm_cfg, lewm_state, head_cfg, head_state = split(src_cfg, state)

    dst = Path(root, args.dst)
    save(dst, 'weights.pt', lewm_cfg, lewm_state)
    save(dst / 'td', f'td_epoch_{epoch}.pt', head_cfg, head_state)
    with open(dst / 'ported_from.json', 'w') as f:
        json.dump({'src': str(src), 'epoch': epoch}, f, indent=2)


if __name__ == '__main__':
    main()
