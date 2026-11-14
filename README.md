# Latent Projection for Latent Planning (anonymous code)

Code for the submission: the core method (fit the state dynamics A of a frozen latent world model, plan with the
goal distance measured on the k least uncertain directions of I − AAᵀ) and the experiments added in the author
response (action-conditioned fits, TD-JEPA frozen head and joint training). Everything is post hoc: the encoder,
predictor and planner stay frozen.

## Files

| File | What it does |
|---|---|
| `module.py` | `NormalDrift` (the structured A of the paper) and its ablation arms; `PCABasis`, `ACTBasis`, `QuasimetricHead` |
| `fit.py` | fit A on cached embeddings of a frozen LeWM (multi-step least squares, Eqs. 7–8) |
| `cost.py` | planning costs: `LP2GoalMSE` (ours), `RandomGoalMSE`, `ACTGoalMSE` (action-conditioned), `TemporalDistance` (TD-JEPA) |
| `eval.py` | CEM planning with a chosen cost; writes the success rate |
| `fit_pca.py` | PCA baseline basis |
| `fit_act.py` | action-conditioned fit, **free A**: OLS z′ ≈ Az + Ba, basis = eigenvectors of the residual covariance |
| `fit_act_ours.py` | action-conditioned fit, **fixed A**: our fitted A plus the action term, basis = eigenvectors of Cov(z′ − Az − Bξ) |
| `fit_td.py` | TD-JEPA quasimetric head trained on the **frozen** LeWM |
| `port_td_jepa.py` | splits a **jointly trained** TD-JEPA checkpoint (upstream code) into a LeWM and a head, so it plans through `eval.py` |
| `scripts/train_td_jepa_joint.sh` | joint TD-JEPA training with the upstream repository, unmodified |
| `config/` | Hydra configs: `train/` for the fits, `eval/` for planning (`<task>.yaml` default setting, `<task>_hard.yaml` harder setting) |

## Setup

Python ≥ 3.10 and a CUDA GPU. `pip install -r requirements.txt` (see the note there on stable-worldmodel).
Set `STABLEWM_HOME` to a directory holding `datasets/` and `checkpoints/<task>/lewm/{config.json,weights.pt}`.
The TwoRoom, PushT, Reacher and OGBench-Cube datasets and world models are those released with LeWM
(https://github.com/lucas-maes/le-wm). The PointMaze datasets and world models will be released with the
camera-ready version. All paths below are relative to `$STABLEWM_HOME/checkpoints`.

## 1. Main experiment: fit A, plan with the projected cost

```bash
# fit A (encoder and predictor frozen); seeds 0-4 give the 5 fits
python fit.py --config-name=normal_drift data=tworoom pretrained_ckpt=tworooms/lewm/weights.pt \
    seed=0 output_model_name=tworooms/lewm/normal_drift_seed0

# plan (harder setting) on the k least uncertain directions
python eval.py --config-name=tworoom_hard policy=tworooms/lewm/weights.pt \
    cost=lp2 cost.k=16 cost.drift_ckpt=tworooms/lewm/normal_drift_seed0/drift_epoch_50.pt seed=1
```

Baselines change only the cost: `cost=l2`; `cost=random cost.k=16 cost.seed=0`;
`cost=pca cost.k=8 cost.drift_ckpt=tworooms/lewm/pca/pca.pt` (after `python fit_pca.py data=tworoom`).

Protocol of the response: k ∈ {2, 4, …, 192} is picked on each validation seed (0, 10, 20, 30, 40) for each fit,
and the picked k is tested on test seeds 1, 11, 21, 31, 41 (50 episodes each). `seed=` sets the evaluation seed.
Add `+video=false` to skip rendering videos.

## 2. Action-conditioned fits (response to the question on conditioning A on the action)

```bash
# free A: unconstrained OLS with the action
python fit_act.py --config-name=act data=tworoom pretrained_ckpt=tworooms/lewm/weights.pt \
    output_model_name=tworooms/lewm/act_residual

# fixed A: our fitted A, plus the action term
python fit_act_ours.py --config-name=act_ours data=tworoom pretrained_ckpt=tworooms/lewm/weights.pt \
    drift_ckpt=tworooms/lewm/normal_drift_seed0/drift_epoch_50.pt output_model_name=tworooms/lewm/act_ours_seed0

# plan with either basis
python eval.py --config-name=tworoom_hard policy=tworooms/lewm/weights.pt \
    cost=act cost.k=128 cost.act_ckpt=tworooms/lewm/act_ours_seed0/basis.pt seed=1
```

## 3. TD-JEPA: frozen head and joint training

Frozen head (only the quasimetric head is trained, on the frozen LeWM; seeds 0-4):

```bash
python fit_td.py --config-name=td data=tworoom pretrained_ckpt=tworooms/lewm/weights.pt \
    seed=0 output_model_name=tworooms/lewm/td_seed0
python eval.py --config-name=tworoom_hard policy=tworooms/lewm/weights.pt \
    cost=td cost.td_ckpt=tworooms/lewm/td_seed0/td_epoch_10.pt cost.alpha=0.0 seed=1
```

Joint training (encoder, predictor and head trained together with the upstream code, unmodified), then the
same planner and evaluation:

```bash
SEED=3072 GPUS="0" TASKS="tworoom" bash scripts/train_td_jepa_joint.sh
python port_td_jepa.py --src tworooms/td_jepa_train/seed3072/weights_epoch_10.pt --dst tworooms/td_jepa_seed3072
python eval.py --config-name=tworoom_hard policy=tworooms/td_jepa_seed3072/weights.pt \
    cost=td cost.td_ckpt=tworooms/td_jepa_seed3072/td/td_epoch_10.pt cost.alpha=0.0 seed=1
```

`cost.alpha` blends the learned distance with latent ℓ2: 0 is the pure TD-JEPA distance, 1 is pure ℓ2.

## License

MIT (see `LICENSE`).
