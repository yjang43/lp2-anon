#!/bin/bash
# Joint TD-JEPA training (encoder, predictor and quasimetric head together) with the upstream code, unmodified:
#   https://github.com/HKBU-KnowComp/Temporal-Distance-JEPA @ b4c17ca
# Recipe as published (10 epochs, the given seed, bf16) except batch 64 x gradient accumulation 2 (= 128),
# because batch 128 does not fit on a 24 GB GPU with the 8-frame windows.
# Datasets are the ones our world models were trained on (config/train/data/*.yaml).
#
# Usage:  SEED=3072 GPUS="0 1" TASKS="tworoom cube" bash scripts/train_td_jepa_joint.sh
#   TD_JEPA_DIR  path to the upstream clone (default: ./Temporal-Distance-JEPA)
#   STABLEWM_HOME  as for the rest of the code; checkpoints land in
#                  $STABLEWM_HOME/checkpoints/<task>/td_jepa_train/seed$SEED/weights_epoch_{1..10}.pt
# GPUS and TASKS pair up in order; one task per GPU.
set -u
: "${SEED:?set SEED}" "${GPUS:?set GPUS}" "${TASKS:?set TASKS}" "${STABLEWM_HOME:?set STABLEWM_HOME}"
TD_JEPA_DIR=${TD_JEPA_DIR:-$PWD/Temporal-Distance-JEPA}
LOG=${LOG:-$PWD/logs/td_jepa_seed$SEED}; mkdir -p "$LOG"
cd "$TD_JEPA_DIR"

# tag -> upstream config | data | dataset file | checkpoint dir
declare -A T=(
[tworoom]="tworoom_train|tworoom|tworoom.h5|tworooms"
[umaze]="tworoom_train|tworoom|d4rl_pointmaze_u_maze_random.h5|pointmaze"
[medium]="tworoom_train|tworoom|d4rl_pointmaze_medium_random.h5|pointmaze_medium"
[cube]="ogb_train|ogb|ogbench/cube_single_expert.h5|cube"
[pusht]="pusht_train|pusht|pusht_expert_train.h5|pusht"
[reacher]="dmc_train|dmc|dmc/reacher_random.h5|reacher"
)

g=($GPUS); t=($TASKS)
[ ${#g[@]} -eq ${#t[@]} ] || { echo "GPUS and TASKS differ in length"; exit 1; }

for i in "${!t[@]}"; do
  tag=${t[$i]}; gpu=${g[$i]}
  IFS='|' read cfg data dsname ckdir <<< "${T[$tag]}"
  (
    export CUDA_VISIBLE_DEVICES=$gpu
    python train.py --config-name=$cfg data=$data "data.dataset.name=$dsname" \
      variant=td_jepa seed=$SEED output_model_name=$ckdir/td_jepa_train/seed$SEED \
      wandb.enabled=false planning_eval.enabled=false num_workers=6 \
      loader.batch_size=64 trainer.accumulate_grad_batches=2 \
      "~data.dataset.rdcc_nbytes" "~data.dataset.rdcc_w0" \
      hydra.run.dir=outputs/train_${tag}_seed$SEED > "$LOG/$tag.log" 2>&1
    echo "done $tag seed$SEED rc=$?" >> "$LOG/summary.txt"
  ) &
done
wait
