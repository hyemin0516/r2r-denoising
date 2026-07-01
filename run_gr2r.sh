#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/hyemin/denoising"
cd "${PROJECT_ROOT}" || { echo "Project root not found!"; exit 1; }

export CUDA_VISIBLE_DEVICES=0

# =================================================================
# 1. 실험 설정
# =================================================================
# CONFIG_FILE="configs/exp10_lr3e5_sv5.0.yaml"
# CONFIG_FILE="configs/exp7.yaml"
# VER="m0.99_ign_src_inter-gram_phase_fda_v2""
CONFIG_FILE="configs/exp10_lr3e5_sv5.0.yaml"
VER="exp_learnable-amp_v2"
CKPT_ROOT="/hyemin/denoising/ckpts/pcr2r"

# =================================================================
# 2. 핵심 변수만 읽어와서 깔끔한 RUN_NAME 생성
# =================================================================
# (참고) 이전 버전 출력 로직: lr{...}_wcons{...}_wuda{...}_${VER}
RUN_NAME=$(python3 -c "
from omegaconf import OmegaConf
cfg = OmegaConf.load('${CONFIG_FILE}')
print(f'${VER}')
")

OUT_DIR="${CKPT_ROOT}/${RUN_NAME}"
mkdir -p "${OUT_DIR}"

echo "🚀 Training Started: ${RUN_NAME}"
echo "📂 Output Dir: ${OUT_DIR}"

# =================================================================
# 3. 모델 실행
# =================================================================
python3 -m gr2r.train_cr2r \
  --config "${CONFIG_FILE}" \
  --wandb_project "pc-r2r-sidd-opt" \
  --ckpt_dir "${OUT_DIR}" \
  2>&1 | tee "${OUT_DIR}/train.log"

echo "✅ Training Finished!"