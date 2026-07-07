#!/usr/bin/env bash
set -o pipefail

PROJECT_ROOT="/e-vepfs-01/ppdc/test/closeloop/intern/ych/lhr/code/vllm_qwen"
LOG_DIR="${PROJECT_ROOT}/log"
LOG_FILE="${LOG_DIR}/infer_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "${LOG_DIR}"

python3 annotate_reveallayer.py \
  --dataset_root /e-vepfs-01/ppdc/test/closeloop/intern/ych/lhr/data/RevealLayer-100K/Benchmark/RevealLayerBenchMark-200 \
  --metadata /e-vepfs-01/ppdc/test/closeloop/intern/ych/lhr/data/RevealLayer-100K/Benchmark/RevealLayerBenchMark-200/metaData.json \
  --output_json /e-vepfs-01/ppdc/test/closeloop/intern/ych/lhr/data/RevealLayer-100K/Benchmark/RevealLayerBenchMark-200/prompt_annotations.json \
  --model_path /e-vepfs-01/ppdc/test/closeloop/intern/ych/lhr/weight/Qwen/Qwen3.6-27B \
  --batch_size 1 \
  --limit 5 2>&1 | tee "${LOG_FILE}"
exit ${PIPESTATUS[0]}