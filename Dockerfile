# SurviveCity v2.1 — training container, sized for a 24 GB GPU.
#
# Primary target: HF Spaces with A10G hardware (24 GB VRAM, $1.00-1.50/hr).
# Also runs as-is on a 30 GB DGX/A100 — pass `--no-4bit --num-generations 12
# --max-completion-length 512` to use the bigger budget.
#
# Backup of the previous "DGX-only / 30 GB / Ubuntu base / Unsloth-free
# stack" Dockerfile lives in Dockerfile.dgx.bak — that file is the one
# the team validated end-to-end on actual DGX hardware in April 2026.
# This file inherits its dependency-install section verbatim (so the
# four cascading dep failures from that history don't come back) and
# only changes the runtime/CMD section to:
#
#   1. fit 24 GB instead of 30 GB by default
#   2. work cleanly on HF Spaces (writable HF_HOME, EXPOSE 7860 so the
#      Space framework sees the container as healthy, lightweight
#      status sidecar, secrets-style HF_TOKEN env var, no hard-coded
#      tokens)
#   3. push checkpoints + metrics to the Hub model repo continuously,
#      so the 1 GB Spaces storage cap never matters
#
# DEPENDENCY HISTORY (unchanged from .bak — read this if you wonder why
# the dep section looks the way it does):
#
# We tried mirroring v1's Dockerfile.dgx (transformers 4.40.2 + peft 0.10.0
# + trl 0.8.6 + Unsloth git head + torchao 0.7.0) and hit FOUR cascading
# dependency failures in sequence:
#
#   1. peft >=0.13 (pulled in by Unsloth) demands torchao >= 0.16.0 in
#      `is_torchao_available()`, but torch 2.5 can only have torchao <= 0.7.
#      → Fixed by uninstalling torchao entirely.
#   2. trl 0.15+ chain-imports `mergekit` from callbacks.py unconditionally.
#      → Fixed by `pip install mergekit`.
#   3. trl 0.15+ also chain-imports `llm_blender` from judges.py.
#      → Tried `pip install llm-blender`, but...
#   4. ...the latest `llm_blender` is built against an older transformers API
#      (`from transformers.utils.hub import TRANSFORMERS_CACHE` — that
#      constant was renamed to `HF_HUB_CACHE` in modern transformers).
#      `llm_blender` simply does not import in our env.
#
# DECISION: drop Unsloth, pin trl==0.15.2, install mergekit, STUB OUT
# llm_blender as a 1-class shim, leave torchao uninstalled. Same as .bak.

FROM nvidia/cuda:12.1.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    # HF Spaces convention: writable cache dirs under /tmp (free tier) or
    # /data (paid tier with persistent storage). We default to /tmp so the
    # container also boots cleanly outside of HF Spaces.
    HF_HOME=/tmp/hf_cache \
    TRANSFORMERS_CACHE=/tmp/hf_cache \
    HF_HUB_ENABLE_HF_TRANSFER=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3.11-dev python3.11-venv python3-pip git curl gcc \
    && ln -sf /usr/bin/python3.11 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org

# torch first (large download — isolate it for cache hits on iterations)
RUN pip install --no-cache-dir \
    torch==2.5.1 torchvision \
    --index-url https://download.pytorch.org/whl/cu121 \
    --trusted-host download.pytorch.org

# Core training stack — modern set with GRPOTrainer.
# trl 0.15.x is the first widely-deployed trl with GRPOTrainer at the
# `from trl import GRPOTrainer` top-level. transformers 4.46.3 + peft 0.13.2
# + accelerate 1.1.1 + datasets 3.1.0 are the release-window-aligned versions
# trl 0.15 was tested against.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    "transformers==4.46.3" \
    "accelerate==1.1.1" \
    "peft==0.13.2" \
    "datasets==3.1.0" \
    "trl==0.15.2"

# 4-bit quantisation — the DEFAULT path on a 24 GB box. Without it the
# bf16 base alone takes ~8 GB and forward/backward + KV cache pushes peak
# above 24 GB at num_generations=8 / max_completion=384. With nf4 + double
# quant the base shrinks to ~3 GB and peak fits inside ~22 GB measured.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    "bitsandbytes>=0.43"

# trl 0.15+ chain-imports `mergekit` from callbacks.py unconditionally.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    "mergekit"

# Patch transformers.utils.hub.py to add the legacy `TRANSFORMERS_CACHE`
# constant. Some libs (incl. llm_blender) still reference the old name.
RUN HUB_PY=/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py \
 && grep -q '^TRANSFORMERS_CACHE' "$HUB_PY" \
    || printf '\n# Legacy alias for libs that haven'\''t caught up to the rename (added by Dockerfile.dgx)\nTRANSFORMERS_CACHE = HF_HUB_CACHE if "HF_HUB_CACHE" in dir() else "~/.cache/huggingface/hub"\n' >> "$HUB_PY"

# Stub out `llm_blender` instead of installing the real one (heavy dep tree
# that conflicts with our pinned transformers/peft stack — see the history
# block at top of file). Our GRPO code never instantiates a judge; we only
# need `import llm_blender` to succeed at module load.
RUN pip uninstall -y llm-blender 2>/dev/null || true \
 && rm -rf /usr/local/lib/python3.11/dist-packages/llm_blender \
 && mkdir -p /usr/local/lib/python3.11/dist-packages/llm_blender \
 && printf '%s\n' \
    '"""Stub llm_blender — survivecity-v2 does not use judge functionality."""' \
    '' \
    'class Blender:' \
    '    """Stub. The real llm_blender package conflicts with our pinned stack."""' \
    '    def __init__(self, *args, **kwargs):' \
    '        raise RuntimeError(' \
    '            "llm_blender.Blender is stubbed in survivecity-v2. "' \
    '            "Our GRPO pipeline does not use judges. If you need this, "' \
    '            "build a separate image with the full llm-blender package."' \
    '        )' \
    '    def loadranker(self, *args, **kwargs): pass' \
    '    def loadfuser(self, *args, **kwargs): pass' \
    '    def rank(self, *args, **kwargs): raise RuntimeError("llm_blender stubbed")' \
    '    def fuse(self, *args, **kwargs): raise RuntimeError("llm_blender stubbed")' \
    > /usr/local/lib/python3.11/dist-packages/llm_blender/__init__.py

# Belt-and-suspenders: ensure torchao is NOT installed.
RUN pip uninstall -y torchao || true

# Full-path import smoke test — same as .bak.
RUN python -c "\
import importlib.util; \
import torch; \
import transformers; \
import peft; \
import datasets; \
import accelerate; \
import bitsandbytes; \
import huggingface_hub; \
import trl; \
import mergekit; \
import llm_blender; \
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig; \
from transformers.processing_utils import Unpack; \
from transformers.quantizers.auto import AutoHfQuantizer; \
from transformers.utils.hub import TRANSFORMERS_CACHE; \
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training, PeftModel; \
from trl import GRPOTrainer, GRPOConfig; \
ta = importlib.util.find_spec('torchao'); \
print('torch       ', torch.__version__); \
print('transformers', transformers.__version__); \
print('peft        ', peft.__version__); \
print('datasets    ', datasets.__version__); \
print('accelerate  ', accelerate.__version__); \
print('trl         ', trl.__version__); \
print('bitsandbytes', bitsandbytes.__version__); \
print('mergekit    ', 'present'); \
print('llm_blender ', 'present (TRANSFORMERS_CACHE alias =', TRANSFORMERS_CACHE, ')'); \
print('torchao     ', 'absent (intended)' if ta is None else 'PRESENT (unexpected)'); \
assert ta is None, 'torchao should be uninstalled at build time'; \
print('FULL TRAINING IMPORT CHAIN OK')"

# Plotting / hub / progress / mem watchdog. hf_transfer accelerates Hub
# uploads (LoRA push every step adds up to ~750 MB over 15 steps, hf_transfer
# parallelises that and roughly halves the per-step push time).
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    wandb "matplotlib>=3.8" "numpy>=1.24" tensorboard \
    "huggingface_hub>=0.20" "hf_transfer>=0.1.6" \
    "tqdm>=4.66" "psutil>=5.9"

# Env runtime deps (server side; harmless even when running training-only).
# We keep these because the container ALSO ships a tiny FastAPI status
# sidecar on port 7860 for HF Spaces — see entrypoint at the bottom.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org \
    "pydantic>=2.0" "fastapi>=0.104" "uvicorn[standard]>=0.24"

# Source — copy after deps so iterative edits don't bust the dep cache
COPY survivecity_v2_env/ survivecity_v2_env/
COPY server/             server/
COPY training/           training/
COPY openenv.yaml        ./
COPY pyproject.toml      ./

# Runtime dirs — checkpoints, eval, transcripts, hf cache.
# On HF Spaces these live on the ephemeral 1 GB layer; we push to the Hub
# repo every step + every 5 min so a Space restart never loses progress.
RUN mkdir -p /app/checkpoints /app/eval_results /app/results /tmp/hf_cache \
    && chmod -R 777 /app/checkpoints /app/eval_results /app/results /tmp/hf_cache

# HF Spaces standard listening port. Our training has no HTTP component,
# but the Space framework marks the container "running" only if SOMETHING
# answers on the port. Our entrypoint runs a tiny FastAPI status sidecar
# at /, /health, /progress to satisfy that — see the CMD at the bottom.
EXPOSE 7860

# -----------------------------------------------------------------------
# Entrypoint
#
# Behaviour:
#   1. If HUGGINGFACE_TOKEN (or HF_TOKEN) is set, pass it through and add
#      `--push-to-hub`. Token comes from the Space's Secrets UI — never
#      hard-coded into the image.
#   2. Pick HUB_MODEL_ID from env var, falling back to
#      "${SPACE_OWNER}/zombiee-v2" if HF Spaces sets SPACE_OWNER, else a
#      sentinel string the user must override.
#   3. Start the OpenEnv FastAPI server in the background on port 7860 so
#      the Space UI shows "running". Logs come from the foreground
#      training process; the server is a no-op for the user.
#   4. Run training in the foreground. When it exits cleanly the
#      container exits with code 0 → HF Space stops billing.
#
# All training flags can be overridden by passing args to `docker run ...`.
# Everything below the `EXTRA_ARGS=` line is the v2.1 24 GB-safe default
# you can change with `EXTRA_ARGS="--num-generations 12 --no-4bit ..."`.
# -----------------------------------------------------------------------
ENV HUB_MODEL_ID="" \
    EXTRA_ARGS=""

CMD ["bash", "-c", "\
echo '=== SurviveCity v2.1 training (24 GB-safe defaults) ==='; \
echo '------------------------------------------------------'; \
echo 'GPU:'; nvidia-smi -L 2>/dev/null || echo '  (no GPU detected — training will be slow / fail)'; \
echo 'CPU/mem:'; nproc 2>/dev/null; free -h 2>/dev/null | head -2; \
echo '------------------------------------------------------'; \
\
# Pick the HF token from either common env-var name. \
HF_TOKEN_VALUE=\"${HUGGINGFACE_TOKEN:-${HF_TOKEN:-}}\"; \
export HUGGINGFACE_TOKEN=\"$HF_TOKEN_VALUE\"; \
\
# Pick the Hub repo id (env var first, fallback to <space-owner>/zombiee-v2). \
HUB_REPO=\"${HUB_MODEL_ID:-}\"; \
if [ -z \"$HUB_REPO\" ] && [ -n \"${SPACE_AUTHOR_NAME:-}\" ]; then \
    HUB_REPO=\"${SPACE_AUTHOR_NAME}/zombiee-v2\"; \
fi; \
\
# Decide whether to push. Push only when both token + repo are present. \
if [ -n \"$HF_TOKEN_VALUE\" ] && [ -n \"$HUB_REPO\" ]; then \
    PUSH_FLAGS=\"--push-to-hub --hub-model-id $HUB_REPO --safety-push-every-min 5\"; \
    echo \"Hub push: ENABLED -> $HUB_REPO (safety push every 5 min)\"; \
else \
    PUSH_FLAGS=\"\"; \
    echo 'Hub push: DISABLED (set HUGGINGFACE_TOKEN + HUB_MODEL_ID Space secrets to enable).'; \
fi; \
\
# Tiny status sidecar — keeps the Space \"running\" while training works. \
# Failures here are non-fatal: training is the canonical workload. \
( \
  python -m uvicorn server.app:app --host 0.0.0.0 --port 7860 \
    > /app/status_server.log 2>&1 & \
  echo $! > /tmp/status_server.pid \
) || echo 'WARN: status sidecar failed to start; training continues.'; \
\
echo '------------------------------------------------------'; \
echo 'Launching training. Logs follow.'; \
echo '------------------------------------------------------'; \
\
# v2.1 24 GB-safe defaults: \
#   --num-generations 8           (vs DGX 12) \
#   --max-completion-length 384   (vs DGX 512) \
#   --rollout-limit 60            (heuristic-rollout horizon) \
#   --safety-push-every-min 5     (lightweight metrics push to Hub) \
#   default 4-bit nf4 quant       (DROP --no-4bit on 24 GB; needed only on >=30 GB) \
exec python -m training.train \
    --model-name Qwen/Qwen2.5-3B-Instruct \
    --max-steps 15 \
    --save-steps 3 \
    --save-total-limit 6 \
    --output-dir /app/checkpoints \
    --num-generations 8 \
    --grad-accum-steps 4 \
    --max-completion-length 384 \
    --rollout-limit 60 \
    --lora-r 32 \
    --lora-alpha 64 \
    --max-seq-length 4096 \
    --report-to tensorboard \
    --resume-from-checkpoint auto \
    $PUSH_FLAGS \
    $EXTRA_ARGS \
"]
