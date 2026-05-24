#!/bin/bash
# Apply vLLM PR #41797 (triton-diff-kv attention backend) at runtime.
# Required for MiMo-V2.5 on sm_121a (GB10) — adds TRITON_ATTN_DIFFKV backend
# that the model selects when FA3 (Hopper-only) isn't available.
# 
# Source: https://github.com/vllm-project/vllm/pull/41797 (still open as of 2026-05-16)
# Approach matches CyberTen's recipe (NVIDIA forum thread post #37) — applied as a
# runtime overlay rather than baked into the image, so the image can keep using
# eugr's perf-validated prebuilt vLLM wheel.

set -e
MOD_DIR="$(dirname "$0")"
VLLM_ROOT="/usr/local/lib/python3.12/dist-packages"

echo "[mimo-pr-41797] Applying triton-diff-kv backend patch..."

if ! patch --dry-run --forward -p1 -d "$VLLM_ROOT" < "$MOD_DIR/pr-41797.patch" > /tmp/patch-dryrun.log 2>&1; then
    if grep -qE 'already applied|previously applied|Reversed' /tmp/patch-dryrun.log; then
        echo "[mimo-pr-41797] Patch already applied, skipping."
        exit 0
    else
        echo "[mimo-pr-41797] Dry-run failed:"
        cat /tmp/patch-dryrun.log
        exit 1
    fi
fi

patch --forward --batch -p1 -d "$VLLM_ROOT" < "$MOD_DIR/pr-41797.patch"
echo "[mimo-pr-41797] Patch applied successfully."
ls -la "$VLLM_ROOT/vllm/v1/attention/backends/triton_attn_diffkv.py" "$VLLM_ROOT/vllm/v1/attention/ops/triton_unified_attention_diffkv.py"
