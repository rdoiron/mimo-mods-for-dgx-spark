#!/bin/bash
# Patch CutlassFp8BlockScaledMMKernel.is_supported to return False on sm_120/sm_121.
#
# Root cause: vLLMs cutlass_scaled_mm_supports_block_fp8(capability) returns True
# for capability >= 100, but the compiled cutlass_scaled_mm_sm120 kernel rejects
# sm_121a (GB10/DGX Spark) at runtime via CUTLASS_CHECK(gemm_op.can_implement(args))
# at csrc/.../cutlass_gemm_caller.cuh:51. This causes the model to fail in
# profile_run with: RuntimeError: cutlass_gemm_caller ... Invalid status
#
# Skipping the kernel here makes choose_scaled_mm_linear_kernel fall through to
# TritonFp8BlockScaledMMKernel (the universal fallback), which JIT-compiles
# per-arch via Triton and works on sm_121a.

set -e
MOD_DIR="$(dirname "$0")"
VLLM_ROOT="/usr/local/lib/python3.12/dist-packages"

echo "[disable-cutlass-block-fp8-sm121] Applying patch..."

if ! patch --dry-run --forward -p1 -d "$VLLM_ROOT" < "$MOD_DIR/skip-sm121.patch" > /tmp/patch-dryrun-cutlass.log 2>&1; then
    if grep -qE "already applied|previously applied|Reversed" /tmp/patch-dryrun-cutlass.log; then
        echo "[disable-cutlass-block-fp8-sm121] Patch already applied, skipping."
        exit 0
    else
        echo "[disable-cutlass-block-fp8-sm121] Dry-run failed:"
        cat /tmp/patch-dryrun-cutlass.log
        exit 1
    fi
fi

patch --forward --batch -p1 -d "$VLLM_ROOT" < "$MOD_DIR/skip-sm121.patch"
echo "[disable-cutlass-block-fp8-sm121] Patch applied successfully."
grep -n "CUTLASS block-FP8 sm_120 kernel rejects sm_121a" "$VLLM_ROOT/vllm/model_executor/kernels/linear/scaled_mm/cutlass.py" || true
