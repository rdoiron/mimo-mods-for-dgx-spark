#!/bin/bash
# Alias RTX PRO 6000 Blackwell Server Edition fused_moe configs as NVIDIA_GB10.
#
# vLLM ships per-(device, shape) tuned Triton MoE configs in
# vllm/model_executor/layers/fused_moe/configs/. There are NO configs for
# NVIDIA_GB10 (the DGX Spark GPU), so the Triton kernel falls back to default
# tile/warp/stages params -- "Performance might be sub-optimal".
#
# RTX PRO 6000 Blackwell Server Edition is sm_120 (consumer Blackwell), the
# closest cousin to GB10 (sm_121a). Aliasing its configs is a reasonable
# starting point until proper GB10 tuning runs.
#
# This mod symlinks each RTX_PRO_6000_Blackwell_Server_Edition config to an
# NVIDIA_GB10-named file in the same dir. Triton kernel will then load these.

set -e
CONF_DIR="/usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/configs"

if [ ! -d "$CONF_DIR" ]; then
    echo "[alias-moe-configs] $CONF_DIR not found, skipping."
    exit 0
fi

count=0
for src in "$CONF_DIR"/*device_name=NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition*; do
    [ -f "$src" ] || continue
    dst=$(basename "$src" | sed 's/NVIDIA_RTX_PRO_6000_Blackwell_Server_Edition/NVIDIA_GB10/')
    full_dst="$CONF_DIR/$dst"
    if [ -e "$full_dst" ]; then
        echo "[alias-moe-configs] Already aliased: $dst"
    else
        ln -s "$(basename "$src")" "$full_dst"
        echo "[alias-moe-configs] Aliased: $dst"
        count=$((count + 1))
    fi
done

echo "[alias-moe-configs] Created $count NVIDIA_GB10 aliases (from RTX PRO 6000 Blackwell)."
