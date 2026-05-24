# MiMo-V2.5 Mods for DGX Spark Clusters

A collection of runtime patches and modifications for running MiMo-V2.5 on NVIDIA DGX Spark clusters (sm_121a/GB10) with vLLM.

## Overview

These mods address compatibility issues encountered when deploying MiMo-V2.5 on DGX Spark hardware, particularly:

1. **CUTLASS block-FP8 kernel bypass** – Forces Triton fallback for sm_120/sm_121 devices
2. **Chat template fixes** – Corrects tool-call parsing and reasoning content handling
3. **MOE config aliasing** – Links RTX PRO 6000 Blackwell configs to GB10
4. **Reasoning content emission** – Renames `reasoning` field to `reasoning_content` for OpenAI compatibility
5. **Tool call delta fixes** – Removes problematic `name=null` emissions and empty tool calls
6. **Claude thinking close recognition** – Improves reasoning end detection for Claude-style thinking
7. **Empty tool call stripping** – Filters out incomplete tool call deltas

## Installation

Each mod lives in its own directory with a `run.sh` script. Apply them sequentially:

```bash
# From the spark-vllm-docker directory
cd /home/ryan/spark-vllm-docker

# Apply all MiMo mods
for mod in mods/mimo-*; do
    if [ -f "$mod/run.sh" ]; then
        echo "Applying $(basename $mod)..."
        bash "$mod/run.sh"
    fi
done
```

Or apply individually:

```bash
# 1. CUTLASS block-FP8 kernel bypass (critical for sm_121a)
bash mods/mimo-disable-cutlass-block-fp8-sm121/run.sh

# 2. Chat template fix (needed for tool calling)
bash mods/mimo-fix-tool-template/run.sh

# 3. MOE config aliasing (optional, improves MoE performance)
bash mods/mimo-alias-moe-configs-gb10/run.sh

# 4. Reasoning content emission (client compatibility)
bash mods/mimo-emit-reasoning-content/run.sh

# 5. Tool call delta fixes (removes name=null, deduplicates)
bash mods/mimo-fix-tool-call-deltas/run.sh

# 6. Claude thinking close recognition (better reasoning end detection)
bash mods/mimo-recognize-claude-thinking-close/run.sh

# 7. Empty tool call stripping (filters incomplete tool calls)
bash mods/mimo-strip-empty-tool-calls/run.sh
```

## Prerequisites

- **Hardware**: NVIDIA DGX Spark (GB10, sm_121a) cluster
- **Software**: vLLM 0.21.1rc1.dev30+ (tested with eugr's spark-vllm-docker)
- **Model**: MiMo-V2.5 (XiaomiMiMo/MiMo-V2.5, FP8)
- **Framework**: vLLM with Ray V2 executor + MTP N=1

## Mod Details

### 1. mimo-disable-cutlass-block-fp8-sm121

**Purpose**: Workaround for CUTLASS block-FP8 kernel rejection on sm_121a  
**Problem**: vLLM's `cutlass_scaled_mm_supports_block_fp8(cap)` returns `True` for capability >= 100, but the compiled `cutlass_scaled_mm_sm120` kernel rejects sm_121a at runtime  
**Solution**: Patches `CutlassFp8BlockScaledMMKernel.is_supported` to return `False` for sm_120/sm_121, forcing fallback to TritonFp8BlockScaledMMKernel  

### 2. mimo-fix-tool-template

**Purpose**: Fixes tool-call parsing in chat template  
**Problem**: Original template assumes `tool_call.arguments` is a dict, but OpenCode/Droid sends it as a string  
**Solution**: Adds `from_json` conversion for string arguments in the Jinja template  

### 3. mimo-alias-moe-configs-gb10

**Purpose**: Optimizes MoE kernel dispatch for GB10  
**Problem**: vLLM ships no tuned fused_moe configs for NVIDIA_GB10  
**Solution**: Symlinks RTX PRO 6000 Blackwell Server Edition configs (also sm_120) as NVIDIA_GB10  

### 4. mimo-emit-reasoning-content

**Purpose**: Makes reasoning content OpenAI-spec compliant  
**Problem**: vLLM uses `reasoning` field (vLLM-specific), but clients expect `reasoning_content`  
**Solution**: Renames `reasoning` to `reasoning_content` across all response protocols  

### 5. mimo-fix-tool-call-deltas

**Purpose**: Cleans up malformed tool-call streaming deltas  
**Problem**: Parser emits `name=null` in continuation deltas, confusing strict validators  
**Solution**: Strips `name=None` from tool-call deltas and deduplicates immediate duplicate argument fragments  

### 6. mimo-recognize-claude-thinking-close

**Purpose**: Improves reasoning end detection for Claude-style thinking  
**Problem**: Naive detection incorrectly marks paired template examples as reasoning endings  
**Solution**: Implements pair-check heuristics for `</think>` and `
