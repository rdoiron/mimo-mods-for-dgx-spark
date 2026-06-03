# MiMo-V2.5 Mods for DGX Spark Clusters

A collection of runtime patches and modifications for running MiMo-V2.5 on NVIDIA DGX Spark clusters (sm_121a/GB10) with vLLM.

## Overview

These mods address compatibility issues encountered when deploying MiMo-V2.5 on DGX Spark hardware, particularly:

1. **CUTLASS block-FP8 kernel bypass** – Forces Triton fallback for sm_120/sm_121 devices
2. **Chat template fixes** – Corrects tool-call parsing and reasoning content handling
3. **MOE config aliasing** – Links RTX PRO 6000 Blackwell configs to GB10
4. **Reasoning content emission** – Renames `reasoning` field to `reasoning_content` for OpenAI compatibility
5. **Tool call delta fixes** – Removes problematic `name=null` emissions (step 1; steps 2/3 retired — superseded by the PR #42969 root fix, #9)
6. **Claude thinking close recognition** – Improves reasoning end detection for Claude-style thinking
7. **Empty tool call stripping** – Filters out incomplete tool call deltas
8. **Claude XML leakage scrubbing** – Strips Anthropic-leaked closing tags from streamed content/reasoning
9. **Tool-call parser root fix (PR #42969)** – Clears `current_function_name` on function close; fixes multi/sequential tool-call corruption (`}}` / "not well-formed (invalid token)")

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

# 5. Tool call delta fixes (removes name=null only; steps 2/3 retired)
bash mods/mimo-fix-tool-call-deltas/run.sh

# 6. Claude thinking close recognition — SKIP (defunct, see section 6 below)
#    bash mods/mimo-recognize-claude-thinking-close/run.sh

# 7. Empty tool call stripping (filters incomplete tool calls)
bash mods/mimo-strip-empty-tool-calls/run.sh

# 8. Claude XML leakage scrubbing (strips </parameter>, </invoke>, math-italic antml)
bash mods/mimo-scrub-claude-xml-leakage/run.sh

# 9. Tool-call parser root fix — clears current_function_name on close (PR #42969).
#    Apply AFTER #5; it's the root fix that let steps 2/3 of #5 be retired.
bash mods/mimo-clear-function-name-pr42969/run.sh
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

**Purpose**: Strips `name=null` from continuation tool-call deltas (OpenAI streaming-spec compliance)  
**Problem**: Parser emits `DeltaFunctionCall(name=None, ...)` on continuation chunks; pydantic `exclude_unset` still serializes the explicit `None` as `"name": null`, which strict client validators (OpenCode AI-SDK Zod / Droid) reject with "Expected 'function.name' to be a string"  
**Solution**: Replaces `DeltaFunctionCall(name=None, arguments=...)` with `DeltaFunctionCall(arguments=...)` so `name` stays unset and is omitted  
**Note (2026-06-02)**: This mod is now **step 1 only**. Its former steps 2 (no-op `_create_remaining_args_delta`) and 3 (`_emit_delta` dedup) were symptom workarounds for the `}}` corruption and have been **retired** — that corruption is fixed at the root by **#9 (`mimo-clear-function-name-pr42969`)**. Verified: multi/sequential tool calls reconstruct to valid JSON with steps 2/3 removed.  

### 6. mimo-recognize-claude-thinking-close — ⚠️ DEFUNCT, kept for reference

**Status**: **NOT INCLUDED** in production `startmimo.sh`. Disabled 2026-05-23. Kept in the repo so anyone re-attempting this approach has the prior art and the failure analysis.

**Original purpose**: Teach vLLM's qwen3 reasoning parser about Anthropic-style thinking closers (canonical `</think>`, math-italic `</𝑎𝑛𝑡𝑚𝑙:thinking>`, ASCII namespaced `</thinking>`) that MiMo-V2.5 emits as artifacts of training-data leakage, so they cleanly terminate the reasoning channel instead of leaking into visible content.

**Why it doesn't work as written**: V8 added a guard requiring a `<think>` opener inside the `input_ids` slice before declaring reasoning ended (to avoid false positives on chat-template tool-call examples). But vLLM's parser orchestrator (`vllm/parser/abstract_parser.py:parse_delta`) calls `is_reasoning_end` on the **per-delta token slice** — a handful of tokens at a time, *never* containing the request's `<think>` opener. The guard therefore always returns `False`, `state.reasoning_ended` stays `False` forever, the orchestrator never hands off to the tool parser, and Droid/OpenCode tool calls come through as raw XML inside the content channel.

**What we use instead**:
1. `repetition_penalty=1.2` in `--override-generation-config` (HF discussion #6, S1quence's reply 3) — keeps the Unicode-contamination thought-loop from running away in the first place
2. `mimo-scrub-claude-xml-leakage` (below) — strips any residual `</parameter>`, `</invoke>`, math-italic antml-close tags that the sampling penalty doesn't catch, from the streamed SSE `delta.content` / `delta.reasoning_content`

**If you want to revive it**: the right fix is at the orchestrator boundary — either feed cumulative `input_ids` into `is_reasoning_end` so the opener stays visible, or move closer-detection into `is_reasoning_end_streaming` (where the parser already keeps cumulative `current_text`). The text-based suffix-scan primitives in `_mod_extract_reasoning_streaming` are fine; only the `is_reasoning_end` hook is broken.

### 7. mimo-strip-empty-tool-calls

**Purpose**: Drops incomplete tool-call deltas (no name, no arguments) before the SSE wire
**Problem**: Wrap-up deltas with `id`+`type` but no `name` violate strict AI-SDK Zod schemas (OpenCode breaks)
**Solution**: Patches `ChatCompletionStreamResponse.model_dump_json` to filter empty entries from `delta.tool_calls`

### 8. mimo-scrub-claude-xml-leakage

**Purpose**: Strip Claude-grammar XML closers and Anthropic-thinking close-tag residue that leak into `delta.content` and `delta.reasoning_content`
**Problem**: MiMo-V2.5's training mix contains Anthropic-conversation leakage. The model intermittently emits `</parameter>`, `</invoke>`, `</function_calls>` into the content channel and corrupted math-italic forms like `</𝑎𝑛𝑡𝑚𝑙:thinking>`, `</𝑎𝑛𝑡𝑚�>`, `</𝑎�>` (U+FFFD) into the reasoning channel
**Solution**: Wraps `ChatCompletionStreamResponse.model_dump_json` with a tag-scrubbing serializer. Maintains a per-`(request_id, field)` tail buffer so tags split across SSE chunks still get caught. Tags inside legitimate `<tool_call>` blocks are unaffected — by the time content reaches this hook, the tool-call parser has already consumed real tool calls
**Known limit**: This will also strip these tag literals from code/documentation output. Acceptable trade-off for our use cases (coding agents, not Claude-format docs)

### 9. mimo-clear-function-name-pr42969

**Purpose**: Root-cause fix for tool-call corruption on multi/sequential tool calls (backport of vLLM [PR #42969](https://github.com/vllm-project/vllm/pull/42969))  
**Problem**: `qwen3xml_tool_parser`'s function-end handler sets `current_function_open = False` but does **not** clear `current_function_name`. On the next `<tool_call>`, the guard `if self.current_function_open or self.current_function_name:` re-fires `_end_element("function")` → a spurious `}` (the `}}` corruption) and desyncs the (reset) expat parser so the following `</tool_call>` throws `not well-formed (invalid token)`. Surfaced in NVIDIA forum thread 370459 #8; confirmed as the live trigger for multi-`read` corruption (e.g. a coding agent reading several files in one turn).  
**Solution**: Clear `self.current_function_name = None` when the function closes.  
**Impact**: Fixes both the `}}` bug and the multi-call `not well-formed` parse failures with one change, and made steps 2/3 of #5 redundant (now retired). Apply **after** #5.  
**Upstream**: PR #42969 was open at backport time; replace this mod with the upstream commit once it merges.  
