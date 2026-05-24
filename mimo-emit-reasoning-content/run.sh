#!/bin/bash
# Patch vLLM to emit `reasoning_content` (OpenAI / DeepSeek-R1 standard)
# instead of `reasoning` (vLLM-specific field name) in chat completion responses
# — both streaming SSE chunks and non-streaming final messages.
#
# Background:
# - vLLM's DeltaMessage (entrypoints/openai/engine/protocol.py) and ChatMessage
#   (entrypoints/openai/chat_completion/protocol.py) define a `reasoning` field
#   marked in the source as "vLLM-specific fields that are not in OpenAI spec".
# - All major OpenAI-compatible clients (OpenCode @ai-sdk/openai-compatible,
#   Droid generic-chat-completion-api, OpenWebUI, DeepSeek-R1 SDK) expect
#   `reasoning_content`. Without this rename, clients can't render the thinking
#   UI and the raw `<think>` block may leak into the visible content area.
#
# Patches applied:
#   1. engine/protocol.py:           DeltaMessage.reasoning → reasoning_content
#   2. chat_completion/protocol.py:  ChatMessage.reasoning  → reasoning_content
#   3. chat_completion/serving.py:   delta_message.reasoning →
#                                    delta_message.reasoning_content
#                                    reasoning=reasoning kwarg →
#                                    reasoning_content=reasoning
#   4. chat_completion/batch_serving.py: same kwarg rename
#   5. responses/streaming_events.py: delta_message.reasoning → ...content
#
# Idempotent: detects existing patch via the renamed engine.DeltaMessage field.

set -e
VLLM_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"
ENG_PROTO="$VLLM_ROOT/entrypoints/openai/engine/protocol.py"
CC_PROTO="$VLLM_ROOT/entrypoints/openai/chat_completion/protocol.py"
SERVING="$VLLM_ROOT/entrypoints/openai/chat_completion/serving.py"
BATCH="$VLLM_ROOT/entrypoints/openai/chat_completion/batch_serving.py"
EVENTS="$VLLM_ROOT/entrypoints/openai/responses/streaming_events.py"

for f in "$ENG_PROTO" "$CC_PROTO" "$SERVING" "$BATCH" "$EVENTS"; do
    if [ ! -f "$f" ]; then
        echo "[mimo-emit-reasoning-content] ERROR: $f not found"
        exit 1
    fi
done

# Idempotency check: skip when all 4 markers are present.
ENG_OK=$(grep -cE '^\s+reasoning_content: str \| None = None' "$ENG_PROTO" || true)
CC_OK=$(grep -cE '^\s+reasoning_content: str \| None = None' "$CC_PROTO" || true)
KWARG_OLD=$(grep -cE '\breasoning=reasoning\b' "$SERVING" "$BATCH" 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
PARSER_OLD=$(grep -cE 'DeltaMessage\(reasoning=' "$VLLM_ROOT"/reasoning/*.py 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')

if [ "$ENG_OK" = "1" ] && [ "$CC_OK" = "1" ] && [ "$KWARG_OLD" = "0" ] && [ "$PARSER_OLD" = "0" ]; then
    echo "[mimo-emit-reasoning-content] Patch already fully applied, skipping."
    exit 0
fi

echo "[mimo-emit-reasoning-content] Backing up originals..."
for f in "$ENG_PROTO" "$CC_PROTO" "$SERVING" "$BATCH" "$EVENTS"; do
    [ -f "$f.bak-pre-reasoning-content" ] || cp "$f" "$f.bak-pre-reasoning-content"
done

echo "[mimo-emit-reasoning-content] (1/5) DeltaMessage field rename in $ENG_PROTO"
sed -i 's/^\(\s\+\)reasoning: str | None = None/\1reasoning_content: str | None = None/' "$ENG_PROTO"

echo "[mimo-emit-reasoning-content] (2/5) ChatMessage field rename in $CC_PROTO"
sed -i 's/^\(\s\+\)reasoning: str | None = None/\1reasoning_content: str | None = None/' "$CC_PROTO"

echo "[mimo-emit-reasoning-content] (3/5) delta_message.reasoning references in $SERVING, $EVENTS"
sed -i 's/\bdelta_message\.reasoning\b\([^_]\|$\)/delta_message.reasoning_content\1/g' "$SERVING" "$EVENTS"

echo "[mimo-emit-reasoning-content] (4/5) reasoning=reasoning kwarg in $SERVING"
sed -i 's/\breasoning=reasoning\b/reasoning_content=reasoning/g' "$SERVING"

echo "[mimo-emit-reasoning-content] (5/6) reasoning=reasoning kwarg in $BATCH"
sed -i 's/\breasoning=reasoning\b/reasoning_content=reasoning/g' "$BATCH"

echo "[mimo-emit-reasoning-content] (6/6) reasoning= kwarg in $VLLM_ROOT/reasoning/*.py"
# All reasoning parsers construct `DeltaMessage(reasoning=delta_text)` or
# `DeltaMessage(reasoning=delta_text, content=...)` etc. With DeltaMessage's
# field now named `reasoning_content`, the old `reasoning=` kwarg silently
# slips through pydantic's `extra="allow"` and serializes as the wrong key.
# Narrow patterns only — broader replacements like `\.reasoning\b` will mangle
# `vllm.reasoning.basic_parsers` imports and break the modules entirely.
for f in "$VLLM_ROOT"/reasoning/*.py; do
    [ -f "$f.bak-pre-reasoning-content" ] || cp "$f" "$f.bak-pre-reasoning-content"
done
# Pattern A: DeltaMessage(reasoning=...) — first kwarg in constructor (same line)
sed -i 's/DeltaMessage(reasoning=/DeltaMessage(reasoning_content=/g' "$VLLM_ROOT"/reasoning/*.py
# Pattern B: , reasoning= — additional kwarg in constructor (same line)
sed -i 's/, reasoning=/, reasoning_content=/g' "$VLLM_ROOT"/reasoning/*.py
# Pattern C: leading-whitespace reasoning= — indented kwarg on its own line
# (multi-line DeltaMessage(\n    reasoning=...\n) form). The leading-space-only
# match avoids hitting variable assignments like `reasoning = delta_text[:i]`
# which have spaces around the `=`.
sed -i 's/^\(\s\+\)reasoning=/\1reasoning_content=/' "$VLLM_ROOT"/reasoning/*.py

echo "[mimo-emit-reasoning-content] (7) tool_parsers: delta.reasoning, r.reasoning, DeltaMessage(reasoning=, reasoning=None,"
# Backup tool_parsers
for f in "$VLLM_ROOT"/tool_parsers/*.py; do
    [ -f "$f.bak-pre-reasoning-content" ] || cp "$f" "$f.bak-pre-reasoning-content"
done
# Pattern D: `delta.reasoning` attribute access (qwen3xml_tool_parser:1295 bug)
sed -i -E 's/\bdelta\.reasoning\b([^_]|$)/delta.reasoning_content\1/g' "$VLLM_ROOT"/tool_parsers/*.py
# Pattern E: `r.reasoning` attribute access (cohere_command_tool_parser:65-66)
sed -i -E 's/\br\.reasoning\b([^_]|$)/r.reasoning_content\1/g' "$VLLM_ROOT"/tool_parsers/*.py
# Pattern F: `DeltaMessage(reasoning=` (same-line constructor kwarg)
sed -i 's/DeltaMessage(reasoning=/DeltaMessage(reasoning_content=/g' "$VLLM_ROOT"/tool_parsers/*.py
# Pattern G: `^\s+reasoning=None,` (indented kwarg in multi-line DeltaMessage
# call). Specifically `=None,` to avoid matching config-style kwargs in
# deepseekv4_tool_parser/qwen3coder_tool_parser/mistral_tool_parser which use
# `reasoning=get_enable_structured_outputs_in_reasoning()` or
# `reasoning=self.model_can_reason`.
sed -i 's/^\(\s\+\)reasoning=None,/\1reasoning_content=None,/' "$VLLM_ROOT"/tool_parsers/*.py

# Verify
ENG_OK=$(grep -cE '^\s+reasoning_content: str \| None = None' "$ENG_PROTO" || true)
CC_OK=$(grep -cE '^\s+reasoning_content: str \| None = None' "$CC_PROTO" || true)
SERV_DM=$(grep -c 'delta_message\.reasoning_content' "$SERVING" || true)
SERV_KW=$(grep -c 'reasoning_content=reasoning' "$SERVING" || true)
BATCH_KW=$(grep -c 'reasoning_content=reasoning' "$BATCH" || true)
EVENT_DM=$(grep -c 'delta_message\.reasoning_content' "$EVENTS" || true)
KWARG_OLD=$(grep -cE '\breasoning=reasoning\b' "$SERVING" "$BATCH" 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')

QWEN3_REASONING_OK=$(grep -c 'DeltaMessage(reasoning_content=' "$VLLM_ROOT/reasoning/qwen3_reasoning_parser.py" || true)

echo "[mimo-emit-reasoning-content] Verify:"
echo "  engine/protocol.py DeltaMessage renamed: $ENG_OK (expect 1)"
echo "  chat_completion/protocol.py ChatMessage renamed: $CC_OK (expect 1)"
echo "  serving.py delta_message.reasoning_content refs: $SERV_DM (expect >=2)"
echo "  serving.py reasoning_content=reasoning kwargs: $SERV_KW (expect >=10)"
echo "  batch_serving.py reasoning_content=reasoning kwargs: $BATCH_KW (expect >=1)"
echo "  streaming_events.py delta_message.reasoning_content refs: $EVENT_DM (expect >=5)"
echo "  qwen3_reasoning_parser.py DeltaMessage(reasoning_content=): $QWEN3_REASONING_OK (expect 1)"
echo "  remaining old 'reasoning=reasoning' kwargs: $KWARG_OLD (expect 0)"

if [ "$ENG_OK" != "1" ] || [ "$CC_OK" != "1" ] || [ "$KWARG_OLD" != "0" ]; then
    echo "[mimo-emit-reasoning-content] WARN: replacement counts unexpected; investigate."
fi

# Drop cached bytecode so Python re-imports the patched .py
find /usr/local/lib/python3.12/dist-packages/vllm/entrypoints -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

echo "[mimo-emit-reasoning-content] Patch applied successfully."
