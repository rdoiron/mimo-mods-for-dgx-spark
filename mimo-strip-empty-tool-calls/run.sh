#!/bin/bash
# Filter "empty" tool_call deltas before they hit the wire.
#
# Bug observed: when the model emits multiple parallel <tool_call> openers
# and generation ends before the LAST one gets its <function=...> name,
# the qwen3xml tool parser still tracks the half-emitted tool_call and
# generates a "wrap-up" delta with id+type set but no name and no
# arguments — e.g.
#   {"index":2,"id":"chatcmpl-tool-...","type":"function",
#    "function":{"arguments":""}}
#
# OpenAI's streaming spec lets `function.name` be absent in *continuation*
# deltas, but those continuation deltas should NOT include `id` and `type`
# either. A delta with id+type but no name confuses strict validators
# (OpenCode's @ai-sdk/openai-compatible Zod schema treats it as an
# "initial" delta and requires `function.name` to be a string → throws
# "Expected 'function.name' to be a string").
#
# Fix: at serialization time, drop any DeltaToolCall whose function.name
# is None AND function.arguments is None-or-empty. They carry no info
# anyway.
#
# Implementation: monkey-patch DeltaMessage.model_dump_json to filter out
# such empty entries before serializing.

set -e
VLLM_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"
# IMPORTANT: append to chat_completion/protocol.py (where the class is
# defined), NOT engine/protocol.py. Earlier versions of this mod appended
# to engine/protocol.py and called `from ...chat_completion.protocol
# import ChatCompletionStreamResponse` — that import fails silently during
# the cold module load (the try/except swallows the ImportError), the
# wrap never installs, and `model_dump_json` stays as `BaseModel`'s
# implementation. Verified by checking `__qualname__` at runtime.
# Appending here guarantees the class is in module globals when the
# installer runs.
PROTOCOL="$VLLM_ROOT/entrypoints/openai/chat_completion/protocol.py"

if grep -q "MOD_FILTER_EMPTY_TOOL_CALLS" "$PROTOCOL"; then
    echo "[mimo-strip-empty-tool-calls] Patch already applied, skipping."
    exit 0
fi

cp -n "$PROTOCOL" "$PROTOCOL.bak-pre-strip-empty-tool-calls"

# Append the monkey-patch at the end of the file (after class definitions).
cat >> "$PROTOCOL" << 'PYEOF'


# === MOD_FILTER_EMPTY_TOOL_CALLS ===
# Filter out tool_calls with no name and no arguments (incomplete wrap-up
# deltas that confuse strict client validators like AI-SDK).
#
# Why patch ChatCompletionStreamResponse.model_dump_json: when pydantic
# serializes a parent model, it walks nested models through its internal
# serializer — child.model_dump / child.model_dump_json are NOT called.
# So patching DeltaMessage.model_dump alone misses the streaming path.
# Patching the top-level chunk.model_dump_json catches all SSE chunks.
def _install_chunk_filter():
    # ChatCompletionStreamResponse is defined in THIS module, so by the
    # time this code at the bottom of the file runs, the class is in our
    # module globals. Do NOT re-import — that pattern fails during cold
    # load because the submodule is still mid-resolution.
    cls = globals().get("ChatCompletionStreamResponse")
    if cls is None:
        return

    _orig_dump_json = cls.model_dump_json

    def _filtered_chunk_dump_json(self, **kwargs):
        # Walk each choice's delta.tool_calls and drop empty entries.
        changed = False
        for choice in (self.choices or []):
            delta = getattr(choice, "delta", None)
            if delta is None or not getattr(delta, "tool_calls", None):
                continue
            filtered = []
            for tc in delta.tool_calls:
                fn = getattr(tc, "function", None)
                if fn is not None:
                    name_empty = fn.name in (None, "")
                    args_empty = fn.arguments in (None, "")
                    if name_empty and args_empty:
                        changed = True
                        continue
                filtered.append(tc)
            if filtered != list(delta.tool_calls):
                delta.tool_calls = filtered
                changed = True
        return _orig_dump_json(self, **kwargs)

    cls.model_dump_json = _filtered_chunk_dump_json

_install_chunk_filter()
PYEOF

echo "[mimo-strip-empty-tool-calls] Patch appended to $PROTOCOL"
find "$VLLM_ROOT/entrypoints" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Sanity-check: confirm the install actually ran at import time. A fresh
# interpreter cold-imports the patched module and inspects the wrapped
# method's qualname. If this fails, the patch source landed but the wrap
# didn't, and the mod is silently inert in production.
python3 -c "
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionStreamResponse
m = ChatCompletionStreamResponse.model_dump_json
qn = getattr(m, '__qualname__', '')
# Either name is fine — depends on apply order vs. mimo-scrub-claude-xml-leakage
if '_filtered_chunk_dump_json' not in qn and '_scrubbed_dump_json' not in qn:
    print(f'  install-time check: FAIL (model_dump_json is still {qn})')
    import sys; sys.exit(1)
print(f'  install-time check: OK (model_dump_json wrapped as {qn.rsplit(\".\", 1)[-1]})')
"
