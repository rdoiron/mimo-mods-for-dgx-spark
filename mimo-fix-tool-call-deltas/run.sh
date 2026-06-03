#!/bin/bash
# Drop redundant `function.name=null` from subsequent tool-call deltas (STEP 1 only).
#
# ───────────────────────────────────────────────────────────────────────────
# STEPS 2 & 3 RETIRED 2026-06-02. They were symptom workarounds for the `}}`
# tool-call JSON corruption (step 2 = no-op _create_remaining_args_delta;
# step 3 = _emit_delta dedup of duplicate `}`). That corruption is now fixed at
# the ROOT by `mimo-clear-function-name-pr42969` (backport of vLLM PR #42969):
# the parser left `current_function_name` set after a </function> close, so the
# next <tool_call>'s `current_function_open or current_function_name` guard
# re-fired _end_element("function") -> spurious `}` -> `}}` / "not well-formed".
# Clearing the name removes the spurious emit, so the dedup/no-op are redundant.
# Verified clean on multi tool-call probes after the root fix (2026-06-02).
#
# STEP 1 (below) REMAINS — a separate OpenAI-streaming-spec issue #42969 does
# NOT touch: the parser emits DeltaFunctionCall(name=None, arguments=...) on
# continuation chunks; pydantic exclude_unset still serializes an explicitly-set
# None as `"name": null`, which strict client validators (OpenCode AI-SDK Zod /
# Droid) reject with "Expected 'function.name' to be a string". Replacing
# `DeltaFunctionCall(name=None, arguments=...)` with `DeltaFunctionCall(arguments
# =...)` leaves name unset so pydantic omits it. The FIRST delta (name set) is
# untouched by the sed pattern.
# ───────────────────────────────────────────────────────────────────────────

set -e
VLLM_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"

# Idempotency check: both tool_parsers and serving.py
PARSER_OLD=$(grep -c 'DeltaFunctionCall(name=None, arguments=' "$VLLM_ROOT"/tool_parsers/qwen3xml_tool_parser.py 2>/dev/null || echo 0)
SERVING_OLD=$(grep -c 'name=original_fn.name if original_fn else None' "$VLLM_ROOT"/entrypoints/openai/chat_completion/serving.py 2>/dev/null || echo 0)
if [ "$PARSER_OLD" = "0" ] && [ "$SERVING_OLD" = "0" ]; then
    echo "[mimo-fix-tool-call-deltas] step 1 already applied, skipping."
    exit 0
fi

for f in "$VLLM_ROOT"/tool_parsers/*.py; do
    [ -f "$f.bak-pre-tool-call-deltas" ] || cp "$f" "$f.bak-pre-tool-call-deltas"
done

# Same-line + multi-line: DeltaFunctionCall(name=None, arguments=...) → DeltaFunctionCall(arguments=...)
sed -i 's/DeltaFunctionCall(name=None, arguments=/DeltaFunctionCall(arguments=/g' "$VLLM_ROOT"/tool_parsers/*.py
sed -i 's/^\(\s\+\)name=None, arguments=/\1arguments=/' "$VLLM_ROOT"/tool_parsers/*.py

# serving.py _create_remaining_args_delta: drop name=None from the final wrap-up delta.
SERVING="$VLLM_ROOT/entrypoints/openai/chat_completion/serving.py"
[ -f "$SERVING.bak-pre-tool-call-deltas" ] || cp "$SERVING" "$SERVING.bak-pre-tool-call-deltas"
python3 <<PYEOF
fp = "$SERVING"
content = open(fp).read()
old1 = (
    "function=DeltaFunctionCall(\n"
    "                        name=original_fn.name if original_fn else None,\n"
    "                        arguments=remaining_call,\n"
    "                    ),"
)
new1 = "function=DeltaFunctionCall(arguments=remaining_call),"
if old1 in content:
    content = content.replace(old1, new1)
    open(fp, "w").write(content)
    print("[mimo-fix-tool-call-deltas] step 1: name=None removed from _create_remaining_args_delta")
elif "DeltaFunctionCall(arguments=remaining_call)" in content:
    print("[mimo-fix-tool-call-deltas] step 1: serving.py already patched")
else:
    print("[mimo-fix-tool-call-deltas] WARN: step 1 serving.py pattern not matched")
PYEOF

# Drop bytecode cache so Python re-imports
find "$VLLM_ROOT/tool_parsers" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Verify
REMAINING=$(grep -c 'name=None, arguments=' "$VLLM_ROOT"/tool_parsers/*.py 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
echo "[mimo-fix-tool-call-deltas] Remaining 'name=None, arguments=' across tool_parsers: $REMAINING (expect 0)"
[ "$REMAINING" != "0" ] && echo "[mimo-fix-tool-call-deltas] WARN: some sites not patched"
echo "[mimo-fix-tool-call-deltas] step 1 applied (steps 2/3 retired; root fix = mimo-clear-function-name-pr42969)."
