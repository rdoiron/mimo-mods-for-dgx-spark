#!/bin/bash
# Drop redundant `function.name=null` from subsequent tool-call deltas.
#
# The qwen3xml_tool_parser emits DeltaFunctionCall(name=None, arguments="...")
# for every chunk after the first tool-call header. With pydantic
# `exclude_unset=True`, an EXPLICITLY-set None still serializes as `"name": null`
# in the JSON delta. OpenCode's AI SDK Zod validator (@ai-sdk/openai-compatible)
# rejects this on continuation tool-call chunks with:
#   Expected 'function.name' to be a string
#
# OpenAI's streaming spec only sets `function.name` in the FIRST tool-call
# delta; subsequent deltas only update `arguments`. By replacing
# `DeltaFunctionCall(name=None, arguments=...)` with
# `DeltaFunctionCall(arguments=...)`, the `name` field stays unset and
# pydantic's `exclude_unset=True` omits it from JSON entirely.
#
# Safe: `DeltaFunctionCall(name=function_name, arguments=...)` (the FIRST
# delta where name IS set) is NOT touched by the sed pattern.

set -e
VLLM_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"

# Idempotency check: both tool_parsers and serving.py
PARSER_OLD=$(grep -c 'DeltaFunctionCall(name=None, arguments=' "$VLLM_ROOT"/tool_parsers/qwen3xml_tool_parser.py 2>/dev/null || echo 0)
SERVING_OLD=$(grep -c 'name=original_fn.name if original_fn else None' "$VLLM_ROOT"/entrypoints/openai/chat_completion/serving.py 2>/dev/null || echo 0)
if [ "$PARSER_OLD" = "0" ] && [ "$SERVING_OLD" = "0" ]; then
    echo "[mimo-fix-tool-call-deltas] Patch already applied, skipping."
    exit 0
fi

for f in "$VLLM_ROOT"/tool_parsers/*.py; do
    [ -f "$f.bak-pre-tool-call-deltas" ] || cp "$f" "$f.bak-pre-tool-call-deltas"
done

# Same-line: DeltaFunctionCall(name=None, arguments=...) → DeltaFunctionCall(arguments=...)
sed -i 's/DeltaFunctionCall(name=None, arguments=/DeltaFunctionCall(arguments=/g' "$VLLM_ROOT"/tool_parsers/*.py
# Multi-line: when name=None is on its own indented line within a multi-line
# DeltaFunctionCall(\n    name=None, arguments=...\n) construction. Match the
# pattern `\s*name=None, arguments=` at the start of an indented line.
sed -i 's/^\(\s\+\)name=None, arguments=/\1arguments=/' "$VLLM_ROOT"/tool_parsers/*.py

# Patch serving.py's _create_remaining_args_delta — the FINAL tool-call delta
# (with finish_reason="tool_calls") passes `name=original_fn.name if
# original_fn else None` which emits `"name":null` when None. OpenAI spec only
# sets name in the first delta; subsequent deltas should omit it. The final
# wrap-up delta with empty/remaining args should NOT include name.
SERVING="$VLLM_ROOT/entrypoints/openai/chat_completion/serving.py"
[ -f "$SERVING.bak-pre-tool-call-deltas" ] || cp "$SERVING" "$SERVING.bak-pre-tool-call-deltas"
python3 <<PYEOF
import re
fp = "$SERVING"
content = open(fp).read()

# Step 1: fix the name=null bug (only if not already done)
old1 = (
    "function=DeltaFunctionCall(\n"
    "                        name=original_fn.name if original_fn else None,\n"
    "                        arguments=remaining_call,\n"
    "                    ),"
)
new1 = "function=DeltaFunctionCall(arguments=remaining_call),"
if old1 in content:
    content = content.replace(old1, new1)
    print("[mimo-fix-tool-call-deltas] step 1: name=None removed from _create_remaining_args_delta")

# Step 2: make _create_remaining_args_delta a no-op.
# The qwen3xml_tool_parser's streamed_args_for_tool[index] tracker is off-by-
# one (frequently missing the trailing '}'), so the wrap-up delta produces
# '}}' at the end of arguments, breaking JSON.parse on the client. If the
# model already emitted complete JSON, we don't need the wrap-up; if it
# didn't, we'd rather receive incomplete JSON than corrupted JSON.
old2 = """    @staticmethod
    def _create_remaining_args_delta(
        delta_message: DeltaMessage,
        remaining_call: str,
        index: int,
    ) -> DeltaMessage:
        \"\"\"
        Create a delta message for remaining tool arguments, preserving
        id/type/name from the original delta.
        \"\"\"
        original_tc = next(
            (tc for tc in delta_message.tool_calls if tc.index == index),
            None,
        )
        original_fn = original_tc.function if original_tc else None
        return DeltaMessage(
            tool_calls=[
                DeltaToolCall(
                    index=index,
                    id=original_tc.id if original_tc else None,
                    type=original_tc.type if original_tc else None,
                    function=DeltaFunctionCall(arguments=remaining_call),
                )
            ]
        )"""
new2 = """    @staticmethod
    def _create_remaining_args_delta(
        delta_message: DeltaMessage,
        remaining_call: str,
        index: int,
    ) -> DeltaMessage:
        # MOD (4x-spark-cluster): no-op. qwen3xml_tool_parser.streamed_args_for_tool
        # is off-by-one (drops trailing '}'), so this wrap-up reliably duplicates
        # the closing brace and corrupts the JSON on the client side. Skipping
        # it leaves the client's accumulated JSON intact.
        return delta_message"""
if old2 in content:
    content = content.replace(old2, new2)
    open(fp, "w").write(content)
    print("[mimo-fix-tool-call-deltas] step 2: _create_remaining_args_delta turned into no-op")
elif "MOD (4x-spark-cluster): no-op." in content:
    print("[mimo-fix-tool-call-deltas] step 2: already no-op")
    open(fp, "w").write(content)
else:
    print("[mimo-fix-tool-call-deltas] WARN: _create_remaining_args_delta no-op pattern not matched")
    open(fp, "w").write(content)
PYEOF

# Drop bytecode cache so Python re-imports
find "$VLLM_ROOT/tool_parsers" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Verify
REMAINING=$(grep -c 'name=None, arguments=' "$VLLM_ROOT"/tool_parsers/*.py 2>/dev/null | awk -F: '{s+=$NF} END {print s+0}')
PATCHED_QWEN3XML=$(grep -c 'DeltaFunctionCall(arguments=' "$VLLM_ROOT"/tool_parsers/qwen3xml_tool_parser.py 2>/dev/null || true)
echo "[mimo-fix-tool-call-deltas] qwen3xml_tool_parser.py DeltaFunctionCall(arguments=) sites: $PATCHED_QWEN3XML"
echo "[mimo-fix-tool-call-deltas] Remaining 'name=None, arguments=' across tool_parsers: $REMAINING (expect 0)"

if [ "$REMAINING" != "0" ]; then
    echo "[mimo-fix-tool-call-deltas] WARN: some sites not patched"
fi

echo "[mimo-fix-tool-call-deltas] Step 3: dedupe immediate-duplicate \"}\" emits in parser"
# qwen3xml_tool_parser has auto-close paths (lines ~134-199) plus SAX
# EndElementHandler — both can fire _end_element("function") for the same
# `</function>` close, producing two `arguments="}"` deltas back-to-back
# for the same index. Client accumulates them as "}}" → JSON parse fails.
# Patch _emit_delta to track last emit per index and drop immediate
# duplicates of identical single-char arguments=}.
PARSER="$VLLM_ROOT/tool_parsers/qwen3xml_tool_parser.py"
[ -f "$PARSER.bak-pre-dedupe" ] || cp "$PARSER" "$PARSER.bak-pre-dedupe"
python3 <<PYEOF
fp = "$PARSER"
content = open(fp).read()
old = '''    def _emit_delta(self, delta: DeltaMessage):
        """Emit Delta response (streaming output)"""
        self.deltas.append(delta)'''
new = '''    def _emit_delta(self, delta: DeltaMessage):
        """Emit Delta response (streaming output)"""
        # MOD (4x-spark-cluster): dedupe immediate-duplicate arg fragments
        # per-index. The parser has auto-close paths that can fire
        # _end_element("function") right next to a SAX-triggered one, both
        # of which emit arguments="}" — yielding "}}" on the client which
        # breaks JSON.parse. Track last (index, fragment) and drop the
        # immediate duplicate.
        if not hasattr(self, "_mod_last_emit_per_index"):
            self._mod_last_emit_per_index = {}
        try:
            for tc in (delta.tool_calls or []):
                idx = tc.index
                fn = tc.function
                if fn is not None and fn.name in (None, ""):
                    args = fn.arguments
                    if args and args == self._mod_last_emit_per_index.get(idx):
                        return  # drop dup
                    self._mod_last_emit_per_index[idx] = args
                elif fn is not None and fn.name not in (None, ""):
                    # new tool-call header; reset dedup memory for that idx
                    self._mod_last_emit_per_index[idx] = None
        except Exception:
            pass
        self.deltas.append(delta)'''
if old in content:
    content = content.replace(old, new)
    open(fp, "w").write(content)
    print("[mimo-fix-tool-call-deltas] step 3: _emit_delta dedup applied")
elif "MOD (4x-spark-cluster): dedupe immediate-duplicate" in content:
    print("[mimo-fix-tool-call-deltas] step 3: already applied")
else:
    print("[mimo-fix-tool-call-deltas] WARN: _emit_delta pattern not matched")
PYEOF

echo "[mimo-fix-tool-call-deltas] Patch applied successfully."
