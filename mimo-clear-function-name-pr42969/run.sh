#!/bin/bash
# mimo-clear-function-name-pr42969 — root-cause fix (backport of vLLM PR #42969).
# In qwen3xml_tool_parser the function-end handler sets current_function_open=False
# but leaves current_function_name set. On the NEXT <tool_call>, the guard
# `if self.current_function_open or self.current_function_name:` re-fires
# _end_element("function") -> spurious '}' (the }} corruption) AND desyncs expat
# so the following </tool_call> throws "not well-formed (invalid token)",
# corrupting multi/sequential tool calls (e.g. several read calls in one turn).
# Fix: clear current_function_name when the function closes.
VLLM_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"
P="$VLLM_ROOT/tool_parsers/qwen3xml_tool_parser.py"

if grep -q "pr42969 root fix" "$P" 2>/dev/null; then
    echo "[mimo-clear-function-name-pr42969] already applied, skipping."
    exit 0
fi
[ -f "$P.bak-pre-clear-fn-name" ] || cp "$P" "$P.bak-pre-clear-fn-name"

python3 - "$P" <<'PYEOF'
import sys
fp = sys.argv[1]
s = open(fp).read()
old = '''            self.current_function_open = False

        elif name == "tool_call":'''
new = '''            self.current_function_open = False
            # MOD (4x-spark-cluster) pr42969 root fix: clear current_function_name on
            # function close. Leaving it set makes the next <tool_call> guard
            # (current_function_open or current_function_name) re-fire
            # _end_element("function") -> spurious "}" + </tool_call> "not well-formed"
            # parse failure that corrupts multi/sequential tool calls.
            self.current_function_name = None

        elif name == "tool_call":'''
if old not in s:
    print("[mimo-clear-function-name-pr42969] ERROR: anchor not found - NOT applied")
    sys.exit(1)
open(fp, "w").write(s.replace(old, new, 1))
print("[mimo-clear-function-name-pr42969] patched", fp)
PYEOF

find "$VLLM_ROOT/tool_parsers" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
echo "[mimo-clear-function-name-pr42969] done."
