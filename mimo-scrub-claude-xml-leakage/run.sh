#!/bin/bash
# Mod: mimo-scrub-claude-xml-leakage
#
# Strip Claude-style XML closers and Anthropic reasoning closers that MiMo-V2.5
# leaks into `delta.content` and `delta.reasoning_content`.
#
# Observed in Droid sessions 2026-05-24 (sessions 24fc6ac5, 8ecc6fe2):
#   - `</parameter>`, `</invoke>`, `</function_calls>` appearing in either the
#     visible text channel OR mid-thinking. These belong to Claude Code's
#     XML tool-call grammar, NOT MiMo's `<tool_call><function=...>` format.
#     If they reach `delta.content`, MiMo's qwen3xml tool parser has already
#     declined to consume them — so they're leakage, not real tool calls.
#   - `</𝑎𝑛𝑡𝑚𝑙:thinking>` (math-italic Unicode antml close) and corrupted
#     short forms like `</𝑎𝑛𝑡𝑚�>` and `</𝑎�>` showing up mid-thinking.
#     These are not legitimate reasoning closers; they're Anthropic-leak
#     residue from MiMo's training mix.
#
# Hook: ChatCompletionStreamResponse.model_dump_json — same SSE-serialization
# point used by mimo-strip-empty-tool-calls. Wraps any prior patch.
#
# Buffering: tags can split across chunks. We keep a per-(request, field)
# tail buffer of up to MAX_PATTERN_LEN-1 chars whose suffix could still be the
# start of any scrub pattern. The buffer flushes when the choice has a
# finish_reason.
#
# Idempotent.

set -e
VLLM_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"
# IMPORTANT: append to chat_completion/protocol.py (where the class is
# defined), NOT engine/protocol.py. Appending to engine/protocol.py runs
# the installer too early — `from ...chat_completion.protocol import ...`
# fails during initial import resolution, the try/except swallows it,
# and the patch never lands. Appending here guarantees the class object
# exists when _install_scrub_filter() runs.
PROTOCOL="$VLLM_ROOT/entrypoints/openai/chat_completion/protocol.py"

if grep -q "MOD_SCRUB_CLAUDE_XML_LEAKAGE" "$PROTOCOL"; then
    echo "[mimo-scrub-claude-xml-leakage] Patch already applied, skipping."
    exit 0
fi

cp -n "$PROTOCOL" "$PROTOCOL.bak-pre-scrub-claude-xml-leakage"

cat >> "$PROTOCOL" << 'PYEOF'


# === MOD_SCRUB_CLAUDE_XML_LEAKAGE ===
# Strip Claude-grammar XML closers and Anthropic-thinking closer variants
# that the model leaks into delta.content / delta.reasoning_content.
# See mods/mimo-scrub-claude-xml-leakage/run.sh for full rationale.

import threading as _scrub_threading

_SCRUB_PATTERNS = (
    "</parameter>",
    "</invoke>",
    "</function_calls>",
    "<function_calls>",
    "</\U0001D44E\U0001D45B\U0001D461\U0001D45A\U0001D459:thinking>",
    "</\U0001D44E\U0001D45B\U0001D461\U0001D45A�>",
    "</\U0001D44E�>",
    "</thinking>",
)
_SCRUB_MAX_PAT = max(len(_p) for _p in _SCRUB_PATTERNS)
_SCRUB_STATE = {}
_SCRUB_LOCK = _scrub_threading.Lock()


def _scrub_field(text, request_id, field, final):
    """Combine carry-over tail + new text, strip patterns, hold back a
    suffix that could still start a pattern. Returns the safe-to-emit string
    (or None if empty)."""
    key = (request_id, field)
    with _SCRUB_LOCK:
        prev = _SCRUB_STATE.pop(key, "")
    buf = prev + (text or "")
    if not buf:
        return None
    for pat in _SCRUB_PATTERNS:
        if pat in buf:
            buf = buf.replace(pat, "")
    if final:
        return buf or None
    # Hold back the largest suffix of buf that is a non-empty prefix of any
    # scrub pattern. That suffix becomes the carry-over for the next chunk.
    hold = 0
    limit = min(_SCRUB_MAX_PAT - 1, len(buf))
    for plen in range(limit, 0, -1):
        suffix = buf[-plen:]
        if any(pat.startswith(suffix) for pat in _SCRUB_PATTERNS):
            hold = plen
            break
    if hold == 0:
        return buf or None
    emit = buf[:-hold]
    tail = buf[-hold:]
    with _SCRUB_LOCK:
        _SCRUB_STATE[key] = tail
    return emit or None


def _install_scrub_filter():
    # ChatCompletionStreamResponse is defined in THIS module, so by the
    # time this code at the bottom of the file runs, the class is in our
    # module globals.
    cls = globals().get("ChatCompletionStreamResponse")
    if cls is None:
        return

    _prev_dump_json = cls.model_dump_json

    def _scrubbed_dump_json(self, **kwargs):
        request_id = getattr(self, "id", "") or ""
        for choice in (self.choices or []):
            delta = getattr(choice, "delta", None)
            if delta is None:
                continue
            final = bool(getattr(choice, "finish_reason", None))
            if hasattr(delta, "content"):
                delta.content = _scrub_field(
                    delta.content, request_id, "content", final,
                )
            if hasattr(delta, "reasoning_content"):
                delta.reasoning_content = _scrub_field(
                    delta.reasoning_content, request_id, "reasoning_content", final,
                )
        return _prev_dump_json(self, **kwargs)

    cls.model_dump_json = _scrubbed_dump_json


_install_scrub_filter()
PYEOF

# Drop bytecode cache so the patched module is picked up on import.
find "$VLLM_ROOT/entrypoints" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

# Sanity-check: make sure the install actually runs at import time. Spawn a
# fresh interpreter so we observe a real cold import.
python3 -c "
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionStreamResponse
m = ChatCompletionStreamResponse.model_dump_json
qn = getattr(m, '__qualname__', '')
if '_scrubbed_dump_json' not in qn:
    print(f'  install-time check: FAIL (model_dump_json is still {qn})')
    import sys; sys.exit(1)
print('  install-time check: OK (model_dump_json wrapped)')
"

echo "[mimo-scrub-claude-xml-leakage] verify:"
echo "  marker present: $(grep -c MOD_SCRUB_CLAUDE_XML_LEAKAGE "$PROTOCOL")"
echo "  pattern tuple:  $(grep -c _SCRUB_PATTERNS "$PROTOCOL")"
echo "  install hook:   $(grep -c _install_scrub_filter "$PROTOCOL")"
python3 -c "
import ast, sys
try:
    ast.parse(open('$PROTOCOL').read())
    print('  python syntax: OK')
except SyntaxError as e:
    print(f'  python syntax: FAIL {e}'); sys.exit(1)
"
echo "[mimo-scrub-claude-xml-leakage] applied"
