#!/bin/bash
# Mod: mimo-recognize-claude-thinking-close (V8)
#
# V7 — Skip-list heuristic for closer detection works. But is_reasoning_end
#      naively returned True whenever tool_call_token_id was in input_ids,
#      catching the chat-template's RENDERED tool-call EXAMPLE block when
#      tools are provided in the request. That set state.reasoning_ended=True
#      at request start, routing the entire response to content channel.
# V8 — Restore the original parser's pair-check on tool_call_token: only
#      treat an unpaired <tool_call> (no matching </tool_call> after it) as
#      implicit reasoning end. Paired template examples are skipped.
#      All V7 skip-list heuristic and scan-back logic preserved.

set -e
VLLM_ROOT="/usr/local/lib/python3.12/dist-packages/vllm"
PARSER="$VLLM_ROOT/reasoning/qwen3_reasoning_parser.py"
BACKUP="$PARSER.bak-claude-thinking-close"

if grep -q "MOD_CLAUDE_THINKING_CLOSE_V8" "$PARSER"; then
    echo "[mimo-recognize-claude-thinking-close] V8 already applied, skipping"
    exit 0
fi

if grep -q "MOD_CLAUDE_THINKING_CLOSE" "$PARSER"; then
    if [ -f "$BACKUP" ]; then
        cp "$BACKUP" "$PARSER"
        echo "[mimo-recognize-claude-thinking-close] reverted prior version, applying V8"
    else
        sed -i "/# === MOD_CLAUDE_THINKING_CLOSE/,\$d" "$PARSER"
        echo "[mimo-recognize-claude-thinking-close] stripped prior version inline, applying V8"
    fi
fi

[ -f "$BACKUP" ] || cp "$PARSER" "$BACKUP"

cat >> "$PARSER" <<'PYEOF'


# === MOD_CLAUDE_THINKING_CLOSE_V8 ===
# Same closer/heuristic machinery as V7, but is_reasoning_end now uses the
# original parser's pair-check on tool_call_token: only treat unpaired
# <tool_call> (no matching </tool_call> after) as implicit reasoning end.
# This skips chat-template example blocks when tools are in the request.

_MOD_CLOSERS = [
    "</think>",                                                       # canonical
    "</\U0001D44E\U0001D45B\U0001D461\U0001D45A\U0001D459:thinking>",  # </𝑎𝑛𝑡𝑚𝑙:thinking>
    "</thinking>",                                              # ASCII namespaced
    "</thinking>",                                                    # bare ASCII
]

_MOD_SKIP_PRECEDING = {"`", "\\"}


def _mod_is_real_closer_pos(text, idx):
    if idx == 0:
        return True
    return text[idx - 1] not in _MOD_SKIP_PRECEDING


def _mod_find_first_real_closer(text):
    best_idx = None
    best_marker = None
    for m in _MOD_CLOSERS:
        start = 0
        while True:
            idx = text.find(m, start)
            if idx == -1:
                break
            if _mod_is_real_closer_pos(text, idx):
                if best_idx is None or idx < best_idx:
                    best_idx = idx
                    best_marker = m
                break
            start = idx + 1
    return best_idx, best_marker


def _mod_partial_prefix_at_end(text):
    max_len = 0
    for m in _MOD_CLOSERS:
        max_check = min(len(m) - 1, len(text))
        for plen in range(max_check, 0, -1):
            if text.endswith(m[:plen]):
                prefix_start = len(text) - plen
                if _mod_is_real_closer_pos(text, prefix_start):
                    if plen > max_len:
                        max_len = plen
                break
    return max_len


def _mod_extract_reasoning_streaming(
    self, previous_text, current_text, delta_text,
    previous_token_ids, current_token_ids, delta_token_ids,
):
    if len(delta_token_ids) == 1 and delta_token_ids[0] in (
        self.start_token_id, self.end_token_id,
    ):
        return None

    prev_idx, _ = _mod_find_first_real_closer(previous_text)
    if prev_idx is not None:
        return DeltaMessage(content=delta_text)

    cur_idx, cur_marker = _mod_find_first_real_closer(current_text)
    if cur_idx is not None:
        marker_end = cur_idx + len(cur_marker)
        prev_partial = _mod_partial_prefix_at_end(previous_text)
        prev_safe_end = len(previous_text) - prev_partial
        reasoning = current_text[prev_safe_end:cur_idx]
        content = current_text[marker_end:]
        return DeltaMessage(
            reasoning_content=reasoning if reasoning else None,
            content=content if content else None,
        )

    prev_partial = _mod_partial_prefix_at_end(previous_text)
    cur_partial = _mod_partial_prefix_at_end(current_text)
    prev_safe_end = len(previous_text) - prev_partial
    cur_safe_end = len(current_text) - cur_partial

    if cur_safe_end > prev_safe_end:
        return DeltaMessage(reasoning_content=current_text[prev_safe_end:cur_safe_end])
    return None


def _mod_extract_reasoning(self, model_output, request):
    parts = model_output.partition(self.start_token)
    body = parts[2] if parts[1] else parts[0]

    idx, marker = _mod_find_first_real_closer(body)
    if idx is not None:
        return body[:idx] or None, body[idx + len(marker):] or None

    if not getattr(self, "thinking_enabled", True):
        return None, body

    tool_call_index = body.find(getattr(self, "_tool_call_tag", "<tool_call>"))
    if tool_call_index != -1:
        return body[:tool_call_index] or None, body[tool_call_index:] or None

    return body, None


def _mod_is_reasoning_end(self, input_ids):
    """Reasoning has ended if:
    - tokens AFTER the last <think> opener (decoded) contain a real closer, OR
    - an UNPAIRED <tool_call> token appears (paired template examples skipped).
    Returns False if input has no <think> opener."""
    start_id = self.start_token_id
    end_id = self.end_token_id
    tool_call_id = getattr(self, "_tool_call_token_id", None)
    tool_call_end_id = getattr(self, "_tool_call_end_token_id", None)

    input_list = list(input_ids)

    # Find last <think> opener position
    last_start = None
    for i in range(len(input_list) - 1, -1, -1):
        if input_list[i] == start_id:
            last_start = i
            break

    if last_start is None:
        # No <think> in input - heuristic doesn't apply, defer
        return False

    # Scan tokens AFTER the last <think> for tool_call implicit end or end token
    suffix_ids = input_list[last_start + 1:]

    # Check for unpaired <tool_call> (pair-check matches original parser logic)
    if tool_call_id is not None:
        for i, tid in enumerate(suffix_ids):
            if tid == tool_call_id:
                # Is there a matching </tool_call> after this position?
                if tool_call_end_id is not None and any(
                    suffix_ids[j] == tool_call_end_id
                    for j in range(i + 1, len(suffix_ids))
                ):
                    # Paired - skip (chat template example or completed call)
                    continue
                return True  # unpaired - real implicit end

    # Decode suffix and apply text-based heuristic for </think>-family closers
    try:
        text_after = self.model_tokenizer.decode(suffix_ids, skip_special_tokens=False)
    except Exception:
        # Fallback: scan for end_id directly in suffix
        for tid in suffix_ids:
            if tid == end_id:
                return True
        return False

    idx, _ = _mod_find_first_real_closer(text_after)
    return idx is not None


def _mod_is_reasoning_end_streaming(self, input_ids, delta_ids):
    combined = list(input_ids) + list(delta_ids)
    return _mod_is_reasoning_end(self, combined)


Qwen3ReasoningParser.extract_reasoning_streaming = _mod_extract_reasoning_streaming
Qwen3ReasoningParser.extract_reasoning = _mod_extract_reasoning
Qwen3ReasoningParser.is_reasoning_end = _mod_is_reasoning_end
Qwen3ReasoningParser.is_reasoning_end_streaming = _mod_is_reasoning_end_streaming
PYEOF

echo "[mimo-recognize-claude-thinking-close] V8 verify:"
echo "  V8 marker:                    $(grep -c MOD_CLAUDE_THINKING_CLOSE_V8 "$PARSER")"
echo "  skip-list set:                $(grep -c _MOD_SKIP_PRECEDING "$PARSER")"
echo "  closer list:                  $(grep -c _MOD_CLOSERS "$PARSER")"
echo "  pair-check (tool_call_end):   $(grep -c _tool_call_end_token_id "$PARSER")"
echo "  monkey-patch assignments:     $(grep -c 'Qwen3ReasoningParser\.' "$PARSER")"
python3 -c "
import ast, sys
src = open('$PARSER').read()
try:
    ast.parse(src)
    print('  python syntax:                OK')
except SyntaxError as e:
    print(f'  python syntax:                FAIL {e}'); sys.exit(1)
"
echo "[mimo-recognize-claude-thinking-close] V8 applied"
