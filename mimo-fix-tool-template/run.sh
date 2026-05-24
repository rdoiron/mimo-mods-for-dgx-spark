#!/bin/bash
# Verify the patched MiMo chat template is present.
#
# This mod ships chat_template.jinja alongside this run.sh. launch-cluster.sh
# copies the mod directory to /workspace/mods/mimo-fix-tool-template/ inside
# the container at startup. The launch script then passes
#   --chat-template /workspace/mods/mimo-fix-tool-template/chat_template.jinja
# to vllm serve so this template overrides the one in tokenizer_config.json.
#
# The patch (vs the original HF tokenizer_config template) adds a from_json
# branch when tool_call.arguments is a string — the OpenCode/Droid multi-turn
# tool-call replay format that causes "Expected 'function.name' to be a string"
# downstream when the original template iterates the string with |items.
#
# Mirrors the fix Eugene applied to Qwen3.5 (mods/fix-qwen3.5-chat-template).

set -e
MOD_DIR="$(dirname "$0")"
TEMPLATE="$MOD_DIR/chat_template.jinja"

if [ ! -f "$TEMPLATE" ]; then
    echo "[mimo-fix-tool-template] ERROR: $TEMPLATE not found"
    exit 1
fi

if ! grep -q '_tool_call_arguments = tool_call.arguments | from_json' "$TEMPLATE"; then
    echo "[mimo-fix-tool-template] ERROR: template missing the from_json patch"
    exit 1
fi

echo "[mimo-fix-tool-template] Patched template present: $TEMPLATE"
echo "[mimo-fix-tool-template] $(wc -l < "$TEMPLATE") lines, $(wc -c < "$TEMPLATE") bytes"
echo "[mimo-fix-tool-template] vLLM should be launched with --chat-template $TEMPLATE"
