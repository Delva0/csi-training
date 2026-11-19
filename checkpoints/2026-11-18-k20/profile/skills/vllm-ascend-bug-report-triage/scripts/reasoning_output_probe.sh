#!/usr/bin/env bash
# Dump what a tag's frontend does with reasoning/thinking output: the response field name, the
# request-history field it reads, the registered reasoning/tool parser names and the `include_reasoning`
# switch. Read-only; prints to stdout.
#
# usage: bash reasoning_output_probe.sh <vllm-ascend-ref> [vllm-ref]
#   vllm-ref omitted: derived from the vllm-ascend ref (v0.18.0rc1 -> v0.18.0) and printed with its date
#                     so the pair can be sanity-checked; pass it explicitly when they differ.
# exit: 0 printed, 2 usage, 3 no such clone / ref
set -u
usage() { sed -n '2,9p' "$0" >&2; exit 2; }
[ "$#" -ge 1 ] || usage
ref=$1; vref=${2:-}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ASC=$(bash "$HERE/repo_root.sh" vllm-ascend) || exit 3
VLLM=$(bash "$HERE/repo_root.sh" vllm) || exit 3
git -C "$ASC" rev-parse -q --verify "$ref^{commit}" >/dev/null || { echo "no such ref in vllm-ascend: $ref" >&2; exit 3; }
if [ -z "$vref" ]; then
  # v0.18.0rc1 -> v0.18.0 ; used only as a starting point, the date line shows whether it is sane
  base=$(printf '%s' "$ref" | sed -E 's/^v([0-9]+\.[0-9]+)\..*/v\1.0/')
  vref=$base
fi
git -C "$VLLM" rev-parse -q --verify "$vref^{commit}" >/dev/null || { echo "no such ref in vllm: $vref" >&2; exit 3; }

echo "== vllm-ascend $ref ($(git -C "$ASC" log -1 --format='%h %ci' "$ref"))"
echo "== vllm        $vref ($(git -C "$VLLM" log -1 --format='%h %ci' "$vref"))"

echo
echo "== response field (the name the client must read) =="
git -C "$VLLM" grep -n -E '^class ChatMessage|^class DeltaMessage' "$vref" -- vllm/entrypoints/openai 2>/dev/null | head -4
git -C "$VLLM" grep -n -B1 -A1 'reasoning: str | None' "$vref" -- vllm/entrypoints/openai 2>/dev/null | head -12
echo "-- reasoning_content occurrences in the frontend (absent = field was removed):"
git -C "$VLLM" grep -l 'reasoning_content' "$vref" -- vllm 2>/dev/null | head -8

echo
echo "== request history: which field the renderer reads for an assistant message =="
git -C "$VLLM" grep -n 'reasoning = message.get' "$vref" -- vllm 2>/dev/null | head -4

echo
echo "== include_reasoning (strips reasoning when false) =="
git -C "$VLLM" grep -n -E 'include_reasoning' "$vref" -- vllm/entrypoints/openai 2>/dev/null | head -8

echo
echo "== registered reasoning parsers at this ref =="
for f in vllm/reasoning/__init__.py vllm/reasoning/abs_reasoning_parsers.py; do
  git -C "$VLLM" show "$vref:$f" 2>/dev/null | grep -nE '^    "[a-z0-9_]+"' | head -40 && break
done

echo
echo "== registered tool-call parsers at this ref (a *tool* parser name is not a reasoning parser) =="
git -C "$VLLM" show "$vref:vllm/tool_parsers/__init__.py" 2>/dev/null | grep -nE '^    "[a-z0-9_.\-]+"' | head -40

echo
echo "== qwen3 parser: does it know the no-start-token (Qwen3.5) template? =="
git -C "$VLLM" show "$vref:vllm/reasoning/qwen3_reasoning_parser.py" 2>/dev/null | grep -nE 'Qwen3\.5|<think>|prompt_is_reasoning_end' | head -8

echo
echo "== vllm-ascend side: any reasoning/thinking patch at this ref? =="
git -C "$ASC" ls-tree -r --name-only "$ref" -- vllm_ascend/patch 2>/dev/null | grep -iE 'reason|think|tool_call' | head -10 || echo "  none"
