#!/usr/bin/env bash
# Print the ref facts for a report and write them, with a one-screen report skeleton, to
# /workspace/scratch/answer.md. Use it in batch 1 to buy the ref facts without reading them.
#
# The version argument is resolved the way a reporter writes it ("0.18.0", "v0.23.0rc1", "0.26.rc1",
# "0.21.0rc1-310p", a dev string ending in "+g<sha>"): resolve_ref.sh normalises the spelling and tries
# the tag/branch spellings before the snapshot runs, so the file never starts with
# "fatal: ambiguous argument". If nothing resolves the skeleton is still written (the run always has a
# scratch file) but the script exits 2 with the candidate tags, so an unusable ref block is visible.
#
#   bash open_answer.sh "<repo>#<issue> — <symptom in one line>" [version-or-ref] [repo-path-or-name]
#
# The third argument is a clone path or a clone name ("vllm-ascend", "vllm", "Mooncake"); a name is
# mapped through repo_root.sh, because a bare name used to fail every git call and read as "the tag
# does not exist". Exit 3 = the repo argument names no clone; exit 2 = the version does not resolve.
#
# The file is SCRATCH NOTES, not the deliverable: the reply text is what is scored, so never rewrite
# this file and never compose the report in a heredoc (the characters are charged twice). The whole
# file is printed, because its ref facts are the payload of batch 1.
# Read-only against the clones; writes only /workspace/scratch/answer.md.
set -uo pipefail

OUT=/workspace/scratch/answer.md
TITLE="${1:?usage: open_answer.sh \"<repo>#<issue> — <symptom>\" [version-or-ref] [repo-path-or-name]}"
RAWREF="${2:-}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p /workspace/scratch

# --- the repo argument is a path OR a clone name; keep its failure separate from a ref failure -----
REPO=""
REPO_ERROR=""
if ! REPO=$(bash "$HERE/repo_root.sh" "${3:-}" 2>&1); then
  REPO_ERROR="$REPO"
  REPO=""
fi

# --- resolve the reported version to a ref that exists in the clone --------------------------------
REF=""
CANDIDATES=""
BRANCHES=""
SHA_HINT=""
if [ -n "$RAWREF" ] && [ -n "$REPO" ]; then
  if REF=$(bash "$HERE/resolve_ref.sh" "$RAWREF" "$REPO" 2>/dev/null); then
    :
  else
    REF=""
    RESOLVE_ERR=$(bash "$HERE/resolve_ref.sh" "$RAWREF" "$REPO" 2>&1)
    CANDIDATES=$(printf '%s\n' "$RESOLVE_ERR" \
      | sed -n 's/^\(v[0-9][^ ]*\)$/\1/p' | sort -u | tr '\n' ' ')
    # Release branches matter: an rc/dev build usually lives on `releases/vX.Y`
    # before the tag exists, and the tree there differs from the nearest tag.
    BRANCHES=$(printf '%s\n' "$RESOLVE_ERR" \
      | sed -n 's/^[[:space:]]*\(origin\/releases\/[^[:space:]]*\)$/\1/p' | sort -u | tr '\n' ' ')
    # A `X.Y.ZrcN.devK+g<sha>` string resolves through its sha; say so instead of
    # leaving the caller to reconstruct it.
    case "$RAWREF" in
      *+g*) SHA_HINT=$(printf '%s' "$RAWREF" | grep -oE 'g[0-9a-f]{7,40}' | head -1 | cut -c2- || true) ;;
    esac
  fi
fi

{
  printf '# %s\n' "$TITLE"
  if [ -n "$REF" ]; then
    printf 'ref: %s  [SCRATCH NOTES - the reply text is the deliverable, do not rewrite this file]\n' "$REF"
    if [ "$REF" != "$RAWREF" ]; then
      printf 'requested "%s" -> resolved "%s"\n' "$RAWREF" "$REF"
    fi
    printf '\n## ref facts\n\n```text\n'
    bash "$HERE/ref_snapshot.sh" "$REF" "$REPO" 2>&1 | sed -n '1,60p'
    printf '```\n\n'
  elif [ -z "$REPO" ]; then
    printf 'ref: NOT RESOLVED - the repo argument names no clone  [SCRATCH NOTES]\n'
    printf 'repo argument: %s\n' "${3:-<empty>}"
    printf '%s\n' "$REPO_ERROR" | sed 's/^/  /'
    printf 're-run with the clone name or path, e.g. ... %s vllm-ascend\n\n' "$RAWREF"
  elif [ -n "$RAWREF" ]; then
    printf 'ref: UNRESOLVED from "%s"  [SCRATCH NOTES - the reply text is the deliverable]\n' "$RAWREF"
    printf 'candidate tags: %s\n' "${CANDIDATES:-(none matching; list them with git tag --list)}"
    printf 'candidate release branches: %s\n' "${BRANCHES:-(none matching; git branch -r --list 'origin/releases/*')}"
    printf 'next: read the tree at the reported sha, not at the nearest tag - a `X.Y.ZrcN.devK+g<sha>`%s\n' \
      "${SHA_HINT:+ (sha $SHA_HINT)}"
    printf '      version is a branch build. re-run with the whole collect_env string, e.g. ... 0.13.0rc4.dev3+g%s vllm-ascend\n' \
      "${SHA_HINT:-<sha>}"
    printf '      or: git -C <repo> ls-tree -r --name-only <sha> -- <path>\n\n'
  else
    printf 'ref: <tag>  [no version given: bash scripts/version_ref.sh <version>]\n\n'
  fi
  cat <<'MD'
## Report (this is also the reply, in order, about 12 lines)
symptom -> mechanism: <one line, version-scoped>          [verified | reported | hypothesis]
workaround: <exact flag/env/version delta>; cost: <what it trades away>; <confirmed by reporter | proposed>
root cause: <repo>, <path:line at the reported ref>, gating condition: <...>, data flow: <value/shape/ordering>
why this symptom: <the log line / stall point / accuracy delta it produces>
evidence: <label + ref/commit/comment per claim>
falsifying check: <the measurement that would kill this>
status and limits: <fixed-in | open | known issue>; <what you did not verify>
MD
} > "$OUT"

echo "wrote $OUT ($(wc -c <"$OUT") bytes; scratch only - the reply is the deliverable)"
# Print the whole scratch file, not just its header: the ref facts are the payload, and a run that
# sees only the header re-derives them by hand with its remaining calls.
sed -n '1,80p' "$OUT"
if [ -z "$REPO" ]; then
  echo "!! REPO NOT FOUND: the repo argument names no clone under /workspace/repos."
  echo "!! Fix the repo argument (a clone name such as vllm-ascend or vllm, or a full path)."
  echo "!! This is NOT evidence that the reported tag, path or file is absent."
  exit 3
fi
if [ -n "$RAWREF" ] && [ -z "$REF" ]; then
  echo "!! REF UNRESOLVED: \"$RAWREF\" is not a tag, branch or commit in $REPO."
  echo "!! candidate tags: ${CANDIDATES:-none}"
  echo "!! the ref block above is empty on purpose: resolve it in this same message, before answering."
  exit 2
fi
# The gate: the reply is the deliverable, so the search gets a few small batches and the reply goes
# out even when a mechanism stays unsettled.
echo "HARD GATE - the reply is the deliverable."
echo "      * own output = reasoning + tool arguments + content, drawn from one run-level allowance; a tool"
echo "        result is free. A run that never writes content ships nothing."
echo "      * Three tool-calling messages, then THE REPLY: batch 1 is the thread plus this script, batch 2"
echo "        the one code window or class page the symptom needs, batch 3 only if batches 1-2 named"
echo "        neither a lever nor a path:line. The message after the third is content, whatever it holds."
echo "      * Append one cheap command to this same call: grep -n -m3 '#<N>' <skill_root>/references/"
echo "        symptoms/issue-ledger/index.md - a hit means the report is already analysed and batch 2 is"
echo "        that row's page, not a fresh code read."
echo "      * The reply message carries content and NOTHING else: reasoning_content is emitted before"
echo "        content, so a closer that drafts the answer ships the draft and loses the reply. A sentence"
echo "        that already carries the workaround and the status IS the reply."
echo "      * The reply is ~1.2-2 K characters (a CJK reply costs more per character): lever, path:line,"
echo "        mechanism, limits, falsifier - cut the alternatives, never the reply."
echo "      * An unsettled mechanism is a [hypothesis] line with its falsifier, not another batch; print"
echo "        line numbers with 'git show <ref>:<path> | nl -ba | sed -n A,Bp' instead of counting them."
echo "      * 2-3 independent calls per batch, one question each, command strings short."
echo "      If the thread named the fix, the version or the workaround, the reply follows it directly:"
echo "      derive nothing further - a ruled-out matrix is already an answer."
echo "      A spoken version is not a tag ('v0.20' may be v0.20.2rc1); enumerate with 'git tag --list'."
echo "      'git tag --contains <sha>' needs --list 'v0.*' (the vllm clone's proto-v0.* tags come first)."
