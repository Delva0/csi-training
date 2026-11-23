#!/usr/bin/env bash
# batch-split-audit.sh [root ...]
#
# Static audit of decode/prefill batch-split sites for index-UNIT confusion.
# Units in this code base: num_decodes / num_prefills count REQUESTS;
# num_decode_tokens / num_prefill_tokens count TOKENS, and with speculative
# decoding num_decode_tokens = num_decodes * (1 + num_speculative_tokens).
# A request-indexed tensor (block_tables, seq_lens, state_indices) sliced by a
# token count, or a cu_seqlens-style cumulative array (query_start_loc,
# actual_seq_lengths, actual_seq_qlen) passed without rebasing to 0, reads
# outside its own storage once spec decode makes num_decode_tokens >> num_decodes.
#
# The same confusion also appears on the HOST side, without any device frame: the
# per-request schedule array num_scheduled_tokens (one entry per request, holding
# that request's scheduled TOKEN count) sliced by num_decode_tokens. numpy then
# reports it directly, e.g. at v0.21.0rc1 worker/pcp_utils.py:660 ->
# "ValueError: could not broadcast input array from shape (29,) into shape (31,)"
# (#11231, fixed by v0.23.0). That is why num_scheduled_tokens /
# num_padded_scheduled_tokens / cu_num_scheduled_tokens are in the pattern below.
#
# Usage: bash batch-split-audit.sh <vllm-ascend-root> [more roots]
set -uo pipefail
roots=("$@"); [ ${#roots[@]} -eq 0 ] && roots=(.)
mapfile -t files < <(grep -rl --include='*.py' -E 'num_decode_tokens' "${roots[@]}" 2>/dev/null | sort)
if [ ${#files[@]} -eq 0 ]; then
  echo "no python file mentions num_decode_tokens under: ${roots[*]}"; exit 0
fi
# the offset may be a bare name or self./attn_metadata.-qualified: pcp_utils writes
# num_scheduled_tokens[self.num_decode_tokens :], fa3_v1 writes block_tables[num_decode_tokens:, :]
req_tok='(block_tables|block_table_tensor|seq_lens|seq_lens_list|actual_seq_lengths_kv|state_indices|paged_state_indices|num_scheduled_tokens|num_padded_scheduled_tokens|cu_num_scheduled_tokens|query_lens)[[:space:]]*\[[[:space:]]*[A-Za-z_.]*num_decode_tokens'
cu_abs='(query_start_loc|actual_seq_lengths|actual_seq_lengths_q|actual_seq_qlen|chunk_seq_lens)[[:space:]]*\[[[:space:]]*num_decodes[[:space:]]*:'
hits=0; rebases=0
for f in "${files[@]}"; do
  shadow=""
  grep -qE 'num_decode_tokens[[:space:]]*=[[:space:]]*num_decodes([^_a-zA-Z]|$)' "$f" && \
    shadow="  [note: this file also rebinds num_decode_tokens := num_decodes (request count), so a hit here may be benign]"
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    printf 'UNIT-MISMATCH  %s:%s%s\n    %s\n' "$f" "${line%%:*}" "$shadow" \
      "$(echo "${line#*:}" | sed 's/^[[:space:]]*//')"
    hits=$((hits+1))
  done < <(grep -nE "$req_tok" "$f" 2>/dev/null)
  while IFS= read -r line; do
    [ -z "$line" ] && continue
    case "${line#*:}" in *'- num_decode_tokens'*|*'-num_decode_tokens'*) continue;; esac
    printf 'REBASE-CHECK   %s:%s  (cu_seqlens-style slice; no "- num_decode_tokens" on the line)\n    %s\n' \
      "$f" "${line%%:*}" "$(echo "${line#*:}" | sed 's/^[[:space:]]*//')"
    rebases=$((rebases+1))
  done < <(grep -nE "$cu_abs" "$f" 2>/dev/null)
done
echo "---"
echo "files scanned: ${#files[@]}   unit mismatches: $hits   unrebased cu_seqlens slices: $rebases"
if [ $((hits+rebases)) -eq 0 ]; then
  echo "verdict: no split-site unit confusion found at this pin"
else
  echo "verdict: inspect every site above before theorising about the kernel."
  echo "         Request-indexed tensors take num_decodes; a cu_seqlens array must start at 0,"
  echo "         i.e. query_start_loc[num_decodes:] - num_decode_tokens."
fi
