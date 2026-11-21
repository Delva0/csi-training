#!/usr/bin/env bash
# lora-graph-descriptor-audit.sh <vllm-ascend-root> [vllm-root]
#
# Static audit for two LoRA findings that share one question: *which* LoRA code
# the pinned tree actually runs, and whether the graph-capture case list it
# advertises is the set it really captures.
#
#   1. punica backend selection in vllm_ascend/lora/punica_npu.py --
#      `vllm.lora.ops.torch_ops` (PyTorch per-slice gather+cast+einsum, launch
#      bound) vs `vllm_ascend.lora.lora_ops` (AscendC kernels).  The predicate is
#      version-specific: unconditional torch at v0.9.2rc1; at v0.13.0rc2 torch
#      only for 310P or max_lora_rank >= 128.
#   2. every `has_lora` producer, split into "live batch" and "config constant".
#      A BatchDescriptor is the cudagraph dispatch key (num_tokens, num_reqs,
#      uniform, has_lora); when startup capture writes the same descriptor for
#      every case while the dispatcher advertises both, the missing key is
#      captured lazily at request time and upstream's guard
#      (`vllm/compilation/monitor.py` validate_cudagraph_capturing_enabled)
#      raises 'CUDA graph capturing detected at an inappropriate time'.
#
# Usage: bash lora-graph-descriptor-audit.sh /path/to/vllm-ascend [/path/to/vllm]
# Read-only; exits 0 even when a tree is a partial export.
set -uo pipefail
ac_root=${1:-.}
vl_root=${2:-}
[ -n "$vl_root" ] && [ ! -d "$vl_root" ] && vl_root=""

g() { grep -rn --include='*.py' -E "$2" "$1" 2>/dev/null; }

echo "== lora-graph-descriptor-audit =="
echo "vllm-ascend: $ac_root${vl_root:+   vllm: $vl_root}"

echo
echo "-- 1. punica backend (vllm_ascend/lora/punica_npu.py) --"
pn=$(ls "$ac_root"/vllm_ascend/lora/punica_npu.py "$ac_root"/vllm_ascend/lora/punica_wrapper/punica_npu.py 2>/dev/null | head -1)
if [ -z "$pn" ]; then
  echo "  no punica_npu.py found (partial export?) -- grep it: rg -n 'torch_ops' $ac_root/vllm_ascend"
else
  echo "  file: ${pn#"$ac_root"/}"
  if grep -q 'vllm_ascend.lora.lora_ops\|vllm_ascend\.lora import' "$pn"; then
    echo "  -> AscendC lora_ops path EXISTS; selection predicate (device type / rank):"
    grep -n -E '_310p|_310P|max_lora_rank|get_ascend_device_type|if .*lora_config' "$pn" | sed 's/^/     /'
  else
    echo "  -> torch_ops ONLY (pure-PyTorch punica: no AscendC LoRA kernel in this tree)"
  fi
  grep -n 'from vllm.lora.ops.torch_ops' "$pn" | sed 's/^/     torch fallback at /'
fi
if [ -f "$ac_root/csrc/kernels/bgmv_shrink.cpp" ]; then
  echo "  csrc/kernels/bgmv_shrink.cpp present (kernel sources shipped)"
else
  echo "  no csrc/kernels/bgmv_shrink.cpp under this root"
fi

echo
echo "-- 2. has_lora producers (descriptor key component) --"
for root in "$ac_root" "$vl_root"; do
  [ -z "$root" ] && continue
  hits=$(g "$root" 'has_lora[[:space:]]*=' | grep -v '/tests\?/')
  [ -z "$hits" ] && { echo "  (${root##*/}) no has_lora assignments"; continue; }
  echo "  (${root##*/})"
  while IFS= read -r line; do
    case "$line" in
      *lora_id_to_lora_request*|*lora_requests*|*input_batch*) tag="LIVE-BATCH" ;;
      *specialize_lora*|*max_lora_rank*)                    tag="CONFIG-CONST" ;;
      *)                                                       tag="?" ;;
    esac
    printf '    %-12s %s\n' "$tag" "$line"
  done <<< "$hits"
done
echo "  A CONFIG-CONST producer inside the dummy-run/capture path is the defect:"
echo "  capture then writes one descriptor per size for every case in the loop."

echo
echo "-- 3. is the per-case flag consumed on the downstream dummy run? --"
if [ -n "$pn" ]; then
  mr=$(dirname "$pn")/../worker/model_runner_v1.py
  [ -f "$mr" ] || mr=$(ls "$ac_root"/vllm_ascend/worker/model_runner_v1.py 2>/dev/null)
  if [ -f "$mr" ]; then
    sig=$(grep -n 'activate_lora\|force_has_lora' "$mr")
    if [ -z "$sig" ]; then
      echo "  no activate_lora/force_has_lora in model_runner_v1.py (older fork: the case list itself may not exist)"
    else
      while IFS= read -r l; do printf '    %s\n' "$l"; done <<< "$sig"
      n=$(printf '%s\n' "$sig" | wc -l)
      echoed=$(printf '%s\n' "$sig" | grep -c 'has_lora[[:space:]]*=[^=].*activate_lora\|has_lora[[:space:]]*=[^=].*force_has_lora')
      if [ "$echoed" -eq 0 ]; then
        echo "  -> flag appears only in the signature/plumbing, never in the has_lora computation:"
        echo "     capture cases collapse onto one descriptor (the #6279 mechanism)."
      else
        echo "  -> the per-case flag reaches the descriptor; re-read the capture loop before blaming it."
      fi
    fi
    grep -n 'GPUModelRunner.capture_model' "$mr" | sed 's/^/    delegates capture: /'
  fi
fi

echo
echo "-- 4. upstream side: guard, flag toggles, specialize default --"
if [ -n "$vl_root" ]; then
  if grep -qs 'def validate_cudagraph_capturing_enabled' "$vl_root/vllm/compilation/monitor.py" 2>/dev/null; then
    grep -n 'def validate_cudagraph_capturing_enabled' -A6 "$vl_root/vllm/compilation/monitor.py" | sed 's/^/  /'
  else
    echo "  no vllm/compilation/monitor.py in this upstream tree: this guard does not exist"
    echo "  here, so a pasted 'CUDA graph capturing detected...' line comes from a newer"
    echo "  vllm (check the image's vllm version before attributing it to the fork)."
  fi
  g "$vl_root/vllm" 'set_cudagraph_capturing_enabled\(' | grep -v '/tests\?/' | sed 's/^/  /'
  g "$vl_root/vllm/config" 'cudagraph_specialize_lora' | sed 's/^/  /'
  g "$vl_root/vllm/v1/cudagraph_dispatcher.py" 'lora_cases|relax_for_mixed_batch_cudagraphs\(\)' | sed 's/^/  /'
else
  echo "  (no vllm root given)"
fi

echo
echo "-- reading --"
echo "  CONFIG-CONST has_lora + both lora_cases advertised  => base-model (no adapter)"
echo "      request lazily captures at request time => guard raises => worker dies."
echo "  Workarounds, cheapest first: --enforce-eager / cudagraph_mode NONE;"
echo "      traffic carries no adapter => do not enable LoRA at all (lora_config None"
echo "      makes both sides compute has_lora=False, so graphs replay); else send only"
echo "      adapter requests (the asymmetry also confirms the mechanism); else the"
echo "      one-line fix in _dummy_run (use the per-case activate_lora)."
echo "      NOT cudagraph_specialize_lora=false: keys are built from that same flag, so"
echo "      capture's config-const has_lora=False then misses the has_lora=True key set"
echo "      and the fork raises 'Aclgraph runtime mode mismatch at dummy_run' at start."
echo "  torch_ops backend + an eager-only SoC (310P) => per-step launch overhead, not"
echo "      arithmetic; see references/lora-adapters/decode-overhead-vs-base-model.md."
exit 0
