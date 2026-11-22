"""Minimal torch_npu profiler start/stop probe: does the patch release segfault?

Purpose
-------
`/start_profile` + `/stop_profile` on a vllm-ascend serve dies with
`!!!!!!! Segfault encountered !!!!!!!` while the reporter's other `torch_npu`
patch releases are fine (recorded #15983: `2.10.0.post4` fails, `.post2`/`.post6`
work). This probe reproduces — or refutes — that lane without vLLM, so the
answer does not depend on the reporter's model, TP size or serve command.

Run it once per patch release, in the same image:

    python -c "import torch_npu; print(torch_npu.__version__)"
    uv run --no-project python templates/profiler-patch-bisect.py --tag post4

Exit code 0 + a trace directory = this lane is not reproduced here; go back to
the RPC/thread lane (`references/observability-and-logs/online-profiler-stop-segfault.md`
Rule 6). A signal death between `stop: begin` and `stop: done` = the patch-release
lane, and this script is the attachment for the upstream report.

The `_ExperimentalConfig` below is copied from
`vllm_ascend/profiler/torch_npu_profiler.py:49-56` at v0.26.0rc1 — edit it if the
pin's wrapper differs, because the option set is part of the condition.
"""

import argparse
import os
import sys
import traceback


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="run", help="label for this patch release")
    ap.add_argument("--dir", default="./prof_bisect", help="trace output directory")
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument(
        "--data-simplification",
        choices=("true", "false"),
        default="true",
        help="wrapper default is true; flip it to test that option as a variable",
    )
    ap.add_argument(
        "--first",
        choices=("start", "stop"),
        default="start",
        help="start then stop (normal) or stop without start (see worker.py:1058-1060)",
    )
    args = ap.parse_args()

    import torch
    import torch_npu  # noqa: F401  (registers the PrivateUse1 backend)

    dev = "npu:0"
    print(f"torch={torch.__version__} torch_npu={torch_npu.__version__}", flush=True)
    print(f"cann={os.environ.get('ASCEND_HOME_PATH', '<unset>')}", flush=True)

    out = os.path.abspath(os.path.join(args.dir, args.tag))
    os.makedirs(out, exist_ok=True)

    experimental_config = torch_npu.profiler._ExperimentalConfig(
        export_type=torch_npu.profiler.ExportType.Text,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        msprof_tx=False,
        aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
        l2_cache=False,
        op_attr=False,
        data_simplification=(args.data_simplification == "true"),
        record_op_args=False,
        gc_detect_threshold=None,
    )
    prof = torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        with_stack=False,
        profile_memory=False,
        with_modules=False,
        experimental_config=experimental_config,
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(out),
    )

    a = torch.randn(args.size, args.size, device=dev, dtype=torch.float16)
    b = torch.randn(args.size, args.size, device=dev, dtype=torch.float16)

    def work() -> None:
        for _ in range(args.steps):
            a @ b
        torch.npu.synchronize()

    try:
        if args.first == "start":
            print("start: begin", flush=True)
            prof.start()
            print("start: done", flush=True)
            work()
        print("stop: begin", flush=True)
        prof.stop()
        print("stop: done", flush=True)
        # The pta export/parse may run asynchronously after stop(); a crash here is
        # still this lane.
        torch.npu.synchronize()
        print("export: done", flush=True)
    except Exception:
        traceback.print_exc()
        return 2

    files = sorted(os.listdir(out))
    print(f"traces under {out}: {files or '<none — parsing produced nothing>'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
