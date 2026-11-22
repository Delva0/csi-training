# `cannot import name '<X>' from 'vllm' (unknown location)`: the import bound a stray `vllm` directory

Worked for #527 (vllm-ascend v0.7.3rc1 `f025df0`, SOURCES vllm v0.7.3 `ed6e907`;
reporter's traceback matched that tree's layout at `entrypoints/openai/api_server.py:34`).

## The shape

`vllm serve` (or `from vllm import …` in the reporter's own script) dies at import:

```text
File ".../vllm/entrypoints/openai/api_server.py", line 34, in <module>
  from vllm.engine.multiprocessing.client import MQLLMEngineClient
File ".../vllm/engine/multiprocessing/__init__.py", line 10, in <module>
  from vllm import PoolingParams
ImportError: cannot import name 'PoolingParams' from 'vllm' (unknown location)
```

The env dump beside it said `vllm 0.7.3+empty`, which is a **distractor**: `+empty`
is vllm's marker for the documented `VLLM_TARGET_DEVICE=empty` build
(`vllm/setup.py:506-507`, the route vllm-ascend's own docs prescribe), and an
installed v0.7.3 does export the name (`vllm/__init__.py:23` `from
vllm.pooling_params import PoolingParams`, `:63` in `__all__`;
`vllm/pooling_params.py:8` defines the class). Nothing in vllm-ascend can be the
owner either: it reads the **submodule** (`vllm_ascend/worker/pooling_model_runner.py:27`
`from vllm.pooling_params import PoolingParams`), never the top-level attribute.

## The tell: the message's location text

The tail after the module name says what the interpreter actually bound, and it is
the cheapest evidence in the whole report — read it before any version or source
walk. Reproduced locally (`uv run --no-project python -c 'from vllm import PoolingParams'`
from a directory holding each fixture):

| message tail | what `import vllm` bound | consequence |
|---|---|---|
| `(unknown location)` | a **bare `vllm` directory** (no `__init__.py`) — `vllm.__file__ is None`, `vllm.__path__` lists the directory — and **no regular `vllm` package anywhere on that `sys.path`** (a real `site-packages/vllm/__init__.py`, even later on the path, would have won) | the submodules the traceback names still import from that directory; every top-level name is missing |
| `(<path>/vllm/__init__.py)` | a *regular* package at that path — a checkout or a second install | the message was produced by **that** file, so diff it against the pin |

So `(unknown location)` is not a version mismatch, not a partial wheel and not
CANN: some `vllm` directory (launch cwd, `PYTHONPATH`, or an editable-install
`.pth` pointing into a checkout) precedes the real package, and the interpreter
cannot see the installed one at all. #527's reporter fixed it by deleting the
`vllm` folder in the current path (*"删除当前路径下的vllm文件夹"*,
issuecomment-2808451427, 2025-04-16) — **reporter-confirmed outcome, mechanism
matches the reproduction above**; his traceback frames
(`$HOME/hzy/vllm/vllm/entrypoints/…`) named a home-directory tree rather than
`site-packages`, which is the same finding stated by the paste itself.

This is the sibling of `plugin-load-module-missing.md`: there the metadata finds a
module that is gone, here a directory supplies a module the metadata did not
install. Same host family in the corpus — #541 (a day after #540, same box)
resolved upstream `vllm` from a `/workspace/vllm` checkout while `vllm_ascend`
came from `site-packages` (`plugin-load-module-missing.md`, candidate 2).

**A `/vllm-workspace/vllm/vllm/…` frame is not this signature by itself.** The
official images `git clone` vllm into `/vllm-workspace/vllm` and `pip install -e` it
(`Dockerfile:41,43` at vllm-ascend v0.9.1 `0740d102`), so every ordinary traceback
from those images carries that path — #527's home-directory tree was a finding
precisely because that is *not* where an image installs vllm. #2809's follow-up
(`ImportError: cannot import name 'SamplingParams' from 'vllm' (unknown location)`,
every frame under `/vllm-workspace/vllm/vllm/`, the script started from
`/workspace/code`, `example.py` still on the tutorial's `from vllm import LLM,
SamplingParams`) is therefore **undecided by the path alone**: keep the step-1 probe
as the decider. One fact the paste does give for free — the frame it names,
`vllm/v1/sample/logits_processor/interface.py:11`, does **not** exist in vllm v0.9.1
(`b6553be`; `vllm/v1/sample/logits_processor/` is absent from that tree), so the
image's vllm is newer than the tag's own pair. Ask for `python -c "import vllm;
print(vllm.__version__, vllm.__file__)"` plus `pip show vllm`, and do not quote a
pinned line number for that frame.

## Steps, in order

1. One line, in the reporter's shell **and** launch directory:
   `python -c "import vllm, sys; print(sys.executable); print(vllm.__file__); print(list(vllm.__path__)); print(vllm.__version__)"`.
   `None` + a directory, or a path outside `site-packages`, decides it.
2. Find the directory: `ls -d ./vllm 2>/dev/null; echo $PYTHONPATH; ls
   $CONDA_PREFIX/lib/python3.*/site-packages/__editable__*vllm*.pth`.
3. Move or delete it (`mv vllm vllm.bak`) or launch from a directory that does not
   contain it, then re-run the same `python -c`. Do not reinstall vllm, torch-npu
   or CANN for this shape.
4. If the probe shows a *regular* package in the wrong place (second `.pth`, two
   interpreters), take `plugin-load-module-missing.md`'s residue checks and
   `scripts/pip-pairing-check.sh` before any code theory.

## Open

#527's fix is reporter-confirmed but the report carries no post-fix log, and the
`sys.path` entry that put the directory first (cwd, `PYTHONPATH`, or a `.pth`) is
not in the paste. Do not claim which one; ask for the step-1 output. Recheck the
`vllm/__init__.py` / `pooling_params.py` line numbers at the report's own pin —
they move between releases, the `(unknown location)` discriminator does not.
