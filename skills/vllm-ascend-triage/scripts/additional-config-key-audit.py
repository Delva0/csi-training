#!/usr/bin/env python3
"""Audit every key of an `--additional-config` JSON against the pinned AscendConfig.

Why: the pydantic error names ONE key, but a serve command copied from a model
tutorial usually carries several keys, and the reported one is often not the only
bad one -- an answer that fixes only it sends the reporter into the next failure.
One call classifies ALL keys, on either schema generation.

Usage (always with uv):

  uv run --no-project python scripts/additional-config-key-audit.py \
      --json '{"fuse_muls_add": true, "enable_dsa_cp": true}'
  uv run --no-project python scripts/additional-config-key-audit.py \
      --doc /workspace/repos/<case>/vllm-project/vllm-ascend/docs/source/tutorials/models/GLM5.3.md [--line 310]
  uv run --no-project python scripts/additional-config-key-audit.py --doc <tutorial.md> \
      --tree /workspace/repos/<case>/vllm-project/vllm-ascend

Strict generation (`@config` pydantic dataclass, `extra="forbid"`) verdicts:
  OK             top-level `AscendConfig` field -> keep the key flat
  OK-PARENT      the key that carries a nested config class (`scheduler_config`,
                 `ascend_compilation_config`, ...) -> keep it, put its fields inside
  NEST-REQUIRED  field of a nested config class -> FATAL flat, nest under the parent
  WARN-STRIPPED  in `_NON_USER_INPUT_KEYS` -> dropped before validation, warning only
  UNKNOWN        not a field anywhere -> FATAL

Permissive generation (plain `__init__` reading `additional_config.get(...)`, the
pre-`@config` era) verdicts, none of them fatal:
  OK-PARENT      as above: the key carrying a nested config class
  FLAT-READ      the module reads this key at top level
  NEST-ONLY      field of a nested class -> the flat key is silently ignored
  IGNORED        unknown key -> silently ignored

Read the generation line first: on a permissive pin no `additional_config` key can
raise `unexpected_keyword_argument`, so that traceback means the reporter's build
is not this tree (check the installed version) or the key belongs to another
pydantic model in the traceback. Exit 0 ran, 2 schema missing / bad JSON.
"""
import argparse
import ast
import glob
import json
import os
import re
import sys

TREE_PATTERNS = [
    "/workspace/repos/*/vllm-project/vllm-ascend",
    "/workspace/repos/*/*/vllm-ascend",
    "/workspace/repos/vllm-project/vllm-ascend",
]


def find_tree(explicit):
    if explicit:
        return explicit
    for pat in TREE_PATTERNS:
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[-1]
    return None


def literal(node):
    try:
        return ast.literal_eval(node)
    except Exception:
        try:
            return ast.unparse(node)
        except Exception:
            return "<expr>"


def class_fields(node):
    """Annotated dataclass fields, plus `__init__` parameters and `self.x = kwargs.get("x", d)`
    of a plain config class (the pre-`@config` generation)."""
    fields = {}
    for stmt in node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            fields[stmt.target.id] = (
                ast.unparse(stmt.annotation),
                literal(stmt.value) if stmt.value is not None else "<required>",
                stmt.lineno,
            )
        elif isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__":
            args = stmt.args
            for arg in list(args.posonlyargs) + list(args.args) + list(args.kwonlyargs):
                if arg.arg == "self":
                    continue
                default = "<required>"
                if arg in args.kwonlyargs:
                    i = args.kwonlyargs.index(arg)
                    if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
                        default = literal(args.kw_defaults[i])
                else:
                    positional = list(args.posonlyargs) + list(args.args)
                    offset = len(positional) - len(args.defaults)
                    if positional.index(arg) >= offset:
                        default = literal(args.defaults[positional.index(arg) - offset])
                fields.setdefault(arg.arg, ("param", default, arg.lineno))
    for sub in ast.walk(node):
        if isinstance(sub, ast.Assign) and isinstance(sub.value, ast.Call):
            call = sub.value
            recv = ast.unparse(call.func)
            if not (recv.endswith(".get") and len(call.args) >= 1):
                continue
            arg = call.args[0]
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            default = literal(call.args[1]) if len(call.args) > 1 else "<required>"
            for tgt in sub.targets:
                if isinstance(tgt, ast.Attribute) and ast.unparse(tgt.value) == "self":
                    fields.setdefault(arg.value, ("kwargs", default, sub.lineno))
    return fields


def annotated_classes(tree):
    classes = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            fields = class_fields(node)
            if fields:
                classes[node.name] = fields
    return classes


def strip_list(tree):
    keys = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "_NON_USER_INPUT_KEYS" for t in node.targets
        ):
            for elt in ast.walk(node.value):
                if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                    keys.setdefault(elt.value, node.lineno)
    return keys


def flat_reads(tree):
    """Keys a plain `__init__` reads: additional_config.get('k'), additional_config['k'],
    and `self.k = ...(additional_config, "k" | ...)` helper calls."""
    reads = {}

    def note(key, line):
        if isinstance(key, str) and key.isidentifier():
            reads.setdefault(key, line)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "get" and node.args and "additional_config" in ast.unparse(node.func.value):
                arg = node.args[0]
                if isinstance(arg, ast.Constant):
                    note(arg.value, node.lineno)
        elif isinstance(node, ast.Subscript) and "additional_config" in ast.unparse(node.value):
            if isinstance(node.slice, ast.Constant):
                note(node.slice.value, node.lineno)
        elif isinstance(node, ast.Assign):
            src = ast.unparse(node.value)
            if "additional_config" not in src:
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                    note(tgt.attr, node.lineno)
    return reads


def parent_map(tree, classes):
    """nested class -> AscendConfig key that carries it (annotation or self.<key> = Class(...))."""
    parent = {}
    top = classes.get("AscendConfig", {})
    for key, (ann, _d, _ln) in top.items():
        name = ann.split("[")[0].strip().rstrip("]")
        if name in classes and name != "AscendConfig":
            parent.setdefault(name, key)
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            callee = ast.unparse(node.value.func)
            if callee not in classes or callee == "AscendConfig":
                continue
            for tgt in node.targets:
                if isinstance(tgt, ast.Attribute) and isinstance(tgt.value, ast.Name) and tgt.value.id == "self":
                    parent.setdefault(callee, tgt.attr)
    return parent


def parse_schema(config_file):
    src = open(config_file, encoding="utf-8").read()
    tree = ast.parse(src)
    classes = annotated_classes(tree)
    top = classes.get("AscendConfig", {})
    parent = parent_map(tree, classes)
    # Strict means annotated pydantic-dataclass fields on AscendConfig itself, not an
    # `__init__` parameter list: the pre-@config generation is a plain class whose
    # __init__(self, vllm_config) also has parameters.
    strict = any(
        isinstance(stmt, ast.AnnAssign)
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "AscendConfig"
        for stmt in node.body
    )
    from_factory = {
        node.name
        for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(isinstance(stmt, ast.FunctionDef) and stmt.name == "from_additional_config" for stmt in node.body)
    }
    return {
        "classes": classes,
        "from_factory": from_factory,
        "top": top,
        "parent": parent,
        "stripped": strip_list(tree),
        "reads": flat_reads(tree),
        "strict": strict,
    }


def nested_owner(key, schema):
    for cname, fields in schema["classes"].items():
        if cname != "AscendConfig" and key in fields:
            return cname, schema["parent"].get(cname)
    return None, None


def classify(key, schema):
    strict = schema["strict"]
    stripped = schema["stripped"].get(key)
    owner, pkey = nested_owner(key, schema)
    if key in schema["parent"].values():
        cls = next(c for c, p in schema["parent"].items() if p == key)
        reader = (f"read by {cls}.from_additional_config" if cls in schema["from_factory"]
                  else f"consumed by {cls}(additional_config, ...)")
        note = "; in _NON_USER_INPUT_KEYS -> injected by the factory, keep it nested" if stripped else ""
        return "OK-PARENT", f"parent key carrying {cls} ({reader}){note}"
    if strict:
        if stripped:
            if key in schema["top"]:
                where = "top-level field, re-derived by the factory"
            elif owner:
                where = f"field of {owner}, resolved by a factory pre-step"
            else:
                where = "not a field"
            return "WARN-STRIPPED", f"{where}; in _NON_USER_INPUT_KEYS -> dropped before validation, warning only"
        if key in schema["top"]:
            return "OK", "top-level AscendConfig field"
        if owner:
            return "NEST-REQUIRED", (
                f"field of {owner} -> FATAL flat; nest under '{pkey}'" if pkey
                else f"field of {owner} -> FATAL flat"
            )
        return "UNKNOWN", "not a field of any config class -> FATAL"
    # permissive
    if key in schema["reads"]:
        return "FLAT-READ", f"read at top level (ascend_config.py:{schema['reads'][key]})"
    if owner:
        return "NEST-ONLY", (
            f"field of {owner}; flat key silently ignored (nested form: '{pkey}')" if pkey
            else f"field of {owner}; flat key silently ignored"
        )
    return "IGNORED", "not read by this module at this pin; grep the tree before calling it inert"


def default_of(key, schema):
    if key in schema["parent"].values():
        return ""
    if key in schema["top"]:
        return f" default={schema['top'][key][1]!r}"
    owner, _ = nested_owner(key, schema)
    if owner:
        return f" default={schema['classes'][owner][key][1]!r}"
    return ""


def corrected(obj, schema):
    out, nested = {}, {}
    for key, val in obj.items():
        verdict, _ = classify(key, schema)
        if verdict in ("OK", "OK-PARENT", "FLAT-READ"):
            out[key] = val
        elif verdict in ("NEST-REQUIRED", "NEST-ONLY", "WARN-STRIPPED"):
            owner, pkey = nested_owner(key, schema)
            if pkey and not isinstance(val, dict):
                nested.setdefault(pkey, {})[key] = val
    for pkey, sub in nested.items():
        out.setdefault(pkey, {}).update(sub)
    return out


def audit(obj, schema, label):
    print(f"## {label}")
    fatal, warn = [], []
    for key in obj:
        verdict, why = classify(key, schema)
        print(f"  {verdict:14s} {key:34s} {why}{default_of(key, schema)}")
        if verdict in ("NEST-REQUIRED", "UNKNOWN"):
            fatal.append(key)
        elif verdict == "WARN-STRIPPED" or verdict == "NEST-ONLY":
            warn.append(key)
    if schema["strict"]:
        print(f"  FATAL: {fatal or 'none'} | warning-only: {warn or 'none'}")
    else:
        print(f"  FATAL: none at this pin | ineffective-if-flat: {warn or 'none'}")
    print(f"  corrected --additional-config: '{json.dumps(corrected(obj, schema))}'")
    print()


def extract_doc(path, line=None):
    hits = []
    pat = re.compile(r"--additional-config\s+'(\{.*?\})'")
    try:
        handle = open(path, encoding="utf-8")
    except OSError as exc:
        die(f"cannot read --doc file ({exc})")
    for i, text in enumerate(handle, start=1):
        if line and i != line:
            continue
        for m in pat.finditer(text):
            try:
                hits.append((i, json.loads(m.group(1))))
            except Exception as exc:
                print(f"  ! {path}:{i} unparsable JSON ({exc})", file=sys.stderr)
    return hits


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="append", default=[], help="JSON string to audit (repeatable)")
    ap.add_argument("--doc", action="append", default=[], help="markdown file to scan for --additional-config")
    ap.add_argument("--line", type=int, help="restrict --doc audit to this line")
    ap.add_argument("--tree", help="vllm-ascend checkout root (locates vllm_ascend/ascend_config.py)")
    ap.add_argument("--config-file", help="explicit path to vllm_ascend/ascend_config.py")
    args = ap.parse_args()

    config_file = args.config_file
    if not config_file and args.tree:
        config_file = os.path.join(args.tree, "vllm_ascend", "ascend_config.py")
    if not config_file:
        tree = find_tree(None)
        if tree:
            config_file = os.path.join(tree, "vllm_ascend", "ascend_config.py")
    if not config_file or not os.path.exists(config_file):
        die("ascend_config.py not found; pass --tree <vllm-ascend root> or --config-file")

    schema = parse_schema(config_file)
    gen = ("STRICT (@config pydantic dataclass, extra='forbid')" if schema["strict"]
           else "PERMISSIVE (plain __init__ + additional_config.get)")
    print(f"# schema: {config_file}\n# generation: {gen}; nested classes: {len(schema['parent'])}; "
          f"stripped keys: {len(schema['stripped'])}\n")

    for raw in args.json:
        try:
            obj = json.loads(raw)
        except Exception as exc:
            die(f"bad --json ({exc}): {raw[:120]}")
        audit(obj, schema, "--additional-config (given)")

    for doc in args.doc:
        hits = extract_doc(doc, args.line)
        if not hits:
            print(f"# no --additional-config JSON found in {doc} (line={args.line})")
        for ln, obj in hits:
            audit(obj, schema, f"{doc}:{ln}")

    if not schema["strict"]:
        print(
            "# No extra=\"forbid\" schema in this module: an unknown additional_config key is\n"
            "# silently ignored at this pin, so a pydantic `unexpected_keyword_argument` cannot\n"
            "# come from AscendConfig here. Either the reporter's build is newer than this tree\n"
            "# (check the installed version) or the named key belongs to another pydantic model\n"
            "# in the traceback."
        )


if __name__ == "__main__":
    main()
