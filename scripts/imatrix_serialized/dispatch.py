"""Pick the right band-serialized generator for a checkpoint, or fail clearly.

Phase 6 of apex_pipeline.sh used to call serialized_gen.py unconditionally. That file
is GraniteMoeHybrid-only AND transformers-5.5-only, so the wired path ImportError'd on
a newer transformers and could never run the qwen3_5_moe case the backend exists for.

Selection is by CAPABILITY PROBE, not by version number, because the version ranges
are genuinely misleading: transformers 5.5.1 already ships Qwen3_5MoeExperts and
Llama4Router, but not GraniteMoeHybridExperts (5.14+) and not Glm4MoeExperts. Probing
for the symbol each generator imports is exact and does not rot when 5.x moves on.

Usage:
    dispatch.py --model DIR [--print-script] [... generator args ...]

With --print-script it prints the chosen script path and exits; otherwise it execs the
generator, passing every argument through unchanged.
"""
import argparse
import importlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


class Variant:
    def __init__(self, script, module, symbols, needs, status="validated"):
        self.script = script
        self.module = module
        self.symbols = symbols
        self.needs = needs          # human-readable requirement, for error messages
        self.status = status

    def missing(self):
        """Return a list of unavailable requirements ([] means this variant can run)."""
        try:
            m = importlib.import_module(self.module)
        except ImportError as e:
            return [f"module {self.module} ({e})"]
        return [f"{self.module}.{s}" for s in self.symbols if not hasattr(m, s)]


# model_type (from config.json) -> variants in preference order
ARCHES = {
    "granitemoehybrid": [
        Variant("serialized_gen_t514.py",
                "transformers.models.granitemoehybrid.modeling_granitemoehybrid",
                ["GraniteMoeHybridExperts", "GraniteMoeHybridTopKRouter"],
                "transformers >= 5.14"),
        Variant("serialized_gen.py",
                "transformers.models.granitemoehybrid.modeling_granitemoehybrid",
                ["GraniteMoeHybridParallelExperts"],
                "transformers 5.5.x"),
    ],
    "qwen3_5_moe": [
        Variant("serialized_gen_qwen35.py",
                "transformers.models.qwen3_5_moe.modeling_qwen3_5_moe",
                ["Qwen3_5MoeExperts", "Qwen3_5MoeTopKRouter"],
                "transformers >= 5.5"),
    ],
    # Llama-4 configs carry model_type "llama4" at the top level and "llama4_text" under
    # text_config; the generator calls cfg.get_text_config(), so both must resolve here.
    "llama4": [
        Variant("serialized_gen_llama4.py",
                "transformers.models.llama4.modeling_llama4",
                ["Llama4TextExperts", "Llama4Router"],
                "transformers >= 5.5", status="experimental"),
    ],
    "llama4_text": [
        Variant("serialized_gen_llama4.py",
                "transformers.models.llama4.modeling_llama4",
                ["Llama4TextExperts", "Llama4Router"],
                "transformers >= 5.5", status="experimental"),
    ],
    "glm4_moe": [
        Variant("glm4moe_serialized_gen.py",
                "transformers.models.glm4_moe.modeling_glm4_moe",
                ["Glm4MoeExperts"],
                "a transformers with fused Glm4MoeExperts (not 5.5.x)",
                status="experimental"),
    ],
}


def read_model_type(model_dir):
    cfg_path = os.path.join(model_dir, "config.json")
    if not os.path.exists(cfg_path):
        raise SystemExit(f"dispatch: no config.json in {model_dir}")
    with open(cfg_path) as f:
        cfg = json.load(f)
    mt = cfg.get("model_type")
    # Multimodal wrappers keep the decoder arch under text_config. Prefer a top-level
    # type we actually support, and only fall through to text_config otherwise, so a
    # wrapper never resolves to an inner name the table does not know.
    if mt not in ARCHES and isinstance(cfg.get("text_config"), dict):
        mt = cfg["text_config"].get("model_type", mt)
    if not mt:
        raise SystemExit(f"dispatch: config.json in {model_dir} has no model_type")
    return mt


def select(model_type):
    """Return (script_path, variant). Raises SystemExit with an actionable message."""
    variants = ARCHES.get(model_type)
    if not variants:
        raise SystemExit(
            f"dispatch: model_type {model_type!r} has no serialized generator.\n"
            f"  supported: {', '.join(sorted(ARCHES))}\n"
            f"  use the default IMATRIX_BACKEND=llama for this model, or add a variant.")
    tried = []
    for v in variants:
        miss = v.missing()
        if not miss:
            return os.path.join(HERE, v.script), v
        tried.append(f"  {v.script}: needs {v.needs}; unavailable: {', '.join(miss)}")
    raise SystemExit(
        f"dispatch: model_type {model_type!r} is supported but no variant matches the "
        f"installed transformers.\n" + "\n".join(tried) +
        "\n  install a transformers matching one of the above "
        "(see scripts/imatrix_serialized/requirements.txt).")


def main():
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--model", required=True)
    ap.add_argument("--print-script", action="store_true")
    ap.add_argument("--help", "-h", action="store_true")
    args, passthrough = ap.parse_known_args()
    if args.help:
        print(__doc__)
        return

    model_type = read_model_type(args.model)
    script, variant = select(model_type)

    if args.print_script:
        print(script)
        return

    note = "" if variant.status == "validated" else f"  [{variant.status}]"
    print(f"dispatch: model_type={model_type} -> {os.path.basename(script)}{note}",
          flush=True)
    argv = [sys.executable, script, "--model", args.model] + passthrough
    os.execv(sys.executable, argv)


if __name__ == "__main__":
    main()
