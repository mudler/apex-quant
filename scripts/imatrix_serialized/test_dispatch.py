"""Checks the Phase 6 generator dispatch. No checkpoint, no GPU.

Guards the blocking issue from review round 3: apex_pipeline.sh Phase 6 called
serialized_gen.py unconditionally, which is GraniteMoeHybrid-only and 5.5-only, so a
fresh install per requirements.txt ImportError'd and the qwen3_5_moe case that
motivates the backend could never run through the wired path.

Run: python3 test_dispatch.py   (or under pytest)
"""
import json
import os
import tempfile

import dispatch


def _model_dir(d, cfg):
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    return d


def test_every_arch_entry_points_at_a_file_that_exists():
    """A table entry naming a missing script would fail only at run time."""
    for mt, variants in dispatch.ARCHES.items():
        for v in variants:
            p = os.path.join(dispatch.HERE, v.script)
            assert os.path.exists(p), f"{mt}: {v.script} does not exist"


def test_selection_picks_a_variant_the_install_supports():
    """Whatever transformers is installed, a resolved variant must be importable."""
    import importlib
    for mt in dispatch.ARCHES:
        try:
            script, variant = dispatch.select(mt)
        except SystemExit:
            continue                      # nothing installed for this arch; fine
        m = importlib.import_module(variant.module)
        for sym in variant.symbols:
            assert hasattr(m, sym), f"{mt}: chose {variant.script} but {sym} is missing"
        assert os.path.exists(script)


def test_unknown_model_type_fails_with_supported_list():
    try:
        dispatch.select("definitely_not_an_arch")
    except SystemExit as e:
        msg = str(e)
        assert "no serialized generator" in msg
        assert "supported:" in msg
        # must point at the escape hatch rather than just dying
        assert "IMATRIX_BACKEND=llama" in msg
        return
    raise AssertionError("expected SystemExit for an unknown model_type")


def test_unavailable_variant_names_what_is_missing():
    """A supported arch with no importable variant must say what to install."""
    saved = dict(dispatch.ARCHES)
    try:
        dispatch.ARCHES["fake_arch"] = [
            dispatch.Variant("serialized_gen.py", "transformers", ["NoSuchSymbol"],
                             "transformers with NoSuchSymbol")]
        try:
            dispatch.select("fake_arch")
        except SystemExit as e:
            msg = str(e)
            assert "NoSuchSymbol" in msg, msg
            assert "transformers with NoSuchSymbol" in msg, msg
            return
        raise AssertionError("expected SystemExit when no variant is importable")
    finally:
        dispatch.ARCHES.clear()
        dispatch.ARCHES.update(saved)


def test_missing_module_is_reported_not_raised():
    v = dispatch.Variant("serialized_gen.py", "no_such_module_xyz", ["X"], "n/a")
    miss = v.missing()
    assert miss and "no_such_module_xyz" in miss[0], miss


def test_model_type_read_from_config_and_text_config():
    with tempfile.TemporaryDirectory() as d:
        plain = _model_dir(os.path.join(d, "a"), {"model_type": "qwen3_5_moe"})
        assert dispatch.read_model_type(plain) == "qwen3_5_moe"
        # multimodal wrapper with a SUPPORTED top-level type keeps the top-level name
        nested = _model_dir(os.path.join(d, "b"),
                            {"model_type": "llama4", "text_config": {"model_type": "llama4_text"}})
        assert dispatch.read_model_type(nested) == "llama4"
        # ...but an unsupported wrapper falls through to the inner decoder arch
        wrapped = _model_dir(os.path.join(d, "c"),
                             {"model_type": "some_vlm", "text_config": {"model_type": "qwen3_5_moe"}})
        assert dispatch.read_model_type(wrapped) == "qwen3_5_moe"


def test_whatever_read_model_type_returns_is_dispatchable():
    """A model_type that parses but is absent from ARCHES is a silent dead end."""
    with tempfile.TemporaryDirectory() as d:
        for cfg in ({"model_type": "llama4", "text_config": {"model_type": "llama4_text"}},
                    {"model_type": "llama4_text"},
                    {"model_type": "qwen3_5_moe"},
                    {"model_type": "granitemoehybrid"}):
            md = _model_dir(os.path.join(d, str(abs(hash(str(cfg))))), cfg)
            mt = dispatch.read_model_type(md)
            assert mt in dispatch.ARCHES, f"{cfg} -> {mt!r} not in ARCHES"


def test_missing_config_fails_clearly():
    with tempfile.TemporaryDirectory() as d:
        try:
            dispatch.read_model_type(d)
        except SystemExit as e:
            assert "no config.json" in str(e)
            return
    raise AssertionError("expected SystemExit when config.json is absent")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
