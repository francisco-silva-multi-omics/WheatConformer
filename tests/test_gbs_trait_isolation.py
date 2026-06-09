from pathlib import Path

def test_gbs_wrapper_resolves_and_passes_one_trait():
    script = (Path(__file__).resolve().parents[1] / "scripts" / "03_run_training.sh").read_text(encoding="utf-8")
    assert "GBS_TRAIN_TRAITS:-${TRAIN_TRAITS:-}" in script
    assert '--trait "$trait"' in script
    assert 'trained_models/stage1_gbs_sawyt_mkl/$trait_slug' in script
