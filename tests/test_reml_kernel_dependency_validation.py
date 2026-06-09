from argparse import Namespace
import pytest
pytest.importorskip("scipy")
from server_training_pipeline.fit_multikernel_reml import validate_kernel_dependencies

@pytest.mark.parametrize("flag,message", [
    ("include_rbf_e", "--include-rbf-e requires --k-g-rbf"),
    ("include_epi2", "--include-epi2 requires --geno-epi2-kernel"),
    ("include_epi2_e", "--include-epi2-e requires --geno-epi2-kernel"),
    ("include_ae", "--include-ae requires --k-a"),
    ("include_ze", "--include-ze requires --k-z"),
])
def test_missing_interaction_dependency_fails(flag, message):
    args = Namespace(include_rbf_e=False, include_epi2=False, include_epi2_e=False, include_ae=False, include_ze=False,
                     k_g_rbf=None, geno_epi2_kernel=None, k_a=None, k_z=None)
    setattr(args, flag, True)
    with pytest.raises(SystemExit, match=message):
        validate_kernel_dependencies(args)
