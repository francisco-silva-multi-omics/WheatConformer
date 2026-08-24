from build_baseline import hmp_genotype_to_dosage


def test_hmp_iupac_and_missing_encoding() -> None:
    assert hmp_genotype_to_dosage("A", "A/G") == 0
    assert hmp_genotype_to_dosage("G", "A/G") == 2
    assert hmp_genotype_to_dosage("R", "A/G") == 1
    assert hmp_genotype_to_dosage("AG", "A/G") == 1
    assert hmp_genotype_to_dosage("N", "A/G") == -9
