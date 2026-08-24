# Correction Validation

The corrected environment parser and scaling were run against all 11,612 local environments. Only deterministic 512-environment kernel blocks were saved under `audit/`; production matrices were not overwritten.

| Kernel | Mean absolute delta | Max absolute delta | Off-diagonal correlation | Materially changed |
|---|---:|---:|---:|---|
| K_geo | 3.84173e-05 | 0.000588894 | 1 | True |
| K_weather | 2.91161e-05 | 0.000939369 | 1 | True |
| K_stress | 3.75855e-05 | 0.000897408 | 1 | True |
| K_mgmt | 0.0318543 | 104.811 | 0.961541 | True |
| K_E | 0.00796765 | 26.2029 | 0.998266 | True |

Complete raw date parsing: SOWING_DATE 10,959/10,959, EMERGENCE_DATE 8,197/8,197, HARVEST_STARTING_DATE 9,392/9,392, HARVEST_FINISHING_DATE 9,262/9,262.
Categorical/product rows selected by the parser audit: 0/19,995 parsed as numeric (expected zero).
Semantic parser hash: `3931ea72bc59365d6bcd457913166d79744ea5b5b181111d3b82a34cc620c2eb`.

The relevant quantitative baseline must be regenerated on the server after full corrected K_E and K_A artifacts are built. The full stage-1 pedigree/multitrait ledger is not present locally.
