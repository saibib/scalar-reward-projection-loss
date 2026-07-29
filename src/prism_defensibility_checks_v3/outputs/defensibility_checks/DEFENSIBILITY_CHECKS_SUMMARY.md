# PRISM Condorcet Defensibility Checks Summary

Parsed pairwise comparison rows: 169,224

## Parse audit status counts
| status         |   count |
|:---------------|--------:|
| ok             |    7990 |
| too_few_models |      21 |
Aggregation: user
Canonical null/check setting: score_delta=5.0, n_min=25, tau=0.0

## Attribute missingness/coverage
| attribute          |   n_conversations |   non_null |   missing |   missing_rate |    mean |   q25 |   median |   q75 |   n_high_q75 |   n_low_q25 |   n_users_high_q75 |   n_users_low_q25 |
|:-------------------|------------------:|-----------:|----------:|---------------:|--------:|------:|---------:|------:|-------------:|------------:|-------------------:|------------------:|
| choice_values      |              7508 |       6628 |       880 |     0.117208   | 66.5412 |    50 |       71 |    88 |         1724 |        1783 |                765 |               816 |
| choice_fluency     |              7508 |       7456 |        52 |     0.00692595 | 82.3899 |    72 |       86 |   100 |         1893 |        1919 |                674 |               823 |
| choice_factuality  |              7508 |       7354 |       154 |     0.0205115  | 79.0783 |    68 |       84 |    98 |         1888 |        1937 |                734 |               880 |
| choice_safety      |              7508 |       6912 |       596 |     0.079382   | 71.6745 |    55 |       78 |    97 |         1803 |        1734 |                648 |               717 |
| choice_diversity   |              7508 |       6668 |       840 |     0.111881   | 65.5204 |    50 |       69 |    87 |         1678 |        1871 |                738 |               822 |
| choice_creativity  |              7508 |       6787 |       721 |     0.0960309  | 61.6483 |    49 |       64 |    83 |         1726 |        1716 |                735 |               731 |
| choice_helpfulness |              7508 |       7447 |        61 |     0.00812467 | 82.4038 |    72 |       88 |   100 |         2129 |        1908 |                763 |               831 |

## High vs low attribute cyclic residuals, canonical setting
| attribute          | conditioning   |   n_models |   n_edges |   rho_cyc |
|:-------------------|:---------------|-----------:|----------:|----------:|
| choice_factuality  | low_q25        |         20 |        79 | 0.368865  |
| choice_fluency     | low_q25        |         19 |        77 | 0.219583  |
| choice_helpfulness | low_q25        |         19 |        83 | 0.216437  |
| choice_diversity   | low_q25        |         20 |        81 | 0.193536  |
| choice_values      | low_q25        |         19 |        61 | 0.173627  |
| choice_safety      | low_q25        |         18 |        51 | 0.143393  |
| choice_creativity  | low_q25        |         18 |        49 | 0.140212  |
| choice_safety      | high_q75       |         19 |        77 | 0.138511  |
| choice_creativity  | high_q75       |         19 |        56 | 0.130615  |
| choice_diversity   | high_q75       |         19 |        66 | 0.111472  |
| choice_values      | high_q75       |         19 |        56 | 0.110647  |
| choice_factuality  | high_q75       |         20 |        88 | 0.107727  |
| choice_fluency     | high_q75       |         20 |        87 | 0.0889325 |
| choice_helpfulness | high_q75       |         21 |       108 | 0.0828695 |

## Bootstrap summary
| setting            | attribute          | conditioning   |   score_delta |   n_min |   tau |   cycle_rate_mean |   cycle_rate_q025 |   cycle_rate_q500 |   cycle_rate_q975 |   cyclic_triples_mean |   cyclic_triples_q025 |   cyclic_triples_q500 |   cyclic_triples_q975 |   rho_cyc_mean |   rho_cyc_q025 |   rho_cyc_q500 |   rho_cyc_q975 |   bootstrap_reps |   threshold |
|:-------------------|:-------------------|:---------------|--------------:|--------:|------:|------------------:|------------------:|------------------:|------------------:|----------------------:|----------------------:|----------------------:|----------------------:|---------------:|---------------:|---------------:|---------------:|-----------------:|------------:|
| pooled             |                    | pooled         |             5 |      25 |     0 |         0.0245602 |        0.0142857  |         0.0240602 |         0.0376504 |                32.665 |                19     |                    32 |                50.075 |      0.0654978 |      0.0561805 |      0.0650584 |      0.0768692 |              200 |         nan |
| attribute_high_q75 | choice_values      | high_q75       |             5 |      25 |     0 |         0.0622922 |        0          |         0.0564709 |         0.140414  |                 4.405 |                 0     |                     4 |                10.025 |      0.18033   |      0.118912  |      0.176258  |      0.270634  |              200 |          88 |
| attribute_high_q75 | choice_fluency     | high_q75       |             5 |      25 |     0 |         0.0480969 |        0.00767717 |         0.0458365 |         0.100224  |                 6.485 |                 0.975 |                     6 |                14     |      0.148099  |      0.101021  |      0.146887  |      0.216144  |              200 |         100 |
| attribute_high_q75 | choice_factuality  | high_q75       |             5 |      25 |     0 |         0.0607901 |        0.0179316  |         0.0598507 |         0.108981  |                 9.255 |                 2     |                     9 |                19.025 |      0.152424  |      0.116228  |      0.150781  |      0.194065  |              200 |          98 |
| attribute_high_q75 | choice_safety      | high_q75       |             5 |      25 |     0 |         0.0716056 |        0.0205198  |         0.0708705 |         0.136848  |                 7.54  |                 2     |                     7 |                16.025 |      0.190127  |      0.123911  |      0.186397  |      0.255501  |              200 |          97 |
| attribute_high_q75 | choice_diversity   | high_q75       |             5 |      25 |     0 |         0.0577479 |        0.0113543  |         0.0579111 |         0.123744  |                 5.285 |                 1     |                     5 |                12     |      0.162236  |      0.10495   |      0.161563  |      0.226725  |              200 |          86 |
| attribute_high_q75 | choice_creativity  | high_q75       |             5 |      25 |     0 |         0.0716254 |        0.0109551  |         0.0681818 |         0.147569  |                 4.395 |                 0.975 |                     4 |                10.025 |      0.182889  |      0.118545  |      0.181831  |      0.251921  |              200 |          83 |
| attribute_high_q75 | choice_helpfulness | high_q75       |             5 |      25 |     0 |         0.0542221 |        0.0195849  |         0.0555556 |         0.0883357 |                13.87  |                 5     |                    14 |                24     |      0.152218  |      0.11385   |      0.151883  |      0.203088  |              200 |         100 |

## Transitive null summary
| name                        | attribute          | conditioning   |   observed_cycle_rate |   null_cycle_rate_mean |   null_cycle_rate_q95 |   p_ge_observed_cycle_rate |   z_observed_cycle_rate |   observed_hodge_rho_cyc |   null_hodge_rho_cyc_mean |   null_hodge_rho_cyc_q95 |   p_ge_observed_hodge_rho_cyc |   z_observed_hodge_rho_cyc |
|:----------------------------|:-------------------|:---------------|----------------------:|-----------------------:|----------------------:|---------------------------:|------------------------:|-------------------------:|--------------------------:|-------------------------:|------------------------------:|---------------------------:|
| pooled                      |                    | pooled         |             0.0150376 |              0.012891  |             0.0210902 |                   0.363184 |               0.403418  |                0.0421803 |                 0.0367551 |                0.0421255 |                     0.0497512 |                  1.65957   |
| choice_values_high_q75      | choice_values      | high_q75       |             0.0615385 |              0.0433846 |             0.0923077 |                   0.343284 |               0.610467  |                0.110647  |                 0.109771  |                0.147022  |                     0.472637  |                  0.036627  |
| choice_fluency_high_q75     | choice_fluency     | high_q75       |             0.0115607 |              0.0363006 |             0.0696532 |                   0.960199 |              -1.237     |                0.0889325 |                 0.110176  |                0.137164  |                     0.915423  |                 -1.30496   |
| choice_factuality_high_q75  | choice_factuality  | high_q75       |             0.0641711 |              0.0384225 |             0.0748663 |                   0.124378 |               1.28637   |                0.107727  |                 0.0953488 |                0.120463  |                     0.19403   |                  0.79793   |
| choice_safety_high_q75      | choice_safety      | high_q75       |             0.0373134 |              0.0380597 |             0.075     |                   0.537313 |              -0.0326916 |                0.138511  |                 0.134349  |                0.17736   |                     0.348259  |                  0.172808  |
| choice_diversity_high_q75   | choice_diversity   | high_q75       |             0.0108696 |              0.0332609 |             0.076087  |                   0.920398 |              -1.06305   |                0.111472  |                 0.101322  |                0.131895  |                     0.288557  |                  0.541325  |
| choice_creativity_high_q75  | choice_creativity  | high_q75       |             0.0819672 |              0.0567213 |             0.114754  |                   0.283582 |               0.826028  |                0.130615  |                 0.133334  |                0.184536  |                     0.487562  |                 -0.0947447 |
| choice_helpfulness_high_q75 | choice_helpfulness | high_q75       |             0.0477941 |              0.0390993 |             0.0698529 |                   0.323383 |               0.491136  |                0.0828695 |                 0.101317  |                0.127813  |                     0.925373  |                 -1.31723   |
