# Evidence map for the second revision

Paths below are relative to this code directory. Tables embedded in the
response use the same generated TeX fragments as the paper or supplement.
The internal response itself is not included in the public release.

| Evidence | Authoritative records | Reproduction or analysis |
|---|---|---|
| Primary nominal comparison and main Figure 2 | `results/strict/campaign_formula_v7_20260914` | `frozen_r1_src/summarize_strict_campaign.py`, `strict_paper_tables.py`, `strict_paper_figures.py` |
| Sampling, finite cache, TTL and main Figure 3 | Same main matrix, keyed families | Same source snapshot and generators |
| Main Figure 4 and shared-network stress | `results/strict/network_stress_formula_v4_20260914` plus primary records | `strict_paper_figures.py` |
| Native load scaling | `results/strict/load_scaling_formula_v2_20260914` | Registered load runner and original statistics |
| Eleven-point breakdown grid | `results/strict/breakdown_formula_v1_20260914` | Registered breakdown runner and original statistics |
| Post-hoc strict/grace deadline diagnostic | `results/strict/deadline_guard_network_stress_v4_20260914` | Registered deadline runner and original statistics |
| Measured startup profile sensitivity | `results/strict/measured_startup_strict_v1_20260914` | Registered startup runner and original statistics |
| Restricted exact-state benchmark | `results/strict/exact_benchmark_suite_v2_20260914` | `run_exact_benchmark_suite.py` and `strict_optimal_benchmark.py` |
| Original service and transport calibration | `results/revision/physical/inference_calibration_v2_20260905`, `inference_calibration_port04_repeat_20260909`, `activation_transport_isolated_20260905` | Corresponding raw JSONL, summary scripts, and timing CSVs |
| Original power and held-out thermal fit | `results/revision/physical/power_thermal_20260905` | `summarize_power_thermal.py`, `fit_thermal_calibration.py` |
| Original 280 physical scheduler trials | `results/revision/physical/physical_scheduler_full_v3_complete_20260906` | Frozen `summarize_physical_trials.py`, decision events, fixed-window PSE logs |
| 160 paired host/container activations | `results/revision/physical/container_startup_full_v6_complete_20260906` | `summarize_container_trials.py` and all activation records |
| Practical additional-success counts and tail/AoI differences | `results/r2/primary_context` | `src/analyze_r2_primary_context.py` using original primary seed pairs |
| 200-run component and allocation factorial | `results/r2/simulation_factorial_20261105` | `src/analyze_r2_simulations.py`, all 200 raw keyed records, global 280-test Holm family |
| Three trained services | `results/revision/physical/portable_model_20260905`, `results/r2/trained_services_20261105` | Portable inference and complete training/export scripts |
| 2,520 new service calibration groups | `results/r2/service_calibration` | `src/analyze_r2_calibration.py` |
| 560 isolated transfers | `results/r2/matched_physical_v4/isolated_transport` | Raw payload repetitions and nonnegative fitted models |
| 140 common-constraint scheduler trials | `results/r2/matched_physical_v4` | `src/analyze_r2_physical.py`, 40-test primary Holm family |
| Interrupted acquisition blocks | `results/r2/acquisition_interruption_20261105` and formal `resumption.json` | Whole-block exclusion/repetition records and all original failure evidence |
| Descriptive interruption sensitivity | `results/r2/interruption_sensitivity` | `src/analyze_r2_interruption_sensitivity.py`, all complete conditions from interrupted attempts, averaged within the original ten blocks |
| 60 actual-switch trials | `results/r2/switch_contention_v3` | `src/analyze_r2_switch.py`, raw probes and interface counters, four-test Holm family |
| Interrupted switch acquisition | `results/r2/switch_interruption_20261105` and formal `resumption.json` | First five complete blocks retained byte for byte, entire sixth block repeated after a pre-trial connection failure |

The original trace-scale matrices contain 1,128 executions. Adding the
200-run factorial gives 1,328 executions. Stochastic design cells use five
seeds. The deterministic-count rounding control has one fixed reconstruction
per policy and no across-seed interval. Flooding outcomes use the final day
alone, while the other trace-scale conditions report days 10--13.
The repeated nominal factorial records do not increase the independent
replicate count. The simulation uncertainty is conditional on the fixed
source cohort and modeled deployment. Hardware blocks are separate
within-deployment replicates.

The generated `*_table.tex`, `*_results.tex`, macro files, CSVs, and plot
source contracts identify the quantities displayed. Result records retain
scenario parameters, source hashes, trace-manifest hashes, and integer
numerators. The release includes every declared condition and method,
including zero service, null differences, and adverse outcomes.
