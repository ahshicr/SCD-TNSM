# Model-to-implementation traceability

This ledger identifies the canonical implementation of each manuscript
mechanism. It prevents a result from being supported by a legacy aggregate
simulator or by the reduced physical adapter. Paths are relative to this code
root. The campaign protocol records hashes for every runtime source. The
original six matrices use the matching files in `frozen_r1_src`. The
additional factorial and live protocols have separate immutable snapshots.

| Manuscript mechanism | Canonical implementation | Independent check | Recorded evidence |
|---|---|---|---|
| Native request events and raw function identity | `src/strict_trace.py::build_request_trace`, `src/prepare_primary_traces.py` | `tests/test_primary_trace_protocol.py`, `tests/test_strict_components.py`, conformance audit | `primary_trace_registry.json`, trace `manifest.json`, and per-array SHA-256 |
| Binomial and unbiased thinning | `src/strict_trace.py::_sample_sparse_cells` | `tests/test_primary_trace_protocol.py`, `tests/test_revision_artifact.py`, workload audit | Sampling method, seed, expected count, retained count |
| Half and double offered load | `src/prepare_load_scaling_traces.py`, `src/run_load_scaling_extension.py` | load protocol and trace hash audit | Native $p=2\times10^{-4}$ and $8\times10^{-4}$ request traces |
| Resolution-bounded breakdown point | `src/prepare_breakdown_traces.py`, `src/run_breakdown_extension.py` | `tests/test_breakdown_protocol.py` and protocol hash audit | Eleven native probabilities, five seeds, three methods, integer miss counts |
| Continuous within-minute releases | `src/strict_trace.py::build_request_trace` | conformance audit and arrival reconstruction family | Release model and immutable `release_s.npy` |
| Application-coherent logical tenants | `src/strict_trace.py::_tenant_from_application` | conformance audit | `app_id.npy`, `tenant_id.npy`, manifest semantics |
| Per-request common random numbers | `src/strict_trace.py::build_request_trace` | trace hash verification | Service, network, and startup arrays shared by methods |
| Capability and memory compatibility | `src/strict_simulate.py::build_requirement_mask`, `_simulate_kernel` | `tests/test_strict_simulator.py`, compatibility family | Per-node capability mask and rejection counts |
| Nonlinear residual link capacity | `src/strict_simulate.py::_route_preview`, `_route_commit` | `tests/test_strict_simulator.py`, network families | Queue and jitter metrics plus scenario dictionary |
| Shared multi-hop FCFS resources | `src/strict_simulate.py::build_scenario_topology`, `_route_commit` | conformance audit, independent-network ablation | Route arrays and network-stress protocol |
| Cross traffic at each hop start | `src/strict_simulate.py::_route_preview`, `_route_commit` | service-start counterexample test and conformance audit | Background seed and per-hop timing semantics |
| Separate predicted and realized link state | `src/strict_simulate.py::_simulate_kernel` | causality tests | Protocol state semantics and saved primary outcomes |
| Execution-start TTL decision | `src/strict_simulate.py::_cache_status` | `tests/test_strict_components.py`, conformance audit | Predicted and realized cold-start numerators |
| Physical startup distributions | `src/startup_profiles.py`, `src/run_measured_startup_strict_extension.py` | `tests/test_container_trials.py`, campaign audit | 180 source-locked profile--TTL--method--seed runs |
| Raw-function finite cache | `src/strict_simulate.py::build_cache_model`, `_cache_commit` | cache identity and capacity families | Cache mode, slots, memory multiplier, evictions |
| Byte capacity, expiry, and LRU eviction | `src/strict_simulate.py::_cache_commit` | eviction counterexamples | Per-device usable memory and eviction counts |
| Empirical service-duration sampling | `src/strict_simulate.py::_empirical_duration_ms` | percentile and support tests | Trace quantile arrays and imputation mask |
| Temperature-dependent service | `src/strict_simulate.py::_thermal_phase`, `_thermal_job`, `_thermal_segment_peak` | `tests/test_thermal_dynamics.py`, `tests/test_strict_simulator.py` | Robust-power feasibility, fixed-window peaks, fit labels, derating counts |
| Causal prequential pressure forecast | `src/strict_simulate.py::load_service_pressure` | `tests/test_prequential_forecast.py` | Forecast SHA-256 and model-origin metadata |
| Predicted and fixed-window supplied energy | `src/strict_simulate.py::_simulate_kernel`, `_interval_overlap` | overlap and boundary tests | Task energy, supplied energy, numerator fields |
| Device and device-tenant credit buckets | `src/strict_simulate.py::_advance_bucket` | `tests/test_scheduler_core.py` | Guard settings, rejections, shortfall, balances |
| Device and device-tenant virtual debt | `src/strict_simulate.py::_advance_debt` | no-debt test and debt identities | Maximum and final debt fields |
| Exact per-tenant request ceiling | `src/strict_simulate.py::_simulate_kernel` | request-cap tests and conformance audit | Integer cap and 60-second window in every result |
| Strict versus ceiling deadlines | `src/strict_simulate.py::_request_deadline`, `simulate` | deadline tests | Separate integer successes and ratios |
| SCD analytical score | `src/scheduler_core.py::operating_penalty`, `candidate_score` | bound and score tests | Scheduler hash and fixed deployment coefficients |
| Matched DPP comparator | same score functions with only the gate removed | end-to-end identical-when-gate-inactive test | Matched scenario keys and paired seed analysis |
| Receiver-order peak AoI | `src/strict_simulate.py::_receiver_aoi` | stale and initialization counterexamples | Peak sample count, mean, and p95 |
| Held-out scoring with state warm-up | `src/strict_simulate.py::simulate` | campaign protocol audit | Day 0 state evolution and days 10--13 metric bounds |
| Exact restricted-instance optimization | `src/strict_optimal_benchmark.py` | `src/run_exact_benchmark_suite.py` | 81,920 enumerated assignments and source hashes |
| Physical calibration and replay | `src/board_inference_worker.py`, `src/run_physical_trials.py` | physical trace and parity audits | Raw timings, temperatures, PoE samples, session manifests |
| Full debt-feedback removal | `src/scheduler_core.py`, `src/strict_simulate.py`, `src/run_r2_simulations.py` | R2 mechanism tests and 200-run factorial | Separate explicit-price and complete-feedback methods |
| Elastic tenant energy admission | `src/strict_simulate.py`, `src/run_r2_simulations.py` | R2 device-envelope and tenant-borrowing tests | Fixed, elastic, and shared modes under identical device budgets |
| Practical primary effect counts and tail/AoI contrasts | `src/analyze_r2_primary_context.py` | Same-seed integer numerator and denominator checks | `results/r2/primary_context` and shared prose macros |
| Matched physical constraints | `src/run_r2_physical.py`, `src/physical_policy.py` | Common-job, credit, safety and parity reconstruction | 140-trial manifest, complete decision logs, 40-test Holm family |
| Complete CNN and forest inference | `src/train_r2_workloads.py`, `src/r2_portable_models.py` | Framework/export parity and per-board numerical checks | Weights, split indices, losses, 2,520 calibration groups |
| Model-specific physical inputs | `src/run_physical_trials.py::select_service_batch` | Mixed-service input-routing regression test | Per-service input and output parity in every trial |
| Actual Ethernet contention | `src/run_r2_switch_experiment.py`, `src/r2_udp_remote.py` | Packet conservation, negotiated-rate and counter checks | 60 trials, every probe and interface reply, four-test Holm family |
| Whole-block acquisition recovery | `src/resume_r2_physical.py` | Unchanged source/input hashes and retained-record verification | Interrupted attempts, resumption rule, repeated full block |

The physical replay is intentionally listed last. It exercises real transport,
inference, temperature checks, and the shared analytical score. It does not
replace the request-level simulator for raw-function cache or arbitrary
multi-hop state. The manuscript and response letter must preserve this evidence
boundary.
