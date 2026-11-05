# Frozen R1 implementation audit

Date: 2026-09-14

R1 status: **PASS**. This dated audit describes the six original matrices
and their frozen sources. It is not the completion certificate for the
second revision. R2 evidence is mapped in `R2_EVIDENCE_MAP.md`, with each
new campaign's raw-data checks in `analysis/audit.json`.

This audit records why the submitted results are produced by the canonical
request-level implementation rather than by a reduced aggregate surrogate.

## Executed evidence

- Main matrix: 618 complete source-locked runs
- Network-stress extension: 75 complete source-locked runs
- Native load-scaling extension: 30 complete source-locked runs
- Deadline-ceiling diagnostic: 60 complete source-locked runs
- Native low-load breakdown grid: 165 complete source-locked runs
- Physical-startup sensitivity: 180 complete source-locked request-level runs
- Total registered full-trace runs: 1,128
- Restricted exact replay: 81,920 enumerated action sequences
- Implementation conformance: 49 of 49 checks passed
- Canonical and physical regression suite: 128 of 128 tests passed

The final campaign sources are immutable relative to their protocol records.
Their SHA-256 values are:

- `strict_simulate.py`: `14f4cf752eac9b4754c53a74bcd8968c94429c9b5ebba26e1c8edec66e9c6492`
- `scheduler_core.py`: `6c1411e697cf312e0fc96a8ad3bd01e566750e1e80d381bbafebc53cd05f04cb`
- `startup_profiles.py`: `55045f177fdda1b0f68e02c9dc7df30a8d789b8b3965e801e770f1ba12934d49`
- `run_measured_startup_strict_extension.py`: `458e7af97a7834d16692cbcbaeead64f1ae697e5082af0167ed566ceb79727fb`
- deployment JSON: `8ca317116dd5c2afc6050b16a680b1c3265f854eecae2a1619d66e2c01fb6c03`

## Semantics that are implemented directly

1. One retained source invocation is one event. There is no event weight,
   request multiplicity, or weighted micro-batch in the canonical path.
2. All 9,000 raw function identities remain available to cache state. The 16
   strata are used for forecasting, reporting, and one labelled sensitivity.
3. Cache access is tested at execution start. Per-device byte capacity, TTL
   expiry, incompatibility, and LRU eviction are active for predicted and
   realized state separately.
4. Shared ingress, optional inter-switch, and per-device egress resources use
   store-and-forward FCFS calendars. Cross traffic is read at each hop's
   service start. Jitter consumes the shared resource.
5. Predicted state never reads a future realized completion timestamp.
   Completion feedback is processed only after it becomes observable.
6. Service duration is sampled from the stored per-function, per-day empirical
   quantiles. Startup, service, power, and network uncertainty use separate
   request-level random streams shared across compared methods.
7. Temperature follows the exact first-order exponential solution across idle,
   startup, and service phases. Predicted screening checks all phase endpoints.
   Realized service derating is evaluated at actual execution start.
8. Device and device-tenant joule balances, virtual debts, current-window
   commitments, and the per-tenant 4,096-request screening ceiling all evolve
   in fixed 60-second accounting windows.
9. Rejected requests remain failures in offered success. Strict and deadline
   ceiling outcomes retain separate integer numerators and denominators.
10. Receiver-order AoI ignores stale deliveries and excludes fabricated peaks
    at first initialization.
11. Task-attributed energy and fixed-window supplied energy are computed at
    different declared boundaries. Activity overlap uses an exact half-open
    interval and external-power measurements are never imputed as zero.
12. SCD and matched DPP share the score, predictions, debts, eligibility, and
    tie break. Only the strict feasible-candidate gate differs.

## Fail-closed controls

- Every campaign refuses mixed source, deployment, forecast, or trace hashes.
- Missing tasks, duplicate task keys, inconsistent integer numerators, and
  invalid ratios fail the campaign summary.
- `prepare_primary_traces.py` rebuilds or verifies all 16 main and sampling
  control traces. Each manifest contains per-array hashes.
- The table and figure generation pipeline consumes only the six final campaign
  directories. The measured-startup table comes from the sixth protocol and the
  plotted figures use the five declared figure campaigns. Output provenance hashes every table,
  vector figure, raster preview, and source CSV.
- The submission audit checks that manuscript table fragments and figure files
  exactly match those generated outputs.
- The portable artifact uses a credential-free allowlist and excludes legacy
  aggregate simulators, LAN administration records, and historical manuscript
  assets.

## Evidence boundary

Payload size, synthetic cross traffic, and modeled H3C queue residence remain
declared simulation inputs. RDK X5 and Jetson Nano startup and power values are
labelled sensitivity brackets. The 280 physical scheduling trials execute one
trained MLP and a finite-batch controller adapter. They validate deployability,
transport, board service, thermal checking, and measured PoE accounting within
that protocol. They are not relabelled as a hardware deployment of 9,000 raw
function containers or arbitrary multi-hop topologies.
