# Implementation and evidence scope

This file separates implemented controller semantics from calibrated inputs and
from the original and common-constraint board-trial adapters. It is part of
the reproducibility record. Original trace matrices use `frozen_r1_src`.
The added factorial uses its recorded R2 execution snapshot.

## Canonical request-level implementation

The manuscript algorithm is executed by `src/strict_simulate.py` with the
shared score primitives in `src/scheduler_core.py`. One retained invocation is
one simulation event. The canonical path does not aggregate those invocations
into weighted jobs and does not use the retired seven-weight surrogate.

The active implementation includes all of the following.

- Raw Azure function identity for container residency, with a separately
  labelled 16-stratum sensitivity
- Cache access at predicted or realized execution start, not request release
- Per-device byte capacity, TTL expiry, and least-recently-used eviction
- Separate predicted and realized cache, network, compute, thermal, and energy
  state
- Store-and-forward FCFS reservations for every link in the selected route
- A shared link calendar across destinations, plus a deliberately incorrect
  independent-state ablation
- Nonlinear residual capacity and stochastic per-hop service occupancy
- Cross-traffic lookup at each hop's actual or predicted service start, with a
  declared last-value hold only after the finite exogenous trace ends
- Execution-start thermal derating and hard predicted-temperature screening
- Device and tenant joule credits, virtual energy debts, and current-window
  commitments
- Optional elastic tenant borrowing with unchanged device eligibility,
  private request tokens, and full tenant debt pricing
- Separate removal of explicit debt prices, all debt feedback, and hard
  energy eligibility, without conflating them with request screening
- A 4,096-request per-tenant ceiling in fixed 60-second accounting windows
- Deadlines that count every rejected offered request as unsuccessful
- Separate strict-deadline and declared deadline-ceiling numerators and ratios
- A past-only prequential service-pressure predictor
- SCD and matched DPP using the same analytical score, with only SCD applying
  the feasible-candidate gate
- State warm-up from trace day 0 before held-out scoring on days 10 through 13
- Receiver-order peak AoI that excludes the first delivery of each raw function
  and ignores stale out-of-order deliveries
- Attributed task energy and fixed-window supplied energy as two noninterchangeable
  measurements, with exact half-open activity-overlap integration
- Integer numerators for success, deadline, cache, power, thermal, and AoI ratios

`src/audit_implementation_conformance.py` checks 43 executable invariants plus one
source-lock check for each completed campaign protocol and verifies
that completed campaign protocols still match the exact source files used for
execution. The pytest suite independently exercises request conservation,
cache timing and eviction, service-start cross traffic, thermal derating,
deadline ceilings, matched DPP, admission screening, AoI, and wall-clock energy.
`src/compare_strict_campaigns.py` distinguishes scientific output
changes from expected source-hash and runtime changes after a refactor.
`IMPLEMENTATION_TRACEABILITY.md` maps every manuscript mechanism to its
canonical function, independent test, and recorded evidence.
`src/prepare_primary_traces.py` rebuilds or hash-verifies all 16 primary and
sampling-control traces and records them in `primary_trace_registry.json`.

The registered main matrix contains 618 request-level runs in 20 experiment
families. The separate network-stress, deadline-guard, and load-scaling
protocols contain 75, 60, and 30 runs. A reviewer-requested native-resolution
breakdown grid contains another 165 runs over eleven thinning probabilities,
five seeds, and three methods. A sixth protocol contains 180 request-level
physical-startup sensitivity runs over three measured or bracketed profiles,
four TTLs, three methods, and five same-trace seeds. Each protocol stores its exact execution-source hashes, deployment
hash, trace manifests, task keys, and runtime package versions before execution.
An independent small-instance oracle enumerates 16,384 assignments per seed over
five seeds and replays routing, byte-limited cache, thermal, service, energy, and
rejection state. This verifies finite-state optimization on a deliberately
restricted instance and is not substituted for the trace-scale experiments.
The added four-condition, ten-policy factorial contains 200 full runs, for
1,328 executions in total. Fifteen repeated nominal records reproduce the
original primary metrics exactly and do not increase the five-seed sample
size. Its seven outcomes and all 280 registered contrasts use one Holm family.

## Inputs that remain declared models

The public Azure Functions trace has invocation counts, durations, memory, and
trigger metadata. It does not contain request payload size, Ethernet cross
traffic, switch buffer residence, or board power. The following are therefore
declared model inputs rather than fabricated measurements.

- Payload bytes are a fixed function of configured memory and are varied by a
  registered multiplier.
- Background traffic is a seeded exogenous process and is varied in network
  stress experiments.
- Link queueing is generated by the explicit FCFS resource model. It is not
  labelled as measured H3C buffer residence.
- Power and startup coefficients that are transferred between same-model boards
  retain their transfer labels.
- The original request-level RDK X5 and Jetson Nano power and container-startup
  inputs retain their declared assumptions and startup sensitivity brackets.
  The new RDK live adapter uses fixed declared 4-W idle and 3-W incremental
  admission coefficients. Its CPU service is measured, while its individual
  supply power and container startup are not newly measured.

These inputs affect the empirical scope of a result, but they are not omitted
controller state or hidden code shortcuts.

## Physical evidence boundary

The original archived `src/physical_policy.py` adapter drove the board trials
with the same analytical score primitives, calibrated MLP execution, actual
transport, temperature health checks, and measured PoE power where available.
Those trials use a finite batch protocol and one deployed MLP service. They do
not validate raw-function container identity, arbitrary multi-hop routes, or
the complete request-level trace state machine on hardware.

Accordingly, the manuscript may use these trials as calibration and bounded
physical replay evidence. It must not call them an end-to-end hardware
validation of every request-level mechanism. The canonical implementation and
all claims about raw-function caching, multi-hop switch coupling, unbiased
thinning, and trace-scale state evolution come from the source-locked strict
campaigns above.

The additional physical protocol uses the same 14-board cohort and resource
constraints for every compared policy. Its 140-trial design consists of ten
randomized matched blocks across nominal, energy-pressure, network-pressure,
and mixed-service conditions, including both network ablations. The CNN and
random forest are trained models and execute their complete inference graphs
alongside the MLP. All three services have 20 calibration repetitions for
each board and batch size 1, 4, and 16.

Original compute-first bypasses energy and request guards. The added
common-constraint compute-first uses the same guards as SCD and matched DPP.
The new application-coherent tenant grouping also differs from the original
physical generator's trigger-category groups. The original and new records
are therefore interpreted within their own protocols.

The added physical PSE boundary contains seven declared ports, with separate
total and idle-adjusted subtotals. Estimated task charges use common declared
coefficients for every service and policy. Those estimates are not individual
measurements of all 14 endpoints. A separate 60-trial experiment measures
actual Ethernet-path RTT, loss, sender rates, and switch interface counters.
RTT includes host and NIC contributions and is not an internal switch-buffer
residence measurement. Acquisition completion, source checks, and full raw-log
analyses are recorded in each campaign's `analysis/audit.json`.
