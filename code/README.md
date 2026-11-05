# SCD: Switch-Coupled Scheduling for Energy-Bounded Heterogeneous Edge AI

This artifact supports the second revision of **Switch-Coupled Scheduling
for Energy-Bounded Heterogeneous Edge AI on Commodity Ethernet** by Wuzhao
Li and Weian Guo. It preserves the submitted primary results and adds
component, allocation, matched-constraint hardware, diverse-service, and
actual-switch evidence.

Start with [R2_REPRODUCTION.md](R2_REPRODUCTION.md) for executable commands,
[R2_EVIDENCE_MAP.md](R2_EVIDENCE_MAP.md) for paper-to-record mappings, and
[THIRD_PARTY_DATA.md](THIRD_PARTY_DATA.md) for source attribution.

The canonical simulator is request-level. Each retained invocation is one
discrete event with its original anonymized function identity. It directly
implements execution-start TTL checks, byte-limited LRU eviction, shared
per-hop FCFS resources, temperature-dependent service, device and tenant
energy accounts, exact window request screening, and receiver-order peak
AoI. Predicted and realized states are separate. Rejected requests remain
failures in offered success.

The source cohort consists of the 9,000 highest-invocation functions over
the 14-day Azure source interval. Native binomial thinning, fixed seeds,
control reconstructions, and immutable trace manifests are included.
State evolves from day 0. Reported metrics use days 10--13, except that
flooding outcomes use the final trace day alone. The deterministic-count
rounding control uses one fixed reconstruction per policy and no across-seed
interval. The other stochastic design cells use five trace seeds. The original
six trace-scale matrices total 1,128 runs. A separate 200-run factorial
brings the execution record to 1,328. Five repeated seeds per cell remain
five statistical replicates. The fifteen repeated primary records exactly
match the original primary metrics.

The original hardware evidence and the new 14-board campaign occupy
separate directories. The original adapter and common-constraint adapter
have explicitly different resource contracts. The new campaign gives every
policy the same joule buckets, request ceilings, temperature threshold, and
in-flight limit. It executes trained MLP, CNN, and random-forest inference.
Power subtotals cover only the declared measured ports. Actual-switch RTT
and loss are measured separately from internal modeled queue residence.

Full statistical families retain favorable, null, adverse, and no-service
outcomes. Stochastic simulation cells use five paired trace seeds. Hardware comparisons
use ten randomized matched blocks. Individual requests, power samples, and
calibration repetitions are not treated as independent deployments.

The release keeps mathematical and execution provenance separate from
privacy redaction. `PUBLIC_PROVENANCE.json` maps original and distributed
hashes. Public records preserve numeric evidence while removing deployment
identifiers. Historical literal switch connection defaults are converted
to environment lookups. No credentials or private inventory are distributed.

```text
python -m pip install -r requirements.txt
python reproduce.py tests
python reproduce.py statistics
```

Full simulation commands reconstruct omitted deterministic request arrays
and use isolated frozen source packages. They do not require access to the
authors' hardware. The deployment-specific acquisition scripts require an
operator-provided private inventory and explicit connection parameters.

The journal submission and public artifact are distinct deliverables.
Reviewer correspondence and the internal Chinese assessment are excluded
from the public package.
