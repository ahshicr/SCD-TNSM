# Conditional-denominator correction in v2.0.2

The second independent audit identified a no-admission boundary case in
serialized simulation metrics. Frozen run records divide conditional counts
by `max(admitted_requests, 1)`. At zero admission this encodes strict and
ceiling miss ratios as 1 and six other conditional ratios as 0. None of these
eight values is a defined conditional probability when the denominator is zero.

The reporting layer now marks these eight ratios as unavailable in
`analysis/reported_runs.csv` for each of the seven simulation campaigns.
There are 55 affected run records: 20 in the retained matrix and 35 in the
added factorial. Raw JSON, raw CSV, execution snapshots, admission decisions,
and measured data are preserved byte for byte.

In the retained matrix, 16 descriptive rows and 28 paired estimates now
correctly have no numerical estimate. Sixteen formerly significant contrasts
involved an undefined conditional outcome and no longer support an inference.
They remain in the registered Holm family with conservative `p=1`. Holm is
recomputed for that full family. Significance decisions for contrasts with
defined outcomes are unchanged. The added factorial's seven published metrics,
physical results, primary numerical claims, and plotted values are unchanged.
An incomplete finite replicate set now stops analysis instead of producing
a selected complete-case estimate.

The statistical unit remains a complete trace seed or a randomized physical
block. Simulation quantiles use linear interpolation over individual requests.
Physical response quantiles use the left inverse of the request-weighted
empirical CDF. Controller timing quantiles instead concern batch-level events.
Peak AoI pools subsequent informative deliveries in the evaluated release
cohort, including drain completions. These protocols are not pooled.

This release includes the v2.0.1 multiplicity correction. Extract the seven
immutable v2.0.0 evidence archives, then the v2.0.2 correction ZIP. The v2.0.1
overlay is not required. See `EVIDENCE_ASSETS.json` for ordered assets and
`code/results/r2/conditional_denominator_correction_v202.json` for the exact
affected comparisons. The previous release remains available and unchanged.
