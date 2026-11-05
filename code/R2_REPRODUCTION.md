# Reproducing the second revision

The archived numerical record has two source versions. The six original
request-level matrices use `frozen_r1_src`. The added 200-run factorial uses
its own execution snapshot. The first 15 repeated nominal records reproduce
the original SCD, matched-DPP, and compute-first metrics. They remain five
seeded replicates per policy, not extra independent samples.

Run commands from this `code` directory after extracting all evidence assets.
Install the versions in `requirements.txt` in a Python 3.14 environment.
Simulation and statistical reproduction require no laboratory credentials.

```text
python -m pip install -r requirements.txt
python reproduce.py tests
python reproduce.py statistics
python reproduce.py figures
python reproduce.py traces --workers 5
python reproduce.py conformance
python reproduce.py baseline --workers 5
python reproduce.py additions --workers 5
```

`statistics` reconstructs the statistical summaries from the archived result
records and raw hardware logs, including all 280 original scheduler trials
and all 160 host/container activations. It does not rerun hardware. All
derived outputs use a separate directory. `figures` rebuilds all nine retained
main and supplementary figures with the frozen plotting source, compares
all sixteen plot-input tables, and verifies identical PDF renderings in the
registered software and font environment. PDF creation timestamps are excluded
from the pixel comparison. The original figure environment uses Times New
Roman and Arial. Exact visual
comparison requires those fonts as well as the pinned Python libraries.
Numerical statistical reanalysis does not depend on these fonts.
For a second complete reanalysis, choose another fresh path with `--output`. `baseline` and
`additions` execute all registered trace-scale tasks with their respective
source versions. `all` combines these operations. Fresh simulations are
written below `results/reproduced_r2`, with source packages copied into that
directory. Python's safe-path option gives the selected snapshot import
priority. It does not resume a changed source version into a submitted run.

The packaged raw source, processed workload, fixed seeds, and all 81 trace
manifests are sufficient to regenerate the omitted deterministic request
arrays. The `traces` command rebuilds all 81 traces. A separate verifier
compares every regenerated array byte hash and all scientific metadata with
the immutable archived manifests, ignoring only the input's local path.
The simulation commands run this verification before using reconstructed
traces. Allocate space for these arrays, the extracted source, and fresh
simulation outputs. The original run used a 12-logical-core Windows host
with about 80 GB RAM. Hardware measurements were scheduled separately from
simulation and bulk artifact transfers.

For the complete numerical validation used for this release, choose a fresh
directory outside the extracted artifact and run:

```text
python verify_release.py --output ../../validation-run --report ../../validation.json
```

This verifies the original Azure archive hash, reconstructs all original and
new statistics from raw evidence, compares every generated table, rebuilds
all nine figures, runs the full test suite, reconstructs and verifies all 81
request arrays, and repeats the full CNN and forest training. The report
also includes executable conformance checks against the rebuilt traces and
each version's own frozen simulation source, covering every original and new
campaign protocol. These checks execute after reconstruction rather than
relying only on the historical conformance report. The final report
records each comparison and its hashes. It does not repeat live laboratory
measurements or substitute a partial simulation for the archived matrices.

## Upstream cohort and predictor reconstruction

The workload cohort is the 9,000 Azure functions with the highest invocation
totals over the 14-day public source interval. This offline selection is
fixed before thinning. It is not a claim that the public source contains
only 9,000 functions. The upstream script and selected function identities
are included.

```text
python reproduce.py upstream
```

This optional command extracts the included public source archive, rebuilds
the selected cohort in a fresh directory, and augments its duration
quantiles. The included `results/processed/workload.npz` preserves the
upstream processed input, while the canonical simulator reads
`results/strict/processed/workload_strict_v2.npz`.

The historical class aggregate is needed to reconstruct the original
predictor and physical MLP input. Its trigger-category tenant labels are
not the tenant identity of the final request-level simulator or the new
physical campaign. The predictor sums demand over those categories. To
rebuild that upstream chain into fresh output directories:

```text
python frozen_input_src/enhanced_workload.py --workload results/processed/workload.npz --out results/reproduced_r2/forecast_input.npz
python -m src.forecast_online --aggregate results/reproduced_r2/forecast_input.npz --out results/reproduced_r2/forecast_models --epochs 48 --seed 20260905
python -m src.export_portable_model --checkpoint results/reproduced_r2/forecast_models/checkpoint_before_02880.pt --aggregate results/reproduced_r2/forecast_input.npz --out results/reproduced_r2/portable_mlp
```

The new physical aggregate in `results/r2/workload_application_16x4.npz`
instead uses the same stable application-identity mapping as `strict_trace`.
Its metadata report exact conservation of the total and every minute's count.
`src.build_r2_physical_workload` provides that transformation.

## Added trained services

```text
python -m src.train_r2_workloads --out results/reproduced_r2/trained_services
python verify_trained_services.py --out results/reproduced_r2/training_verification
```

This runs the full fixed 80-epoch CNN training and fits the 64-tree forest.
The 1,347/450 split is a new stratified split of scikit-learn's 1,797-example
UCI digits subset. It is not the original UCI training/test benchmark.
The artifact includes the seed, split indices, training losses, data hash,
exported model parameters, and held-out inputs and targets. The original
controller export predictions match the framework on all 450 held-out cases.
All three models execute their actual inference computations on the boards.
The verification command repeats the complete training from a source snapshot
and compares the exported models, all 80 epoch losses, the data split,
evaluation records, and every held-out array against the deployed artifacts.

## Hardware reproduction and evidence

The archive retains 280 original scheduler trials, 160 host/container
activations, 2,520 new service calibration groups, 140 new matched-constraint
scheduler trials, and 60 switch-path trials in separate directories.
The formal designs use ten randomized matched blocks. The new calibration
uses 20 within-session repetitions per board, service, and batch size.

The new scheduler campaign fixes the same energy and request guards for
every method. Its four cells and extra network ablations are all retained.
Four blocks interrupted by power-management timeouts were repeated in full.
Five interruptions occurred across those four blocks.
Completed preceding blocks were preserved byte for byte. After the second
interruption, two staggered read-only observation streams replaced the single
stream. After the third interruption, short-timeout reconnection was added.
Both streams and every failed query are retained. An observed-support gap
exceeding 2.5 seconds still stops the trial.
After the fourth interruption, observation was moved to its own process on
the same controller. The shared monotonic clock preserves all timing units.
After the fifth interruption, one ICMP echo per second from the controller
and one from port 1 were added to diagnose management reachability independently.
These diagnostic packets never enter the scheduler or the energy estimate.
The sampling change applies equally to every policy in each subsequent block.
Every final energy interval still requires complete observed coverage.
Interrupted attempts, recovery rules, and acquisition-source snapshots are
retained separately. No completed policy result was selected according to
whether it favored SCD.

The separate interruption sensitivity includes 35 additional complete trials
from 11 conditions for which every compared policy completed. Repeats are
averaged within their original block, retaining ten statistical blocks.
All ten success-contrast estimates change by less than 0.087 percentage point.
This descriptive analysis excludes power and does not alter the 140-trial
primary analysis or its 40-test correction. Its selection decisions and
all underlying condition records are in `results/r2/interruption_sensitivity`.

New physical results are stored at `results/r2/matched_physical_v4`. New
switch results are stored at `results/r2/switch_contention_v3`. These are
canonical artifact paths. Their raw manifests and recovery metadata identify
the acquisition directories and immutable execution snapshots.

The switch acquisition also experienced one connection-establishment failure
before the second condition of its sixth block. The first five complete
blocks were preserved byte for byte. All six conditions in the sixth block
were repeated in their original order. The original acquisition, including
the excluded completed condition, remains in `results/r2/switch_interruption_20261105`.
Its registered continuation adds at most three connection attempts with
20-second prompt waits before traffic or counter collection. It changes no
packet-generation code, measurement duration, load, or in-window counter
command, and performs no within-trial retries or imputation.

The new power columns are the observed subtotal of seven declared PSE ports,
with a separate idle-adjusted subtotal. They are not total fleet supply
energy. Predicted task charges use the declared common coefficient model and
are reported separately. Returned-delay metrics condition on parity-valid
completions, while success and rejection retain all offered inferences.
The switch experiment measures end-to-end RTT and loss, not internal buffer
residence time. Its original packet records and interface replies accompany
the aggregate tables.

Repeating live acquisition requires a private inventory containing each
board's port identifier, SSH address, username, and pinned host-key digest.
Supply the key path in `EDGE_SSH_KEY` and switch connection parameters in
`EDGE_SWITCH_HOST` and `EDGE_SWITCH_USERNAME`. Switch authentication is
entered through a hidden prompt and is never written into a protocol.
The run modules expose their required paths with `--help`. An inventory
template documents the schema without publishing the original deployment.

## Provenance and public redaction

The public package removes private endpoint addresses, usernames, host-key
fingerprints, and local-user paths. All scientific JSON keys, array shapes,
numeric values, boolean values, and null entries are preserved. Two historical
transport snapshots replace literal switch host/user defaults with environment
lookups. Mathematical, scheduling, and inference code remains byte-identical.
`PUBLIC_PROVENANCE.json` records original and public hashes and the
transformation for every file. The public integrity manifest hashes the
distributed files themselves.

Statistical reanalysis checks the public records and is compared numerically
with the original summaries. Original acquisition hashes remain provenance
identifiers. They are not misrepresented as hashes of redacted files. The
review correspondence, internal assessment, credentials, and private
inventory are not part of the public artifact.

The frozen container analyzer directly checks the collected files' byte counts
and hashes. The reproduction wrapper first verifies both original and public
hashes and byte counts for redacted runtime-information records. It then
creates an isolated temporary view with the corresponding validation-metadata
entries. Distributed acquisition manifests, measurements, numerical fields,
and frozen analysis code remain unchanged. All collected daemon logs are
included, and every metadata correspondence is recorded in the verification
report.
