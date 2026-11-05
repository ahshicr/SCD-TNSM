# SCD research artifact

**Switch-Coupled Scheduling for Energy-Bounded Heterogeneous Edge AI on
Commodity Ethernet**, Wuzhao Li and Weian Guo.

This repository contains the implementation and evidence for the second
revision. The original primary simulation is retained alongside the added
component, allocation, common-constraint hardware, and actual-switch studies.

## Get the complete evidence

Tagged release: [v2.0.0](https://github.com/ahshicr/SCD-TNSM/releases/tag/v2.0.0).

The source repository and the evidence archives are complementary. Download
the assets listed in `EVIDENCE_ASSETS.json` from the tagged release, verify
their SHA-256 digests, and extract every evidence ZIP into the repository
root. Each archive has distinct paths. Ordinary ZIP extraction is sufficient.
The source archive is also supplied for users who do not use Git.

`MANIFEST.sha256` identifies the distributed files. `PUBLIC_PROVENANCE.json`
records acquisition-origin and public-file hashes, including deployment
redaction. These two types of hashes serve different purposes. The complete
raw scientific measurements are included. Deterministic request arrays are
rebuilt from the included workload, seeds, and immutable trace manifests.

## Reproduce

Run from the `code` directory with Python 3.14 and the pinned dependencies.

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

The statistics command reprocesses archived results and raw measurements.
The figure command regenerates all nine retained figures and checks their
source tables and rendered content against the manuscript evidence.
The release's complete validation can also be repeated with
`python verify_release.py --output ../../validation-run --report ../../validation.json`,
using fresh output paths outside the extracted artifact.
The simulation commands execute the full registered trace matrices.
The laboratory is not required for these commands. Fresh output directories
are separate from the archived results. See
[the reproduction guide](code/R2_REPRODUCTION.md) for source selection,
upstream cohort reconstruction, model training, and acquisition instructions.

## Evidence inventory

| Study | Record |
|---|---|
| Original request-level simulations | 1,128 executions in six matrices |
| Added component and allocation factorial | 200 executions, four conditions and ten policies |
| Original physical scheduler | 280 trials, ten matched blocks |
| Original host/container deployment comparison | 160 activations |
| New three-service calibration | 2,520 groups and 17,640 individual inferences on 14 boards |
| New common-constraint physical scheduler | 140 trials, ten matched blocks |
| Actual H3C Ethernet-path experiment | 60 trials, ten matched blocks |

The 15 repeated primary factorial records reproduce the original nominal
results. They do not create extra independent replicates. Five simulation
seeds characterize variation conditional on the fixed source cohort and
modeled deployment. Hardware intervals use randomized matched blocks.
The deterministic-rounding control has one fixed reconstruction per policy.
The 30 request-flood simulations run the full trace but evaluate day 13.
The other simulation conditions evaluate days 10 through 13.

[The evidence map](code/R2_EVIDENCE_MAP.md) links tables and figures to their
records and analysis scripts. `VALIDATION.json` records the completed
numerical reanalysis of the public files. Positive, null, adverse, and
no-service outcomes are retained together. Model-based task charges and
measured PSE subtotals have separate columns and boundaries.

## Licenses and attribution

Author-generated source code is distributed under the MIT license in
`LICENSE`. Author-generated measurements, model parameters, figure source
data, and scientific documentation use CC BY 4.0 as specified in
`LICENSE-DATA.md`. Third-party datasets retain their own licenses and
attribution in [THIRD_PARTY_DATA.md](code/THIRD_PARTY_DATA.md).

Use `CITATION.cff` to cite this software and evidence release. A manuscript
DOI will be added if assigned. Reviewer correspondence, the unpublished
manuscript, the authors' internal assessment, and laboratory credentials
are not part of this repository.
