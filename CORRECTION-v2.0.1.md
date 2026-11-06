# Statistical consistency correction

Version 2.0.1 retains every original simulation and physical measurement.
It corrects the Holm family used by `summarize_strict_campaign.py`.
Previously, an identically zero paired difference produced an undefined
p value and was omitted from the number of tested contrasts. Undefined
no-service outcomes could also remove a registered comparison.

The corrected analysis keeps both kinds of stochastic comparison in their
registered family with conservative p=1. Its effect estimate and confidence
interval remain undefined where there is no outcome. Single-reconstruction
controls have no inferential test. Near-constant, nonzero differences use their
actual sample variation rather than a floating-point closeness shortcut.

All six retained matrices were reprocessed from the 1,128 original run
records. Their 4,788 descriptive rows and all existing effect estimates and
confidence intervals are unchanged. The complete contrast registry now has
4,935 rows, including 42 previously omitted undefined-outcome rows.
Twenty-six secondary comparisons no longer meet the adjusted 0.05 threshold.
The exact affected comparisons and old/new adjusted probabilities are in
`code/results/r2/statistical_correction_v201.json`. The 200-run factorial,
physical analyses, plotted data and table values are unchanged.

The frozen R1 simulator and its original source snapshot remain immutable.
`reproduce.py` uses a separate, hashed `execution_statistics` snapshot for
the corrected summarizer. Four regression tests cover retained zero
comparisons, undefined outcomes, near-constant effects and descriptive-only
single-reconstruction controls.

The complete evidence comprises the seven unchanged v2.0.0 evidence ZIPs
followed by the small v2.0.1 correction ZIP. `EVIDENCE_ASSETS.json` gives the
ordered URLs and hashes. Extract the correction last. `MANIFEST.sha256`
describes the resulting complete v2.0.1 file set.
