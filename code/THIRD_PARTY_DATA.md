# Third-party source data and attribution

The Azure Functions 2019 workload originates from the Microsoft Azure
public datasets project and the serverless workload characterization by
Shahrad and colleagues. The project distributes the source dataset under
Creative Commons Attribution 4.0. The included archive is the unmodified
public source. Derived cohorts, thinning manifests, and scheduler outputs
retain its source attribution.

Source: https://github.com/Azure/AzurePublicDataset

The included `data/azurefunctions_dataset2019.tar.xz` corresponds byte-for-byte
to the producer's [Functions 2019 release asset](https://github.com/Azure/AzurePublicDataset/releases/download/dataset-functions-2019/azurefunctions_dataset2019_azurefunctions-dataset2019.tar.xz).
The producer's SHA-256 and the archived workload protocol both identify
`aff8b3ca7240a41a109e4ee598e0a96e45fcb92e7b8395ac19cb3748cd260d89`
(142,968,140 bytes). The local filename is shorter than the release asset name.

The added image classifiers use the Optical Recognition of Handwritten
Digits dataset by E. Alpaydin and C. Kaynak (1998), DOI
https://doi.org/10.24432/C50P49, distributed by the UCI Machine Learning
Repository under Creative Commons Attribution 4.0.

The specific 1,797-example subset is loaded from scikit-learn's
`load_digits`. The artifact declares its own stratified train/test split.
It does not substitute that split for the original UCI benchmark split.

Loader documentation:
https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html

Third-party datasets retain their original licenses and attribution.
The repository's code license does not replace a dataset's license.
Author-generated experiment logs and code are identified separately in
the release license notices.
