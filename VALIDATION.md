# Validation

All cells in the offline replay executed for 139 monthly observations (2014-01 to 2025-07). Weights were finite and summed to either zero or one. The five-asset comparison loaded all five included input files, computed finite risk/return metrics and completed its plotting workflow. The full SVM Bayesian search and historical refitting were not rerun. This verifies cached-probability replay and the separate comparison, rather than new SVM training or exact paper-table agreement.

Preparation used Python 3.11 in an existing research environment. These checks are not a clean-environment, end-to-end replication of every result.
