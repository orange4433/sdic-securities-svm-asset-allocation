# Validation

All cells in the offline replay executed for 139 monthly observations (2014-01 to 2025-07). Weights were finite and summed to either zero or one. The five-asset comparison loaded all five included input files, computed finite risk/return metrics and completed its plotting workflow. The full SVM Bayesian search and historical refitting were not rerun. This verifies cached-probability replay and the separate comparison, rather than new SVM training or exact paper-table agreement.

Preparation used Python 3.11 in an existing research environment. These checks are not a clean-environment, end-to-end replication of every result.

The English filename update preserved SHA-256 hashes for all five renamed datasets. Both training entry points successfully loaded the renamed workbook, and all four renamed CSV archives were readable. Executable code changes are limited to filenames and file-save status messages; the model calculations were preserved. Full training was not rerun for this filename update.
