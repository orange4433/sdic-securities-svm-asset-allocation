# Validation

## Full training execution

All three training entry points completed successfully on 2026-11-24 using isolated input copies. The archived results and parameter files in the repository were not replaced by these validation runs.

| Entry point | Execution | Result |
| --- | --- | --- |
| `svm/paper_reproduction.ipynb` | Every code cell, sequentially in a fresh namespace | All 12 assets, including US equities/bonds; 139 monthly predictions per asset; all post-training analysis cells completed |
| `svm/train_svm.py` | Complete production script | All 10 assets; 139 monthly predictions per asset; 139 backtest months; output CSVs, parameters and comparison figure produced |
| `svm/original_training.ipynb` | Every code cell, sequentially in a fresh namespace | All 10 assets; 139 monthly predictions per asset; all post-training analysis cells completed |
| `replay/all_weather_svm.ipynb` | All cells | 139 monthly observations; finite replay returns |
| `comparison/compare_assets.py` | Complete script | Five input datasets loaded; comparison completed and two figures produced |

Each full training run retained the original optimization budgets: 20 initial evaluations and 15 per rolling re-optimization. The 12-asset paper workflow completed 144 optimization runs and 2,220 candidate evaluations. Each domestic 10-asset workflow completed 120 runs and 1,850 evaluations. No assets or prediction months were removed to shorten verification. Prediction probabilities, portfolio weights and backtest returns were checked for finite values; probabilities lay in [0, 1].

The domestic script took approximately 277 seconds and its notebook approximately 258 seconds, with partial overlap between the two runs. The original 12-asset notebook, with the plotting fixes below, took approximately 336 seconds. Its English archive copy was separately executed in full after translating comments and renaming input paths; the run summary is in [reproduction/verification.json](reproduction/verification.json). Timings depend on hardware and configuration. Matplotlib used the noninteractive Agg backend during verification.

## Runtime fixes

- Replaced the notebook's remaining `resample('M')` with `resample('ME')` for current pandas month-end compatibility.
- Replaced references to undefined `aw_asset_weights` in the notebook's weight analysis with the already constructed `total_weights_aw` table.
- In the 12-asset paper notebook, expanded the contribution plot's fixed 5-by-2 grid to enough rows for all 12 assets. Its source is the original `SVM_USA.ipynb`; comments/documentation and input filenames were translated, while financial column names were retained. An AST comparison confirmed that the executable model remained unchanged apart from the listed plotting fixes and filename substitutions.
- The requirements specify pandas 2.2 or later and scikit-optimize 0.10.2 or later.
- Added xlrd to read the original US bond ETF `.xls` workbook. The two newly included US-market workbooks are byte-identical to their source files.

The model's training loops, optimization budgets, feature calculations and historical archived results were preserved.

## Tested environment

Python 3.11, using a workspace virtual environment that reused the existing scientific Python installation and installed the missing Excel/optimization dependencies separately:

| Package | Version |
| --- | --- |
| NumPy | 2.4.6 |
| pandas | 3.0.3 |
| SciPy | 1.17.1 |
| scikit-learn | 1.9.0 |
| scikit-optimize | 0.10.2 |
| openpyxl | 3.1.5 |
| xlrd | 2.0.2 |
| Matplotlib | 3.11.1 |

This verifies complete execution and the specific numerical matches documented in [REPRODUCTION.md](REPRODUCTION.md). It does not establish complete paper reproduction or certify every statistical assumption. The final July 2025 observation is a partial month. The compact replay uses saved probabilities rather than new training.

The prior English filename migration preserved SHA-256 hashes for all five renamed data files. Those workbook/CSV inputs remain readable under their English names.
