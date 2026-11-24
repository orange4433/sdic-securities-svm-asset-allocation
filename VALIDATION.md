# Validation

## Full training execution

Both training entry points completed successfully on 2026-11-24 using isolated copies of the repository inputs. The archived results and parameter files in the repository were not replaced by these validation runs.

| Entry point | Execution | Result |
| --- | --- | --- |
| `svm/train_svm.py` | Complete production script | All 10 assets; 139 monthly predictions per asset; 139 backtest months; output CSVs, parameters and comparison figure produced |
| `svm/original_training.ipynb` | Every code cell, sequentially in a fresh namespace | All 10 assets; 139 monthly predictions per asset; all post-training analysis cells completed |
| `replay/all_weather_svm.ipynb` | All cells | 139 monthly observations; finite replay returns |
| `comparison/compare_assets.py` | Complete script | Five input datasets loaded; comparison completed and two figures produced |

Each full training run retained the original optimization budgets: 20 initial evaluations and 15 per rolling re-optimization. All 120 optimization runs finished, totaling 1,850 candidate evaluations for each entry point. No assets or prediction months were removed to shorten verification. Prediction probabilities, portfolio weights and backtest returns were checked for finite values; probabilities lay in [0, 1].

The script took approximately 277 seconds and the notebook approximately 258 seconds in this environment, with partial overlap between the two runs. Timings depend on hardware and configuration. Matplotlib used the noninteractive Agg backend during verification.

## Runtime fixes

- Replaced the notebook's remaining `resample('M')` with `resample('ME')` for current pandas month-end compatibility.
- Replaced references to undefined `aw_asset_weights` in the notebook's weight analysis with the already constructed `total_weights_aw` table.
- The requirements specify pandas 2.2 or later and scikit-optimize 0.10.2 or later.

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
| Matplotlib | 3.11.1 |

This verifies complete execution with the included inputs in the stated environment. It does not establish exact historical paper-table reproduction or certify every statistical assumption. The final July 2025 observation is a partial month. The compact replay uses saved probabilities rather than new training.

The prior English filename migration preserved SHA-256 hashes for all five renamed data files. Those workbook/CSV inputs remain readable under their English names.
