# SDIC Securities: SVM-Based Asset Allocation and Fixed Income Strategy Research

Research conducted during the Fixed Income Strategy Research internship at **SDIC Securities**, centered on **support-vector-machine screening for dynamic asset allocation**, with a second study comparing equity-index futures, convertible bonds and bond funds.

## Research reports

- **[SVM dynamic asset allocation: full report](paper/svm_dynamic_asset_allocation.pdf)** ([DOCX](paper/svm_dynamic_asset_allocation.docx)). This is the main project.
- [Equity-index futures and bond-fund comparison: full report](paper/futures_and_bond_funds_comparison.pdf) ([DOCX](paper/futures_and_bond_funds_comparison.docx)).

## Main SVM study

The research combines macroeconomic regimes, calibrated SVM probabilities, time-series cross-validation and inverse-volatility asset weights. The historical training workflow includes Bayesian hyperparameter optimization and periodic retraining.

| Path | Purpose |
| --- | --- |
| `svm/train_svm.py` | Corrected full training and backtest workflow |
| `svm/original_training.ipynb` | Original research notebook |
| `svm/asset_prices.xlsx` | Historical asset-price inputs |
| `svm/asset_performance_by_economic_regime.xlsx` | Macroeconomic and regime inputs |
| `svm/best_svm_params.json` | Archived model parameters |
| `replay/all_weather_svm.ipynb` | Compact offline strategy replay |
| `replay/*.csv` | Monthly returns and previously estimated SVM probabilities |
| `results/svm_predictions.csv` | Archived SVM prediction results |
| `results/svm_weights.csv` | Archived SVM portfolio weights |
| `results/strategy_backtest_results.csv` | Archived strategy backtest results |
| `results/strategy_performance_comparison.csv` | Archived strategy performance comparison |

## Running

Use Python 3.11 and install `requirements.txt`.

1. For the shortest executable example, open `replay/all_weather_svm.ipynb` from `replay/` and run its cells. It uses the supplied monthly returns and probabilities to compare the SVM-screened portfolio with the All Weather benchmark.
2. To retrain, run `python train_svm.py` from `svm/`, or run `original_training.ipynb` there. Bayesian optimization and repeated time-series fits can be expensive.
3. Run `python compare_assets.py` from `comparison/` for the separate five-asset comparison. Its input directory is resolved relative to the script.

## Reproducibility and limits

The monthly replay and the five-asset comparison were executed against the included data. **The replay consumes saved probabilities; it does not retrain an SVM.** Full Bayesian optimization and historical retraining were not rerun during archive preparation, so exact paper-table agreement is not claimed for a new fit.

The replay covers January 2014 through July 2025; the final month is partial through July 7. The example omits transaction costs. Original financial column names and runtime display text are retained for compatibility; comments and README are in English.

Only research reports, core code and selected inputs/results are included. HR documents, personal certificates, resumes, account credentials and unrelated reference reports are excluded.
