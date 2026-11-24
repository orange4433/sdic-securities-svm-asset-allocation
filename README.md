# SDIC Securities: SVM-Based Asset Allocation and Fixed Income Strategy Research

Research conducted during the Fixed Income Strategy Research internship at **SDIC Securities**, centered on **support-vector-machine screening for dynamic asset allocation**, with a second study comparing equity-index futures, convertible bonds and bond funds.

## Research reports

- **[SVM dynamic asset allocation: full report](paper/svm_dynamic_asset_allocation.pdf)** ([DOCX](paper/svm_dynamic_asset_allocation.docx)). This is the main project.
- [Equity-index futures and bond-fund comparison: full report](paper/futures_and_bond_funds_comparison.pdf) ([DOCX](paper/futures_and_bond_funds_comparison.docx)).

## Main SVM study

The research combines macroeconomic regimes, calibrated SVM probabilities, time-series cross-validation and inverse-volatility asset weights. The historical training workflow includes Bayesian hyperparameter optimization and periodic retraining.

| Path | Purpose |
| --- | --- |
| `svm/paper_reproduction.ipynb` | Full 12-asset paper workflow, including US equities and bonds |
| `svm/train_svm.py` | Full training and backtest for the domestic 10-asset variant |
| `svm/original_training.ipynb` | Domestic 10-asset research notebook |
| `svm/asset_prices.xlsx` | Historical asset-price inputs |
| `svm/asset_performance_by_economic_regime.xlsx` | Macroeconomic and regime inputs |
| `svm/sp500_prices.xlsx`, `svm/us_bond_etf.xls` | Original US-market inputs for the 12-asset workflow |
| `svm/best_svm_params.json` | Archived domestic model parameters |
| `replay/all_weather_svm.ipynb` | Compact offline strategy replay |
| `replay/*.csv` | Monthly returns and previously estimated SVM probabilities |
| `results/svm_predictions.csv` | Archived SVM prediction results |
| `results/svm_weights.csv` | Archived SVM portfolio weights |
| `results/strategy_backtest_results.csv` | Archived strategy backtest results |
| `results/strategy_performance_comparison.csv` | Archived strategy performance comparison |

## Running

Use Python 3.11 and install `requirements.txt`.

1. For the shortest executable example, open `replay/all_weather_svm.ipynb` from `replay/` and run its cells. It uses the supplied monthly returns and probabilities to compare the SVM-screened portfolio with the All Weather benchmark.
2. To reproduce the paper's main SVM results from raw inputs, run every cell in `svm/paper_reproduction.ipynb` with `svm/` as the working directory. All four required input workbooks are included. Use a working copy if you want to retain the archived parameter file: the training notebook writes `best_svm_params.json`. Bayesian optimization and repeated time-series fits can be expensive.
3. For the separate domestic 10-asset variant, run `python train_svm.py` from `svm/`, or run `original_training.ipynb` there.
4. Run `python compare_assets.py` from `comparison/` for the separate five-asset comparison. Its input directory is resolved relative to the script.

## Reproducibility and limits

**The paper is partially reproduced.** Full retraining of the 12-asset paper notebook completed all 144 optimization runs and 2,220 candidate evaluations, producing 139 monthly predictions per asset (January 2014 through July 2025). The freshly trained probabilities match the compact replay's saved probabilities exactly in the tested environment. The SVM portfolio reproduces the paper's **8.05% annual return, 3.59% annual volatility and 2.2410 Sharpe ratio**; the All Weather portfolio reproduces **7.00%, 3.29% and 2.1258**, respectively.

Two maximum-drawdown cells and the ten domestic rows of the paper's classification table come from different archived result versions. They do not match this single 12-asset run. The paper's 60/40 and macro-factor risk-parity benchmark rows remain unverified. See [REPRODUCTION.md](REPRODUCTION.md) for the numerical comparison and limits; the original reports and archived results have been retained unchanged.

Both domestic 10-asset training entry points also completed, each with 120 optimization runs and 1,850 candidate evaluations. The monthly replay and five-asset comparison completed. **The replay consumes saved probabilities; it does not retrain an SVM.** The second report's CSI 500 futures column matches in all 36 tested cells; its cross-sectional fund/bond aggregates are not reproduced by the included five-series comparison. See [VALIDATION.md](VALIDATION.md) for the environment and runtime fixes.

The replay covers January 2014 through July 2025; the final month is partial through July 7. The example omits transaction costs. Original financial column names and runtime display text are retained for compatibility; comments and README are in English.

Only research reports, core code and selected inputs/results are included. HR documents, personal certificates, resumes, account credentials and unrelated reference reports are excluded.
