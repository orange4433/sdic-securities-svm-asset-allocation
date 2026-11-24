# Reproduction of the original research reports

Checked on 2026-11-24 against the original reports in `paper/`. This document records a new computational check; it is not an additional original research report. The original papers and archived result files have not been edited to make the results agree.

## SVM dynamic asset allocation

**Status: partial numerical reproduction.** The original paper uses 12 assets, including the S&P 500 and US bonds. The previously archived `train_svm.py` and `original_training.ipynb` implement a domestic 10-asset variant and cannot by themselves reproduce the full paper.

The paper version is now included as `svm/paper_reproduction.ipynb`, sourced from the original `SVM_USA.ipynb`. Its four input workbooks are included in `svm/`. Comments/documentation and input paths were translated into English. Two plotting-only errors were corrected: an undefined weight-table name and a fixed subplot grid that held only ten assets. Features, labels, training windows, random seeds, probability thresholds, calibration, optimization budgets and portfolio calculations were preserved.

The full workflow was executed from raw workbooks in a fresh namespace: all 12 assets, 139 prediction months per asset, 144 Bayesian searches and 2,220 candidate evaluations. The original initial/rolling search budgets of 20/15 were retained. The retrained probability table equals `replay/svm_probabilities.csv` after mapping asset names, with maximum absolute difference **0.0** in the tested environment. This is a fresh training check in addition to replaying cached predictions.

| Strategy / metric | Original paper | Fresh 12-asset run | Match at paper precision |
| --- | --- | --- | --- |
| SVM annual return | 8.05% | 8.05% | Yes |
| SVM annual volatility | 3.59% | 3.59% | Yes |
| SVM Sharpe ratio | 2.2410 | 2.2410 | Yes |
| SVM maximum drawdown | -2.86% | -2.85% | No |
| All Weather annual return | 7.00% | 7.00% | Yes |
| All Weather annual volatility | 3.29% | 3.29% | Yes |
| All Weather Sharpe ratio | 2.1258 | 2.1258 | Yes |
| All Weather maximum drawdown | -3.78% | -2.86% | No |

### Differences between archived versions

- **Portfolio table:** both reported maximum drawdowns match a separate archived US-asset backtest ending in June 2025 (-2.860189% SVM, -3.776520% All Weather). That older file's annual returns/volatilities/Sharpe ratios differ from the paper. The fresh run and public replay end in July 2025 and match the other six cells. Thus no single checked return series reproduces all eight cells simultaneously. The older saved weights also differ from weights reconstructed using the saved probabilities and current return inputs; the discrepancy cannot be described as only an end-date change.
- **Classification table:** all 30 reported metric cells for the ten domestic assets match the archived domestic 10-asset results. The six cells for the two US assets match the fresh 12-asset run. In that single fresh run, the ten domestic rows differ. This is evidence that the paper's table combines model versions. The checked metrics are accuracy, F1 and predicted-up proportion at the paper's one-decimal percentage precision.
- **Other benchmarks:** the paper's 60/40 and macro-factor risk-parity portfolio rows have not been reproduced in this check.

The model backtest covers January 2014 through July 2025. July contains input observations only through July 7; it is not a completed calendar month. Transaction costs are omitted. Exact matching here does not establish out-of-sample validity, eliminate possible methodological limitations, or guarantee the same fit under every dependency version. Figures were generated during execution, but no pixel-level comparison with the report was performed.

## Equity-index futures and bond-fund comparison

The CSI 500 futures column reproduces **all 36 tested cells**: four metrics for each year 2018-2025 and four full-period metrics. The full-period values round to annual return **9.05%**, annual volatility **23.75%**, Sharpe **0.38**, and maximum drawdown **-35.38%**.

This check uses the archived futures NAV series and the actual functions in `comparison/compare_assets.py`. Its observations run from January 2, 2018 through August 11, 2025. It validates calculations from that NAV; the original construction/rolling of the futures NAV was not reconstructed from contract-level data.

The report's bond-fund and convertible-bond **median/top-third** statistics require the underlying cross-sectional security universe. The included five time series do not supply that universe, so those aggregate cells remain unverified. The five-series script runs successfully, but that alone is not full reproduction of this report.

## Evidence

- [paper_metric_comparison.csv](reproduction/paper_metric_comparison.csv): paper values versus recalculated values, including unmatched and unverified rows. Percentages are compared at the precision printed in the paper.
- [verification.json](reproduction/verification.json): full execution counts, tested package versions, input SHA-256 hashes and archive-run verification.
- [VALIDATION.md](VALIDATION.md): execution checks and runtime fixes.

No parameters or source result values were adjusted to force agreement with the papers.
