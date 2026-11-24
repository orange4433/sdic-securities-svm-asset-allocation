# -*- coding: utf-8 -*-
"""
All-weather strategy with Bayesian-optimized SVM asset screening (compact output).
Author: Quantitative researcher
Date: 2025-08-25
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from skopt import gp_minimize
from skopt.space import Real, Categorical
from skopt.utils import use_named_args
import warnings
import json
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei']  # Display Chinese text.
plt.rcParams['axes.unicode_minus'] = False  # Display minus signs.

# ==================== 1. Read economic-cycle data ====================
sheet1_df = pd.read_excel('asset_performance_by_economic_regime.xlsx', sheet_name='Sheet1')
sheet2_df = pd.read_excel('asset_performance_by_economic_regime.xlsx', sheet_name='Sheet2')

# Remove rows containing the data-source marker.
sheet1_df = sheet1_df[~sheet1_df['指标名称'].astype(str).str.contains("数据来源", na=False)]
sheet2_df = sheet2_df[~sheet2_df['指标名称'].astype(str).str.contains("数据来源", na=False)]

# Convert dates.
sheet1_df['指标名称'] = pd.to_datetime(sheet1_df['指标名称'])
sheet2_df['指标名称'] = pd.to_datetime(sheet2_df['指标名称'])

# ==================== 2. Read asset prices ====================
# Read the complete dataset.
asset_prices_df = pd.read_excel('asset_prices.xlsx').dropna().drop(columns=['中证REITs', 'WTI原油'])
asset_prices_df['日期'] = pd.to_datetime(asset_prices_df['日期'])
asset_prices_df.set_index('日期', inplace=True)
asset_prices_df = asset_prices_df.sort_index()

# Add momentum filters to the original prices.
# Momentum filter for the Nanhua Commodity Index
asset_prices_df['南华商品指数_动量过滤'] = np.where(
    asset_prices_df['南华商品指数'].rolling(1).mean().shift(1) > 0, 1, -1
)
asset_prices_df['南华商品指数_动量过滤'] = asset_prices_df['南华商品指数'] * asset_prices_df['南华商品指数_动量过滤'].fillna(1)

# Momentum filter for Shanghai gold
asset_prices_df['沪金_动量过滤'] = np.where(
    asset_prices_df['沪金'].rolling(2).mean().shift(1) > 0, 1, 0
)
asset_prices_df['沪金_动量过滤'] = asset_prices_df['沪金'] * asset_prices_df['沪金_动量过滤'].fillna(1)

# Full SVM training dataset, starting in 2008
monthly_prices = asset_prices_df.resample('ME').last()  # Month-end prices
monthly_returns = monthly_prices.pct_change().dropna()

# All-weather dataset from late 2009, aligned with macroeconomic data
aw_asset_prices = asset_prices_df.loc['2009-12-31':]
aw_monthly_prices = aw_asset_prices.resample('ME').last()
aw_monthly_returns = aw_monthly_prices.pct_change().dropna()

# ==================== 3. Merge macro states for the all-weather strategy ====================
sheet1_df.rename(columns={'指标名称': '日期', '高于预期为1，低于预期为2': '经济预期情况'}, inplace=True)
sheet2_df.rename(columns={'指标名称': '日期', '高于预期为1，低于预期为2': '通胀预期情况'}, inplace=True)

aw_monthly_returns = aw_monthly_returns.reset_index()
merged_df = aw_monthly_returns.merge(sheet1_df[['日期', '经济预期情况']], on='日期', how='left')
merged_df = merged_df.merge(sheet2_df[['日期', '通胀预期情况']], on='日期', how='left')

# Fill macroeconomic states.
merged_df['经济预期情况'] = merged_df['经济预期情况'].ffill()
merged_df['通胀预期情况'] = merged_df['通胀预期情况'].ffill()
merged_df[['经济预期情况', '通胀预期情况']] = merged_df[['经济预期情况', '通胀预期情况']].fillna(1)
merged_df = merged_df.dropna()

# Ensure the date column contains datetime values.
merged_df['日期'] = pd.to_datetime(merged_df['日期'])
merged_df.set_index('日期', inplace=True)

# ==================== 5. Define the asset universe ====================
all_assets = [
    '中债国债', '中债金融债', '中债企业债', '南华商品指数_动量过滤',
    '中证转债', '红利收益', '上证50', '沪深300', '沪金_动量过滤', '中证1000'
]

# ==================== 6. Technical-indicator functions ====================
def calculate_volatility(returns, window=12):
    """Calculate annualized volatility."""
    return returns.rolling(window).std() * np.sqrt(12)

def calculate_rsi(prices, window=6):
    """Calculate the relative strength index (RSI)."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# ==================== 7. Construct SVM features and labels ====================
svm_features = {}
svm_labels = {}

for asset in all_assets:
    # Use the complete dataset starting in 2008.
    prices = monthly_prices[asset].dropna()
    rets = monthly_returns[asset].dropna()
    
    # Label: 1 if the next-period return is positive.
    y = (rets.shift(-1) > 0).astype(int)
    
    # Construct features.
    feat_df = pd.DataFrame(index=prices.index)
    
    # Basic features, lagged one period
    feat_df['ret_1m'] = rets.shift(1)                    # Previous month's return
    feat_df['ret_3m'] = rets.shift(1).rolling(3).mean()  # Mean return over the previous 3 months
    feat_df['ret_6m'] = rets.shift(1).rolling(6).mean()
    feat_df['ret_12m'] = rets.shift(1).rolling(12).mean()
    
    # Technical indicators, lagged one period
    feat_df['volatility_12m'] = calculate_volatility(rets, 12).shift(1)  # Annualized volatility
    feat_df['rsi_6m'] = calculate_rsi(prices, 6).shift(1)               # Six-month RSI
    
    # Drop initial NaN values.
    feat_df = feat_df.dropna()
    
    # Align labels.
    y = y.reindex(feat_df.index)
    
    svm_features[asset] = feat_df
    svm_labels[asset] = y

# ==================== 8. Trading-oriented scoring function ====================
def custom_scorer(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    
    down_mask = (y_true == 0)
    acc_down = (y_pred[down_mask] == 0).mean() if down_mask.sum() > 0 else 0
    
    score = 0.4 * acc + 0.3 * f1 + 0.3 * auc
    if acc_down < 0.3:
        score *= (acc_down / 0.3)
    return score

# ==================== 9. Rolling training and prediction without future leakage ====================
# Initialize prediction storage.
svm_predictions = pd.DataFrame()
asset_pred_record = {asset: [] for asset in all_assets}
best_params_dict = {asset: None for asset in all_assets}

# Set rolling-training parameters.
INITIAL_TRAIN_START = '2008-01-01'  # Use data starting in 2008.
INITIAL_TRAIN_END = '2013-12-31'    # End of the initial training period
FIRST_PRED_DATE = '2014-01-01'      # First prediction date
RETRAIN_FREQ = 12                   # Re-optimize hyperparameters every 12 months.

# Train and predict separately for each asset.
for asset in all_assets:
    print(f"\n🔍 滚动训练预测中: {asset}")
    
    # Get features and labels for the asset.
    X = svm_features[asset]
    y = svm_labels[asset]
    
    # Align date indices.
    all_dates = X.index.sort_values()
    
    # Store predictions for this asset.
    asset_preds = pd.Series(index=all_dates, dtype=float)
    
    # Identify the first prediction date, January 2014.
    first_pred_date = pd.to_datetime(FIRST_PRED_DATE)
    
    # Initial training period, 2008-2013
    initial_train_dates = all_dates[(all_dates >= INITIAL_TRAIN_START) & (all_dates < first_pred_date)]
    X_initial = X.loc[initial_train_dates]
    y_initial = y.loc[initial_train_dates]
    
    # Drop NaN labels.
    valid_mask = y_initial.notna()
    X_initial = X_initial[valid_mask]
    y_initial = y_initial[valid_mask]
    
    # Skip the asset if the initial training sample is too small.
    if len(X_initial) < 30:
        print(f"⚠️  初始样本不足，跳过 {asset}")
        continue
    
    # Perform Bayesian optimization on the initial training period.
    print(f"  初始贝叶斯优化...")
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Search space
    search_space = [
        Real(0.1, 100, prior='log-uniform', name='C'),
        Real(0.001, 1.0, prior='log-uniform', name='gamma'),
        Categorical(['None', 'balanced', '1:2', '1:3', '1:4'], name='class_weight_str')
    ]
    
    @use_named_args(search_space)
    def objective(**params):
        # Construct class weights.
        if params['class_weight_str'] == '1:2':
            class_weight = {0: 1, 1: 2}
        elif params['class_weight_str'] == '1:3':
            class_weight = {0: 1, 1: 3}
        elif params['class_weight_str'] == '1:4':
            class_weight = {0: 1, 1: 4}
        elif params['class_weight_str'] == 'balanced':
            class_weight = 'balanced'
        else:
            class_weight = None
            
        model = SVC(kernel='rbf', probability=True, C=params['C'], gamma=params['gamma'], class_weight=class_weight)
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
        
        scores = []
        for train_idx, val_idx in tscv.split(X_initial):
            X_train, X_val = X_initial.iloc[train_idx], X_initial.iloc[val_idx]
            y_train, y_val = y_initial.iloc[train_idx], y_initial.iloc[val_idx]
            
            # Drop NaN labels.
            train_mask = y_train.notna()
            val_mask = y_val.notna()
            
            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_val = X_val[val_mask]
            y_val = y_val[val_mask]
            
            if len(X_train) == 0 or len(X_val) == 0:
                continue
                
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            calibrated_model.fit(X_train_scaled, y_train)
            y_proba = calibrated_model.predict_proba(X_val_scaled)[:, 1]
            score = custom_scorer(y_val, y_proba)
            scores.append(score)
        
        return -np.mean(scores) if scores else 0
    
    # Run optimization.
    result = gp_minimize(
        func=objective,
        dimensions=search_space,
        n_calls=20,
        random_state=42,
        verbose=False
    )
    
    # Get the best parameters.
    best_params = dict(zip(['C', 'gamma', 'class_weight_str'], result.x))
    best_params_dict[asset] = best_params
    
    # Construct class weights.
    if best_params['class_weight_str'] == '1:2':
        class_weight = {0: 1, 1: 2}
    elif best_params['class_weight_str'] == '1:3':
        class_weight = {0: 1, 1: 3}
    elif best_params['class_weight_str'] == '1:4':
        class_weight = {0: 1, 1: 4}
    elif best_params['class_weight_str'] == 'balanced':
        class_weight = 'balanced'
    else:
        class_weight = None
        
    print(f"✅ 初始最优参数: C={best_params['C']:.3f}, gamma={best_params['gamma']:.3f}, class_weight={class_weight}")
    
    # Train with the initial parameters.
    model = SVC(kernel='rbf', probability=True, C=best_params['C'], gamma=best_params['gamma'], class_weight=class_weight)
    calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
    
    scaler = StandardScaler()
    X_initial_scaled = scaler.fit_transform(X_initial)
    calibrated_model.fit(X_initial_scaled, y_initial)
    
    # Begin rolling prediction in January 2014.
    last_retrain_date = initial_train_dates[-1]
    
    for current_date in all_dates[all_dates >= first_pred_date]:
        # Check whether re-optimization is due.
        months_since_retrain = (current_date - last_retrain_date).days // 30
        retrain_needed = months_since_retrain >= RETRAIN_FREQ
        
        # Get training data through the previous month.
        train_dates = all_dates[all_dates < current_date]
        print(train_dates)
        X_train = X.loc[train_dates]
        y_train = y.loc[train_dates]
        
        # Drop NaN labels.
        train_mask = y_train.notna()
        X_train = X_train[train_mask]
        y_train = y_train[train_mask]
        
        # Skip if there are too few observations.
        if len(X_train) < 30:
            continue
        
        # If re-optimization is required
        if retrain_needed:
            print(f"  🔄 重新优化超参数 (日期: {current_date.date()})")
            
            # Re-optimize on the current training set.
            @use_named_args(search_space)
            def current_objective(**params):
                # Construct class weights.
                if params['class_weight_str'] == '1:2':
                    class_weight = {0: 1, 1: 2}
                elif params['class_weight_str'] == '1:3':
                    class_weight = {0: 1, 1: 3}
                elif params['class_weight_str'] == '1:4':
                    class_weight = {0: 1, 1: 4}
                elif params['class_weight_str'] == 'balanced':
                    class_weight = 'balanced'
                else:
                    class_weight = None
                    
                model = SVC(kernel='rbf', probability=True, C=params['C'], gamma=params['gamma'], class_weight=class_weight)
                calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=3)
                
                scores = []
                for train_idx, val_idx in tscv.split(X_train):
                    X_train_fold, X_val_fold = X_train.iloc[train_idx], X_train.iloc[val_idx]
                    y_train_fold, y_val_fold = y_train.iloc[train_idx], y_train.iloc[val_idx]
                    
                    # Drop NaN labels.
                    train_mask = y_train_fold.notna()
                    val_mask = y_val_fold.notna()
                    
                    X_train_fold = X_train_fold[train_mask]
                    y_train_fold = y_train_fold[train_mask]
                    X_val_fold = X_val_fold[val_mask]
                    y_val_fold = y_val_fold[val_mask]
                    
                    if len(X_train_fold) == 0 or len(X_val_fold) == 0:
                        continue
                        
                    scaler_fold = StandardScaler()
                    X_train_scaled_fold = scaler_fold.fit_transform(X_train_fold)
                    X_val_scaled_fold = scaler_fold.transform(X_val_fold)
                    
                    calibrated_model.fit(X_train_scaled_fold, y_train_fold)
                    y_proba = calibrated_model.predict_proba(X_val_scaled_fold)[:, 1]
                    score = custom_scorer(y_val_fold, y_proba)
                    scores.append(score)
                
                return -np.mean(scores) if scores else 0
            
            # Run optimization.
            result = gp_minimize(
                func=current_objective,
                dimensions=search_space,
                n_calls=15,  # Reduce the number of calls for speed.
                random_state=42,
                verbose=False
            )
            
            # Update the best parameters.
            best_params = dict(zip(['C', 'gamma', 'class_weight_str'], result.x))
            best_params_dict[asset] = best_params
            
            # Construct class weights.
            if best_params['class_weight_str'] == '1:2':
                class_weight = {0: 1, 1: 2}
            elif best_params['class_weight_str'] == '1:3':
                class_weight = {0: 1, 1: 3}
            elif best_params['class_weight_str'] == '1:4':
                class_weight = {0: 1, 1: 4}
            elif best_params['class_weight_str'] == 'balanced':
                class_weight = 'balanced'
            else:
                class_weight = None
                
            print(f"  ✅ 新最优参数: C={best_params['C']:.3f}, gamma={best_params['gamma']:.3f}, class_weight={class_weight}")
            
            last_retrain_date = current_date
        
        # Train with the current best parameters.
        best_params = best_params_dict[asset]
        if best_params['class_weight_str'] == '1:2':
            class_weight = {0: 1, 1: 2}
        elif best_params['class_weight_str'] == '1:3':
            class_weight = {0: 1, 1: 3}
        elif best_params['class_weight_str'] == '1:4':
            class_weight = {0: 1, 1: 4}
        elif best_params['class_weight_str'] == 'balanced':
            class_weight = 'balanced'
        else:
            class_weight = None
            
        model = SVC(
            kernel='rbf', 
            probability=True, 
            C=best_params['C'], 
            gamma=best_params['gamma'], 
            class_weight=class_weight
        )
        calibrated_model = CalibratedClassifierCV(model, method='isotonic', cv=5)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        calibrated_model.fit(X_train_scaled, y_train)
        
        # Predict at the current date, targeting the next month.
        X_current = X.loc[[current_date]].values.reshape(1, -1)
        X_current_scaled = scaler.transform(X_current)
        pred_proba = calibrated_model.predict_proba(X_current_scaled)[0, 1]
        
        # Store the prediction.
        asset_preds[current_date] = pred_proba
        svm_predictions.loc[current_date, asset] = pred_proba
        
        # Record the prediction.
        true_label = y.loc[current_date] if current_date in y.index else np.nan
        pred_label = 1 if pred_proba >= 0.5 else 0
        
        asset_pred_record[asset].append({
            'date': current_date,
            'true': true_label,
            'pred': pred_label,
            'proba': pred_proba
        })
    
    # Store predictions for this asset.
    svm_predictions[asset] = asset_preds

# Save the best parameters.
with open('best_svm_params.json', 'w') as f:
    json.dump(best_params_dict, f, indent=4)

# ==================== 10. Construct synthetic quadrant assets ====================
asset_grid = {
    (1, 1): ['南华商品指数_动量过滤', '沪金_动量过滤', '上证50', '红利收益', '中证转债'],
    (1, 2): ['沪深300', '上证50', '中证1000', '中证转债', '中债企业债'],
    (2, 1): ['南华商品指数_动量过滤', '沪金_动量过滤', '中债金融债'],
    (2, 2): ['中债国债', '中债金融债', '中债企业债', '红利收益']
}

LOOKBACK_PERIOD = 12
quad_cols = []

for key, assets in asset_grid.items():
    vol = merged_df[assets].rolling(window=LOOKBACK_PERIOD).std() * np.sqrt(12)
    vol = vol.shift(1)
    inv_vol = 1 / vol
    weight = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    merged_df[f'新资产_{key}'] = (merged_df[assets] * weight).sum(axis=1)
    quad_cols.append(f'新资产_{key}')

# ==================== 11. All-weather portfolio with risk parity across quadrants ====================
quad_vol = merged_df[quad_cols].rolling(window=LOOKBACK_PERIOD).std() * np.sqrt(12)
quad_vol = quad_vol.shift(1)
top_inv_vol = 1 / quad_vol
top_weights = top_inv_vol.div(top_inv_vol.sum(axis=1), axis=0)
merged_df['全天候组合收益'] = (merged_df[quad_cols] * top_weights).sum(axis=1)

# ==================== 12. SVM-filtered portfolio ====================
svm_weights = pd.DataFrame(0.0, index=merged_df.index, columns=all_assets)

# Precompute volatility with a minimum floor.
MIN_VOLATILITY = 0.001  # Minimum volatility prevents division by zero.
asset_vol = merged_df[all_assets].rolling(window=LOOKBACK_PERIOD).std() * np.sqrt(12)
asset_vol = asset_vol.shift(1)
asset_vol = asset_vol.fillna(asset_vol.mean())  # Fill NaN values with the mean.
asset_vol = asset_vol.clip(lower=MIN_VOLATILITY)  # Apply the volatility floor.

# Use dates from 2014 onward.
start_date = pd.Timestamp('2014-01-01')
prediction_dates = merged_df.index[merged_df.index >= start_date]

# Use only dates present in both datasets.
common_dates = prediction_dates.intersection(svm_predictions.index)

for idx in common_dates:
    # Get predictions for this date.
    pred_row = svm_predictions.loc[idx]
    
    # Select assets whose predicted probability exceeds 0.5.
    selected_assets = [
        asset for asset in all_assets
        if pd.notna(pred_row[asset]) and pred_row[asset] > 0.5
    ]
    
    if not selected_assets:
        continue
    
    # Get volatility.
    vol_values = asset_vol.loc[idx, selected_assets]
    
    # Calculate inverse-volatility weights.
    inv_vol = 1 / vol_values
    total_inv_vol = inv_vol.sum()
    
    if total_inv_vol > 0:
        weights = inv_vol / total_inv_vol
    else:
        # Use equal weights if the inverse-volatility sum is zero.
        weights = pd.Series(1/len(selected_assets), index=selected_assets)
    
    svm_weights.loc[idx, selected_assets] = weights.values

# SVM portfolio return
svm_portfolio_returns = pd.Series(0.0, index=merged_df.index)
for idx in common_dates:
    weights = svm_weights.loc[idx]
    returns = merged_df.loc[idx, all_assets]
    svm_portfolio_returns.loc[idx] = (weights * returns).sum()

merged_df['SVM筛选组合收益'] = svm_portfolio_returns

# ==================== 13. Backtest analysis ====================
# Analyze dates from 2014 onward only.
backtest_df = merged_df[merged_df.index >= '2014-01-01']

# Cumulative returns
backtest_df['SVM筛选累计收益'] = (1 + backtest_df['SVM筛选组合收益']).cumprod()
backtest_df['全天候累计收益'] = (1 + backtest_df['全天候组合收益']).cumprod()

# Plot results.
plt.figure(figsize=(12, 6))
plt.plot(backtest_df.index, backtest_df['全天候累计收益'], label='全天候策略', linewidth=2, color='blue')
plt.plot(backtest_df.index, backtest_df['SVM筛选累计收益'], label='SVM筛选策略', linewidth=2, color='red')
plt.xlabel('日期')
plt.ylabel('累计收益')
plt.title('全天候策略 vs SVM筛选策略 (2014-至今)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('策略对比.png', dpi=300)
plt.close()

# Performance metrics
def calculate_performance(returns, freq=12):
    cumulative_return = (1 + returns).prod() - 1
    annual_return = (1 + cumulative_return) ** (freq / len(returns)) - 1
    annual_volatility = returns.std() * np.sqrt(freq)
    sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else 0
    wealth_index = (1 + returns).cumprod()
    previous_peaks = wealth_index.cummax()
    drawdowns = (wealth_index - previous_peaks) / previous_peaks
    max_drawdown = drawdowns.min()
    return {
        '累计收益': cumulative_return,
        '年化收益': annual_return,
        '年化波动': annual_volatility,
        '夏普比率': sharpe_ratio,
        '最大回撤': max_drawdown
    }

aw_perf = calculate_performance(backtest_df['全天候组合收益'].dropna())
svm_perf = calculate_performance(backtest_df['SVM筛选组合收益'].dropna())

perf_df = pd.DataFrame([aw_perf, svm_perf], index=['全天候策略', 'SVM筛选策略'])

# ==================== 14. SVM prediction performance by asset ====================
performance_summary = []

for asset in all_assets:
    records = asset_pred_record.get(asset, [])
    if not records:
        continue
        
    dates = [r['date'] for r in records]
    y_true = np.array([r['true'] for r in records])
    y_pred = np.array([r['pred'] for r in records])
    y_proba = np.array([r['proba'] for r in records])
    
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_proba)
    hit_rate = (y_proba > 0.5).mean()  # Fraction with predicted up probability above 0.5
    
    performance_summary.append({
        '资产': asset,
        '样本数': len(y_true),
        '准确率': acc,
        'F1分数': f1,
        'AUC': auc,
        '上涨预测占比': hit_rate
    })

# Convert to a DataFrame for analysis.
perf_df_by_asset = pd.DataFrame(performance_summary).sort_values('AUC', ascending=False)

# ==================== 15. Statistical tests ====================
# Test whether strategy returns differ significantly.
t_stat, p_value = ttest_ind(
    backtest_df['SVM筛选组合收益'].dropna(), 
    backtest_df['全天候组合收益'].dropna(),
    equal_var=False
)

# ==================== 16. Compact output ====================
print("\n" + "="*50)
print("📊 策略绩效对比 (2014-至今):")
print("="*50)
print(perf_df.round(4))
print("\n" + "="*50)

print("\n📈 关键策略表现指标:")
print("-"*50)
print(f"全天候策略: 年化收益 = {perf_df.loc['全天候策略', '年化收益']:.4f}, 夏普比率 = {perf_df.loc['全天候策略', '夏普比率']:.4f}")
print(f"SVM筛选策略: 年化收益 = {perf_df.loc['SVM筛选策略', '年化收益']:.4f}, 夏普比率 = {perf_df.loc['SVM筛选策略', '夏普比率']:.4f}")
print("-"*50)

print("\n📊 策略收益差异显著性检验:")
print(f"SVM筛选 vs 全天候: t-statistic={t_stat:.3f}, p-value={p_value:.3f}")
print("="*50)

print("\n📈 各资产 SVM 预测性能 (按 AUC 排序):")
print("="*50)
print(perf_df_by_asset[['资产', 'AUC', '准确率', 'F1分数', '上涨预测占比']].round(3).to_string(index=False))
print("="*50)


# ==================== 17. Save results ====================
# Save the merged data.
try:
    backtest_df.to_csv('strategy_backtest_results.csv')
    print("✅ Backtest results saved to 'strategy_backtest_results.csv'")
except PermissionError:
    print("⚠️ Could not save strategy_backtest_results.csv; the file may be open in another program.")

# Save SVM predictions.
try:
    svm_predictions.to_csv('svm_predictions.csv')
    print("✅ SVM predictions saved to 'svm_predictions.csv'")
except PermissionError:
    print("⚠️ Could not save svm_predictions.csv; the file may be open in another program.")

# Save SVM weights.
try:
    svm_weights.to_csv('svm_weights.csv')
    print("✅ SVM weights saved to 'svm_weights.csv'")
except PermissionError:
    print("⚠️ Could not save svm_weights.csv; the file may be open in another program.")

print("\n✅ 所有策略回测完成！关键结果已输出")
