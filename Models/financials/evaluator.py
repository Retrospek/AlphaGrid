import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from algos import GARCH

class VolatilityEvaluator:
    def __init__(self, tolerance=1.0):
        """
        tolerance: percentage point margin of error for hit rate calculation
        """
        self.tolerance = tolerance        
    def hit_rate(self, y_true, y_pred):
        
        errors = np.abs(y_true - y_pred)
        hits = np.sum(errors <= self.tolerance)
        hit_rate = hits / len(y_true) * 100
        
        print(f"Hit Rate (±{self.tolerance}%): {hit_rate:.2f}%")
        print(f"Hits: {hits}/{len(y_true)}")
        
        return hit_rate
    
    def garch_baseline(self, returns_series, forecast_horizon=1, p_q=[1,1]):
        """
        Fit GARCH(1,1) model and generate volatility forecasts
        
        Args:
            returns_series: pandas Series of daily returns
            forecast_horizon: days ahead to forecast
            
        Returns:
            garch_predictions: predicted volatility changes
        """
        p = p_q[0]
        q = p_q[1]

        model = GARCH(p=p, q=q)
        fitted_model = model.fit()
        
        garch_predictions = []
        
        # Use expanding window for forecasting
        for i in range(len(returns_series) - 252):
            # Use data up to day i+252
            subset = returns_series.iloc[:i+252]
            
            model_subset = arch_model(subset, vol='Garch', p=1, q=1)
            fitted_subset = model_subset.fit()
            
            # Forecast next period volatility
            forecast = fitted_subset.forecast(horizon=forecast_horizon)
            vol_forecast = np.sqrt(forecast.variance.values[-1, 0])
            
            # Calculate volatility change (current vs forecast)

            current_vol = fitted_subset.conditional_volatility.iloc[-1]
            vol_change = ((vol_forecast - current_vol) / current_vol) * 100
            
            garch_predictions.append(vol_change)
        
        return np.array(garch_predictions)
    
    def sharpe_ratio_test(self, y_true, y_pred, returns_series, threshold=2.0):
        """
        Test economic significance using volatility timing strategy
        
        Args:
            y_true: actual volatility changes
            y_pred: predicted volatility changes  
            returns_series: underlying asset returns
            threshold: volatility change threshold for signal
            
        Returns:
            dict with strategy performance metrics
        """
        # Create trading signals based on volatility predictions
        # Signal: -1 (reduce exposure) if high volatility predicted, +1 (full exposure) otherwise
        signals = np.where(y_pred > threshold, -0.5, 1.0)  # Reduce exposure by 50% in high vol
        
        # Align signals with returns (predictions are for next day)
        aligned_returns = returns_series.iloc[len(returns_series)-len(signals):].values
        
        # Calculate strategy returns
        strategy_returns = signals * aligned_returns
        
        # Calculate performance metrics
        strategy_mean = np.mean(strategy_returns) * 252  # Annualized
        strategy_std = np.std(strategy_returns) * np.sqrt(252)  # Annualized
        strategy_sharpe = strategy_mean / strategy_std if strategy_std > 0 else 0
        
        # Buy and hold benchmark
        benchmark_mean = np.mean(aligned_returns) * 252
        benchmark_std = np.std(aligned_returns) * np.sqrt(252)
        benchmark_sharpe = benchmark_mean / benchmark_std if benchmark_std > 0 else 0
        
        results = {
            'strategy_sharpe': strategy_sharpe,
            'benchmark_sharpe': benchmark_sharpe,
            'sharpe_improvement': strategy_sharpe - benchmark_sharpe,
            'strategy_return': strategy_mean,
            'benchmark_return': benchmark_mean,
            'strategy_volatility': strategy_std,
            'benchmark_volatility': benchmark_std
        }
        
        print("=== Sharpe Ratio Test Results ===")
        print(f"Strategy Sharpe Ratio: {strategy_sharpe:.3f}")
        print(f"Benchmark Sharpe Ratio: {benchmark_sharpe:.3f}")
        print(f"Improvement: {results['sharpe_improvement']:.3f}")
        print(f"Strategy Annual Return: {strategy_mean:.2f}%")
        print(f"Benchmark Annual Return: {benchmark_mean:.2f}%")
        
        return results
    
    def comprehensive_evaluation(self, y_true, y_pred, returns_series):

        print("=== Comprehensive Volatility Model Evaluation ===\n")
        
        hit_rate = self.hit_rate(y_true, y_pred)
        
        print("\n" + "="*50)
        
        print("Fitting GARCH baseline...")
        garch_preds = self.garch_baseline(returns_series)
        
        # Align predictions for comparison
        min_len = min(len(y_true), len(garch_preds))
        y_true_aligned = y_true[-min_len:]
        y_pred_aligned = y_pred[-min_len:]
        garch_preds_aligned = garch_preds[-min_len:]
        
        mae_gnn = mean_absolute_error(y_true_aligned, y_pred_aligned)
        mae_garch = mean_absolute_error(y_true_aligned, garch_preds_aligned)
        
        print(f"\nMAE Comparison:")
        print(f"Your GNN Model: {mae_gnn:.3f}")
        print(f"GARCH Baseline: {mae_garch:.3f}")
        print(f"Improvement: {((mae_garch - mae_gnn) / mae_garch * 100):.1f}%")
        
        print("\n" + "="*50)
        
        sharpe_results = self.sharpe_ratio_test(y_true_aligned, y_pred_aligned, returns_series)
        
        return {
            'hit_rate': hit_rate,
            'mae_gnn': mae_gnn,
            'mae_garch': mae_garch,
            'sharpe_results': sharpe_results
        }

# Example usage:
"""
# Assuming you have:
# y_true: actual volatility changes (%)
# y_pred: your GNN predictions (%)  
# returns: pandas Series of daily returns

evaluator = VolatilityEvaluator(tolerance=2.0)  # 2% tolerance for hit rate

# Run comprehensive evaluation
results = evaluator.comprehensive_evaluation(y_true, y_pred, returns)
"""