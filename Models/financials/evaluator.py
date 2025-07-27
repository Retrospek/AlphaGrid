import numpy as np
import torch
from arch import arch_model
import torch.nn as nn

from Models.Losses.high_variance import VolatilitySpikeLoss

class GARCHBaselineEvaluator:
    def __init__(self, returns_dict, y_true_dict, horizon=1, window=252):
        """
        Initialize the evaluator.

        Args:
            returns_dict (dict): {sector_idx: pd.Series} of raw returns per sector.
            y_true_dict (dict): {sector_idx: np.ndarray} ground truth volatility per sector.
            horizon (int): GARCH forecast horizon.
            window (int): Rolling window size for GARCH training.
            beta (float): beta parameter for SmoothL1Loss.
        """
        self.returns_dict = returns_dict
        self.y_true_dict = y_true_dict
        self.horizon = horizon
        self.window = window
        self.loss_fn = VolatilitySpikeLoss()
        self.evaluation_loss = nn.L1Loss()

    def _fit_garch(self, window_data):
        """
        Fit GARCH(1,1) and return sqrt of forecasted variance.
        """
        model = arch_model(window_data, vol='Garch', p=1, q=1)
        fitted = model.fit(disp='off')
        forecast = fitted.forecast(horizon=self.horizon)
        return np.sqrt(forecast.variance.iloc[-1, 0])

    def get_rolling_garch_forecasts(self):
        """
        Run rolling GARCH forecasts for all sectors.

        Returns:
            dict: {sector_idx: np.ndarray of GARCH rolling forecasts}
        """
        all_preds = {}
        for sector, returns_series in self.returns_dict.items():
            returns = returns_series.values
            preds = []
            for i in range(self.window, len(returns)):
                window_data = returns[i - self.window:i]
                pred = self._fit_garch(window_data)
                preds.append(pred)
            all_preds[sector] = np.array(preds)
        return all_preds

    def compute_smooth_l1_loss(self, garch_preds_dict, verbose=True):
        """
        Compute Smooth L1 loss between true volatility and GARCH predictions per sector.

        Args:
            garch_preds_dict (dict): {sector_idx: np.ndarray} of GARCH rolling forecasts.

        Returns:
            dict: {sector_idx: smooth_l1_loss}
        """
        losses = {}
        for sector in self.returns_dict.keys():
            y_true = self.y_true_dict[sector]
            preds = garch_preds_dict[sector]

            min_len = min(len(y_true), len(preds))
            y_true_aligned = torch.tensor(y_true[-min_len:], dtype=torch.float32)
            preds_aligned = torch.tensor(preds[-min_len:], dtype=torch.float32)

            loss = self.loss_fn(preds_aligned, y_true_aligned).item()
            losses[sector] = loss

            if verbose:
                print(f"Sector {sector} - Smooth L1 Loss (GARCH): {loss:.4f}")

        return losses