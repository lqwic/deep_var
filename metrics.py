import numpy as np
from scipy.stats import chi2, norm
import pandas as pd

def pof_test(var: np.ndarray, target: np.ndarray, alpha: float = 0.99) -> float:
    """
    Kupiec's Proportion of Failure Test (POF). Tests that the number of exceptions
    corresponds to the VaR confidence level.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: VaR confidence level. Default is 0.99.

    Returns:
        p-value of POF test.
    """
    exception = target < var
    t = len(target)
    m = exception.sum()
    p = 1 - alpha
    lr_pof = -2 * (m * np.log(p / (m / t)) + (t - m) * np.log((1 - p) / (1 - m / t)))
    pvalue = 1 - chi2.cdf(lr_pof, df=1)
    return pvalue

def berkowitz_test(var: np.ndarray, target: np.ndarray, alpha: float = 0.99) -> float:
    """
    Berkowitz Test assesses the calibration of VaR forecasts. It tests whether the realized
    losses are consistent with the predicted VaR level.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: VaR confidence level. Default is 0.99.

    Returns:
        p-value of the Berkowitz test.
    """
    z = (target - var.mean()) / var.std()
    lr_berkowitz = -2 * (np.log(norm.cdf(z)).sum() - np.log(alpha) * (target < var).sum() - np.log(1 - alpha) * (target >= var).sum())
    pvalue = 1 - chi2.cdf(lr_berkowitz, df=1)
    return pvalue

def quantile_loss(var: np.ndarray, target: np.ndarray, alpha: float = 0.99) -> float:
    """
    Quantile loss also known as Pinball loss. Measures the discrepancy between
    true values and a corresponding 1-alpha quantile.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: VaR confidence level. Default is 0.99.

    Returns:
        The average value of the quantile loss function.
    """
    return np.where(target < var, alpha * (var - target), (1 - alpha) * (target - var)).mean()

def quadratic_loss(var: np.ndarray, target: np.ndarray, alpha: float = 0.99, a: float = 1.0) -> float:
    """
    Quadratic Loss measures the squared difference between the predicted VaR and returns,
    penalizing negative returns with weight (return - VaR)^2 and negative VaRs with
    weight -a * VaR.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: Weight parameter for return - VaR. Default is 0.99.
        a: Weight parameter for negative VaRs. Default is 1.

    Returns:
        Quadratic Loss value.
    """
    return np.where(target < var, (target - var)**2, -a * var).mean()

def smooth_loss(var: np.ndarray, target: np.ndarray, alpha: float = 0.99, d: float = 25.0) -> float:
    """
    Smooth Loss penalizes observations for which return - VaR < 0 more heavily with weight (1-alpha).

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: Weight parameter. Default is 0.99.
        d: Parameter Default is 25.

    Returns:
        Smooth Loss value.
    """
    return ((alpha - (1 + np.exp(d * (target - var)))**(-1)) * (target - var)).mean()

def tick_loss(var: np.ndarray, target: np.ndarray, alpha: float = 0.99) -> float:
    """
    Tick Loss penalizes exceedances with weight alpha and non-exceedances with weight 1 - alpha.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: Weight parameter. Default is 0.99.

    Returns:
        Tick Loss value.
    """
    return ((alpha - (target < var).astype(float)) * (target - var)).mean()

def firm_loss(var: np.ndarray, target: np.ndarray, alpha: float = 0.99, a: float = 1.0) -> float:
    """
    Firm Loss imposes the opportunity cost of capital upon the firm.

    Parameters:
        var: Predicted VaRs.
        target: Corresponding returns.
        alpha: Weight parameter. Default is 0.99.
        a: Opportunity cost of capital. Default is 1.

    Returns:
        Firm Loss value.
    """
    return np.where(target < var, (target - var)**2, -a * var).mean()

def metrics(var: np.ndarray, target: np.ndarray, alpha: float = 0.99) -> dict:
    metrics_dict = {}

    metrics_dict['POF Test p-value'] = pof_test(var, target, alpha)
    metrics_dict['Berkowitz Test p-value'] = berkowitz_test(var, target, alpha)
    metrics_dict['Quantile Loss'] = quantile_loss(var, target, alpha)
    metrics_dict['Quadratic Loss'] = quadratic_loss(var, target, alpha)
    metrics_dict['Smooth Loss'] = smooth_loss(var, target, alpha)
    metrics_dict['Tick Loss'] = tick_loss(var, target, alpha)
    metrics_dict['Firm Loss'] = firm_loss(var, target, alpha)

    return metrics_dict

def calculate_metrics_table(target: pd.Series, predictions_dict: dict, alpha: float = 0.99) -> pd.DataFrame:
    """
    Calculate metrics table for multiple models.

    Parameters:
        target: Target series.
        predictions_dict: Dictionary with keys as model names and values as predicted VaRs.
        alpha: VaR confidence level. Default is 0.99.

    Returns:
        DataFrame containing metrics for each model.
    """
    metrics_list = [
        'POF Test p-value',
        'Berkowitz Test p-value',
        'Quantile Loss',
        'Quadratic Loss',
        'Smooth Loss',
        'Tick Loss',
        'Firm Loss'
    ]
    metrics_table = pd.DataFrame(index=metrics_list)

    for model, predictions in predictions_dict.items():
        metrics_dict = metrics(predictions, target, alpha)
        metrics_table[model] = metrics_dict.values()

    return metrics_table
