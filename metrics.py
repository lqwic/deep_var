import numpy as np
from scipy.stats import chi2, norm
import pandas as pd

def pof_test(
    var: np.ndarray,
    target: np.ndarray,
    alpha: float = 0.99,
) -> float:
    """
    Kupiec’s Proportion of Failure Test (POF). Tests that a number of exceptions
    corresponds to the VaR confidence level.

    Parameters:
        var: Predicted VaRs.
        target: Corresponded returns.
        alpha: VaR confidence level. Default is 0.99.

    Returns:
        p-value of POF test.
    """
    exception = target < var
    t = len(target)
    m = exception.sum()
    nom = (1 - alpha)**m * alpha**(t-m)
    den = (1 - m/t)**(t - m) * (m / t)**m
    lr_pof = -2 * np.log(nom / den)
    pvalue = 1 - chi2.cdf(lr_pof, df=1)
    return pvalue

def cc_test(var, target, alpha=0.05):
    """
    Christoffersen's Conditional Coverage Test (CC) assesses the independence of exceptions
    in VaR forecasts. It tests whether the exceptions occur randomly or exhibit dependence.

    Parameters:
        var (numpy.ndarray): Predicted VaRs.
        target (numpy.ndarray): Corresponded returns.
        alpha (float): Significance level. Default is 0.05.

    Returns:
        float: p-value of the CC test.
    """
    exceptions = (target < var).astype(int)
    num_exceptions = exceptions.sum()
    num_forecasts = len(var)
    num_bins = 2
    num_parameters = 1
    expected_exceptions = num_exceptions * (num_forecasts / num_bins)
    chi_squared = ((num_exceptions - expected_exceptions)**2) / expected_exceptions
    p_value = 1 - chi2.cdf(chi_squared, df=num_parameters)
    return p_value

def berkowitz_test(var, target, alpha=0.05):
    """
    Berkowitz Test assesses the calibration of VaR forecasts. It tests whether the realized
    losses are consistent with the predicted VaR level.

    Parameters:
        var (numpy.ndarray): Predicted VaRs.
        target (numpy.ndarray): Corresponded returns.
        alpha (float): Significance level. Default is 0.05.

    Returns:
        float: p-value of the Berkowitz test.
    """
    num_forecasts = len(var)
    excess_losses = target - var
    z_scores = excess_losses / np.std(excess_losses, ddof=1)
    p_value = 1 - norm.cdf(z_scores.mean())
    return p_value

def quantile_loss(var : np.ndarray, target: np.ndarray, alpha : float = 0.99) -> float:
    """
    Quantile loss also known as Pinball loss. Measures the discrepancy between
    true values and a corresponded 1-alpha quantile.

    Parameters:
        var:
            Predicted VaRs.
        target:
            Corresponded returns.
        alpha:
            VaR confidence level. Default is 0.99.

    Returns:
        The avarage value of the quantile loss function.
    """
    qloss = np.abs(var-target)
    qloss[target < var] = qloss[target < var] * 2 * alpha
    qloss[target >= var] = qloss[target >= var] * 2 * (1 - alpha)
    return qloss.mean()

def quadratic_loss(var : np.ndarray, target: np.ndarray, alpha : float = 0.99, a=1):
    """
    Quadratic Loss measures the squared difference between the predicted VaR and PnL,
    penalizing negative profits with weight (PnL - VaR)^2 and negative VaRs with
    weight -a * VaR.

    Parameters:
        var (numpy.ndarray): Predicted VaRs.
        target (numpy.ndarray): Corresponded returns.
        alpha (float): Weight parameter for PnL - VaR. Default is 0.99.
        a (float): Weight parameter for negative VaRs. Default is 1.

    Returns:
        float: Quadratic Loss value.
    """
    quadratic_loss_value = np.where(target < var, 1 + (target - var)**2, 0)
    return quadratic_loss_value.mean()


def smooth_loss(var : np.ndarray, target: np.ndarray, alpha : float = 0.99, d = 25):
    """
    Smooth Loss penalizes observations for which PnL - VaR < 0 more heavily with weight (1-alpha).

    Parameters:
        var (numpy.ndarray): Predicted VaRs.
        target (numpy.ndarray): Corresponded returns.
        alpha (float): Weight parameter. Default is 0.99.
        d (int): Parameter Default is 25.

    Returns:
        float: Smooth Loss value.
    """
    N = len(var)
    smooth_loss_value = (1 / N) * np.sum(alpha - (1 + np.exp(d * (target - var)))**(-1)) * (target - var)
    return smooth_loss_value.mean()


def tick_loss(var : np.ndarray, target: np.ndarray, alpha : float = 0.99,):
    """
    Tick Loss penalizes exceedances with weight alpha and non-exceedances with weight 1 - alpha.

    Parameters:
        var (numpy.ndarray): Predicted VaRs.
        target (numpy.ndarray): Corresponded returns.
        alpha (float): Weight parameter. Default is 0.99.

    Returns:
        float: Tick Loss value.
    """
    tick_loss_value = np.sum((alpha - (target > 0)) * (target - var))
    return tick_loss_value


def firm_loss(var : np.ndarray, target: np.ndarray, alpha : float = 0.99, a = 1):
    """
    Firm Loss imposes the opportunity cost of capital upon the firm.

    Parameters:
        var (numpy.ndarray): Predicted VaRs.
        target (numpy.ndarray): Corresponded returns.
        alpha (float): Weight parameter. Default is 0.99.
        a (float): Opportunity cost of capital. Default is 1.

    Returns:
        float: Firm Loss value.
    """
    firm_loss_value = np.where(target < var, ((target - var)**2), -a * var).mean()
    return firm_loss_value

def metrics(var : np.ndarray, target: np.ndarray, alpha : float = 0.99):
    metrics_dict = {}

    pof_pvalue = pof_test(var, target, alpha)
    metrics_dict['POF Test p-value'] = pof_pvalue

    cc_pvalue = cc_test(var, target, alpha)
    metrics_dict['CC Test p-value'] = cc_pvalue

    berkowitz_pvalue = berkowitz_test(var, target, alpha)
    metrics_dict['Berkowitz Test p-value'] = berkowitz_pvalue

    qloss = quantile_loss(var, target, alpha)
    metrics_dict['Quantile Loss'] = qloss

    quad_loss = quadratic_loss(var, target, alpha)
    metrics_dict['Quadratic Loss'] = quad_loss

    smooth_loss_val = smooth_loss(var, target, alpha)
    metrics_dict['Smooth Loss'] = smooth_loss_val

    tick_loss_val = tick_loss(var, target, alpha)
    metrics_dict['Tick Loss'] = tick_loss_val

    firm_loss_val = firm_loss(var, target, alpha)
    metrics_dict['Firm Loss'] = firm_loss_val

    return metrics_dict

def calculate_metrics_table(target, predictions_dict, alpha=0.99):
    """
    Calculate metrics table for multiple models.

    Parameters:
        target (pd.Series): Target series.
        predictions_dict (dict): Dictionary with keys as model names and values as predicted VaRs.
        alpha (float): VaR confidence level. Default is 0.99.

    Returns:
        pd.DataFrame: DataFrame containing metrics for each model.
    """
    metrics_list = [
      'POF Test p-value',
      'CC Test p-value',
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
        metrics_table[model] = metrics_dict

    return metrics_table
