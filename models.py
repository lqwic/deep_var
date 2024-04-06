import numpy as np
import pandas as pd
from scipy import stats
from tqdm.auto import trange

from gluonts.dataset.common import ListDataset
from gluonts.mx.model import deepar
from gluonts.mx.trainer import Trainer


class BaseModel:
    '''Parent class'''

    def __init__(self, alpha=99, window_size=300, **kwargs):
        self.name = 'BaseModel'
        self.alpha = alpha
        self.window_size = window_size

    def fit(self, ts):
        raise NotImplementedError("fit method must be implemented in child class")

    def predict_var_one_day(self, returns, weights):
        raise NotImplementedError("predict_var_one_day method must be implemented in child class")

    def predict_var_rolling_window(self, ts, weights):
        var_values = []
        for i in trange(self.window_size, len(ts)):
            current_date = ts.index[i].date()
            current_returns = ts.iloc[i - self.window_size:i]
            current_var = self.predict_var_one_day(current_returns, weights)
            var_values.append({'Date': current_date, 'VaR': current_var})
        var_values_df = pd.DataFrame(var_values)
        return var_values_df
    

class HistoricalSimulation(BaseModel):
    def __init__(self, alpha=99, window_size=300):
        super().__init__(alpha=alpha, window_size=window_size)
        self.name = 'HistoricalSimulation'

    def fit(self, ts):
      pass

    def hs(self, returns, alpha):
        '''Historical Simulation VaR'''
        return np.percentile(returns, 100 - alpha)

    def predict_var_one_day(self, returns, weights):
        R = returns.corr()
        V = np.zeros(len(weights))
        for i in range(len(weights)):
            if weights[i] < 0:
                V[i] = weights[i] * self.hs(returns.iloc[:, i], alpha=self.alpha)
            else:
                V[i] = weights[i] * self.hs(returns.iloc[:, i], alpha=100 - self.alpha)
        return -np.sqrt(V @ R @ V.T)
    

class VarianceCovariance(BaseModel):
    def __init__(self, alpha=99, window_size=300):
        super().__init__(alpha=alpha, window_size=window_size)
        self.name = 'VarianceCovariance'

    def fit(self, ts):
      pass

    def vc(self, x, alpha):
        '''Variance-covariance VaR'''
        c = alpha / 100
        return x.std() * stats.norm.ppf(c)

    def predict_var_one_day(self, returns, weights):
        R = returns.corr()
        V = np.zeros(len(weights))
        for i in range(len(weights)):
            V[i] = np.abs(weights[i]) * self.vc(returns.iloc[:, i], alpha=self.alpha)
        return -np.sqrt(V @ R @ V.T)
    
    
class MonteCarlo(BaseModel):
    def __init__(self, alpha=99, window_size=300):
        super().__init__(alpha=alpha, window_size=window_size)
        self.name = 'MonteCarlo'

    def fit(self, ts):
      pass

    def mc(self, x, alpha, n_sims=5000, seed=42):
        '''Monte Carlo VaR'''
        np.random.seed(seed)
        sim_returns = np.random.normal(x.mean(), x.std(), n_sims)
        return np.percentile(sim_returns, alpha)

    def predict_var_one_day(self, returns, weights):
        R = returns.corr()
        V = np.zeros(len(weights))
        for i in range(len(weights)):
            if weights[i] < 0:
                V[i] = weights[i] * self.mc(returns.iloc[:, i], alpha=self.alpha)
            else:
                V[i] = weights[i] * self.mc(returns.iloc[:, i], alpha=100 - self.alpha)
        return -np.sqrt(V @ R @ V.T)
    

class DeepARModel(BaseModel):
    '''Class for fitting and predicting with GluonTS DeepAR estimator '''

    def __init__(self, context_length=15, alpha = 99, window_size = 300,
                 epochs=5, learning_rate=1e-4, n_layers=2., dropout=0.1):
        super().__init__(alpha=alpha, window_size=window_size)
        self.name = 'DeepVaR'
        self.context_length = context_length
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.n_layers = n_layers
        self.dropout = dropout

    def df_to_np(self, ts):
        return ts.to_numpy().T

    def list_dataset(self, ts, train=True):

        '''expects as input a pandas df with datetime index and
        columns the asset returns and outputs the train or test dataset in
        a proper form to be used as intput to a GluonTS estimator'''

        custom_dataset = self.df_to_np(ts)
        start = pd.Timestamp(ts.index[0])
        if train == True:
            ds = ListDataset([{'target': x, 'start': start}
                              for x in custom_dataset[:, :-1]],
                             freq='1D')
        else:
            ds = ListDataset([{'target': x, 'start': start}
                              for x in custom_dataset],
                             freq='1D')
        return ds

    def fit(self, ts):

        '''expects as input a pandas df with datetime index and
        columns the returns of the assets to be predicted'''
        # iniallize deepar estimator
        estimator = deepar.DeepAREstimator(
            prediction_length=1,
            context_length=self.context_length,
            freq='1D',
            trainer=Trainer(epochs=self.epochs,
                            ctx="cpu",
                            learning_rate=self.learning_rate,
                            num_batches_per_epoch=50,
                            ),
            num_layers=self.n_layers,
            dropout_rate=self.dropout,
            cell_type='lstm',
            num_cells=50
        )
        # prepare training data
        list_ds = self.list_dataset(ts, train=True)
        # train deepar on training data
        predictor = estimator.train(list_ds)
        self.estimator = predictor
        return predictor

    def predict_ts(self, ts):
        '''expects as input a pandas df with datetime index and
        columns the returns of the assets to be predicted'''
        # get the test data in proper form
        test_ds = self.list_dataset(ts, train=False)
        return self.estimator.predict(test_ds, num_samples=1000)

    def predict_var_one_day(self, returns, weights):
        V = np.zeros(len(weights))
        predictions_it = self.predict_ts(returns)
        predictions = list(predictions_it)
        for i in range(len(weights)):
            if weights[i] < 0 :
                V[i] = weights[i] * np.percentile(predictions[i].samples[:, 0], self.alpha)
            else:
                V[i] = weights[i] * np.percentile(predictions[i].samples[:, 0], 100-self.alpha)
        R = returns.corr()
        return -np.sqrt(V @ R @ V.T)
    