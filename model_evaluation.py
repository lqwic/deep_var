import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import trange

from models import BaseModel
from metrics import metrics

class EvaluateModel:
    def __init__(self, model: BaseModel, data: pd.DataFrame,
                 alpha = 99, window_size = 300, train_size = 0.8):
        self.model = model(alpha = alpha, window_size = window_size)
        self.data = data
        self.alpha = alpha
        self.window_size = window_size
        self.train_size = train_size

    def temporal_split(self):
        split_index = int(len(self.data) * self.train_size)
        train_data = self.data.iloc[:split_index]
        test_data = self.data.iloc[split_index - self.window_size:]
        self.train = train_data
        self.test = test_data
        return train_data, test_data

    def generate_random_weights(self):
        num_columns = self.data.shape[1]
        random_weights = np.random.rand(num_columns)
        normalized_weights = random_weights / sum(random_weights)
        return normalized_weights

    def generate_random_single_asset_weight(self):
        num_columns = self.data.shape[1]
        weights = np.zeros(num_columns)
        selected_asset = np.random.randint(num_columns)
        weights[selected_asset] = 1
        return weights

    def train_model(self, train):
        self.estimator = self.model.fit(train)

    def calculate_test_var(self, weights):
        portfolio = self.test.dot(weights)
        portfolio = pd.DataFrame(portfolio)
        self.portfolio = portfolio
        var_values_df = self.model.predict_var_rolling_window(self.test, weights)
        var_values_df = var_values_df.set_index('Date')
        var_values_df.index = pd.to_datetime(var_values_df.index)
        var_df = portfolio.merge(var_values_df, on='Date')
        var_df.columns = ['Returns', 'VaR']
        self.var_df = var_df
        return var_df

    def plot_breakdown_graph(self):
        fig, ax = plt.subplots(figsize=(10, 6))
        pos = self.portfolio[self.portfolio.apply(lambda x: x >= 0)][self.window_size:]
        neg = self.portfolio[self.portfolio.apply(lambda x: x <  0)][self.window_size:]
        self.var_df['Violation'] = (self.var_df['Returns'] < self.var_df['VaR'])
        breakdowns = self.var_df[self.var_df['Violation'] == 1]
        plt.scatter(pos.index,pos, c = 'blue', alpha = 0.7)
        plt.scatter(neg.index,neg, c = 'orange', alpha = 0.7)
        plt.scatter(breakdowns.index, breakdowns.Returns, c = 'red', alpha = 0.8,  label = 'violations')
        plt.plot(self.var_df.index, self.var_df.VaR, label = 'VaR', c = 'black')
        plt.legend()
        plt.title(f"{self.model.name} {self.alpha}% violations")
        plt.show()

    def generate_metrics_table(self, n_samples=10):
        metrics_sums = {}
        for _ in trange(n_samples):
            weights = self.generate_random_weights()
            var_df = self.calculate_test_var(weights)
            single_run_metrics = metrics(var_df['VaR'], var_df['Returns'])
            for metric_name, metric_value in single_run_metrics.items():
                if metric_name in metrics_sums:
                    metrics_sums[metric_name] += metric_value
                else:
                    metrics_sums[metric_name] = metric_value
        metrics_means = {metric: total / n_samples for metric, total in metrics_sums.items()}
        metrics_df = pd.DataFrame(list(metrics_means.items()), columns=['Metric', 'Mean Value'])
        metrics_df = metrics_df.set_index('Metric')
        return metrics_df

    def evaluate_model(self, n_samples=10):
        self.temporal_split()
        self.train_model(self.train)
        weights = self.generate_random_weights()
        self.calculate_test_var(weights)
        self.plot_breakdown_graph()
        print(self.generate_metrics_table(n_samples = n_samples))
