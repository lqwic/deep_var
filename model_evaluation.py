import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import trange

from metrics import metrics, calculate_metrics_table

class EvaluateModel:
    def __init__(self, model, data: pd.DataFrame,
                 alpha = 99, window_size = 300, train_size = 0.8):
        self.model = model
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
        weights = self.generate_random_weights()
        self.calculate_test_var(weights)
        self.plot_breakdown_graph()
        print(self.generate_metrics_table(n_samples = n_samples))


class EvaluateMultipleModels:
    def __init__(self, trained_models, data, alpha=99, window_sizes=None, train_size=0.8):
        self.trained_models = trained_models
        self.data = data
        self.alpha = alpha
        self.window_sizes = window_sizes if window_sizes else [300] * len(trained_models)
        self.train_size = train_size

    def split_data(self):
        split_index = int(len(self.data) * self.train_size)
        self.train_data = self.data.iloc[:split_index]
        self.test_data = self.data.iloc[split_index:]

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

    def evaluate_models(self, weights):
        self.predictions_dict = {}
        self.var_df = pd.DataFrame(index=self.test_data.index)
        for model, window_size in zip(self.trained_models, self.window_sizes):
            test_data_model = self.test_data.iloc[-window_size:]
            predictions = model.predict_var_rolling_window(test_data_model, weights)
            predictions = predictions.set_index('Date')
            predictions.index = pd.to_datetime(predictions.index)
            self.predictions_dict[model.name] = predictions.values.flatten()
            self.var_df[model.name] = predictions.values.flatten()
        self.portfolio = self.test_data[max(self.window_sizes):].dot(weights)
        portfolio = self.portfolio.values
        self.metrics_table = calculate_metrics_table(portfolio, self.predictions_dict, self.alpha)

    def plot_model_results(self):
        plt.figure(figsize=(15, 10))
        plt.plot(self.portfolio.index, self.portfolio.values, label=f"Actual returns", alpha=0.5)
        for model in self.trained_models:
            var = self.predictions_dict[model.name]
            plt.plot(self.portfolio.index, var, label=f"{model.name} VaR", linestyle='--')
        plt.legend()
        plt.title("Model Predictions vs. Actual Returns")
        plt.show()

    def get_metrics_table(self):
        return self.metrics_table

    def get_var_df(self):
        return self.var_df

    def run_evaluation(self):
        self.split_data()
        weights = self.generate_random_weights()
        self.evaluate_models(weights)
        self.plot_model_results()
        print(self.get_metrics_table())
        