import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import metrics, calculate_metrics_table

class EvaluateModel:
    def __init__(self, model, test_data: pd.DataFrame, weights: np.ndarray, alpha: float = 99, window_size: int = 300):
        self.model = model
        self.test_data = test_data
        self.weights = weights
        self.alpha = alpha
        self.window_size = window_size

    def calculate_test_var(self):
        portfolio = self.test_data.dot(self.weights)
        portfolio = pd.DataFrame(portfolio)
        self.portfolio = portfolio
        var_values_df = self.model.predict_var_rolling_window(self.test_data, self.weights)
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
        plt.savefig(f"{self.model.name}_plot.png")
        plt.show()

    def generate_metrics_table(self):
        metrics_dict = metrics(self.var_df['VaR'], self.var_df['Returns'])
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index', columns=['Value'])
        return metrics_df

    def evaluate_model(self):
        self.calculate_test_var()
        self.var_df.to_csv(f"{self.model.name}_predictions.csv", index=True)
        self.plot_breakdown_graph()
        metrics_table = self.generate_metrics_table()
        print(metrics_table)
        