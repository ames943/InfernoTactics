"""Serializable running statistics used to stabilize policy returns."""

import math

import numpy as np


class RunningMeanStd:
    """Welford online mean/variance with a conservative initial scale."""

    def __init__(self, epsilon=1e-4, init_var=5e3 ** 2):
        self.mean = 0.0
        self.var = float(init_var)
        self.count = epsilon

    def update(self, values):
        batch_mean = float(np.mean(values))
        batch_var = float(np.var(values))
        batch_count = len(values)
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        new_mean = self.mean + delta * batch_count / total_count
        first_moment = self.var * self.count
        batch_moment = batch_var * batch_count
        combined_moment = (
            first_moment
            + batch_moment
            + delta ** 2 * self.count * batch_count / total_count
        )
        self.mean = new_mean
        self.var = combined_moment / total_count
        self.count = total_count

    def normalize(self, values):
        return [(value - self.mean) / (math.sqrt(self.var) + 1e-8) for value in values]

    def state_dict(self):
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, state):
        self.mean = float(state["mean"])
        self.var = float(state["var"])
        self.count = float(state["count"])
