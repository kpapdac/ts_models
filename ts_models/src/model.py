from pydlm import dlm, trend, seasonality, dynamic
import numpy as np
from src import plate_models
import torch
import pyro
from pyro.contrib.forecast import ForecastingModel, Forecaster, eval_crps

class dynamicModel:
    def __init__(self, y: list, features: np.array, dates: list):
        self.y = y
        self.features = features
        self.dates = dates
        self.model = None

    def construct_(self):
        self.model = dlm(self.y)
        self.model = self.model + trend(1, name='lineTrend', w=1.0, discount=0.8)
        feat = np.array([self.features[i,:] for i in range(len(self.features.shape[0]))])
        self.model = self.model + dynamic(feat, discount=1, name='dyn')

    def fit_(self):
        self.model.fit()

    def get_error(self):
        return self.model.getMSE()

    def get_resid(self):
        return self.model.getResidual()

    def get_coef(self, coef_ind):
        return [self.model.getLatentState()[i][coef_ind] for i in range(len(self.model.getLatentState()))]

    def predict_insample(self, date_ind):
        date_ind_end = self.model.n - date_ind
        (predict_mean, predict_var)  = self.model.predict(date=self.model.n - date_ind_end)
        return (predict_mean, predict_var)

    def predict_outsample(self, featureDict_):
        featureDict = {'dyn': featureDict_}
        (predict_mean, predict_var) = self.model.predict(date=self.model.n - 1, featureDict)
        return (predict_mean, predict_var)

class linearModel:
    def __init__(self, data: np.array, covariates: np.array):
        self.data = torch.tensor(data)
        self.covariates = torch.tensor(covariates)
        if self.data.shape[0] != self.covariates.shape[0]:
            raise ValueError('Data and covariates dimensions do not match.')
        self.forecaster

    def construct_(self, learning_rate=0.1, fit_end_ind):
        pyro.set_rng_seed(1)
        pyro.clear_param_store()
        self.forecaster = Forecaster(plate_models.linearBayesian(), self.data[:fit_end_ind], self.covariates[:fit_end_ind], learning_rate=learning_rate)

    def predict_(self, fit_end_ind):
        self.samples = self.forecaster(self.data[fit_end_ind:], self.covariates, num_samples=1000)

    def get_quantile(self):
        p10, p50, p90 = quantile(self.samples, (0.1, 0.5, 0.9)).squeeze(-1)
        return p10, p50, p90

class hierarchicalModel:
    def __init__(self):
        pass