from pydlm import dlm, trend, seasonality, dynamic
import numpy as np

class dynamicModel:
    def __init__(self, y: list, features: np.array, dates: list):
        self.y = y
        self.features = features
        self.dates = dates
        self.model = None

    def constructor(self):
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

class hierarchicalModel:
    def __init__(self):
        pass