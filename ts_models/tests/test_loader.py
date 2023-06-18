from src import loader
import pandas as pd
import numpy as np
import unittest

class test_loader(unittest.TestCase):
    def setUp(self):
        self.flat_file = '/root/ts_models/ts_models/src/DP_LIVE_16062023204717297.csv'

    def test_read(self):
        rate_load = loader.ratesLoader(self.flat_file)
        self.loc, self.freq, self.year, self.value = rate_load.read_data()
        self.df = pd.DataFrame(np.concatenate([self.loc, self.freq, self.year, self.value], axis=1), columns = ['loc','freq','year','value'])
        self.df.value = self.df.value.astype(float)
        self.assertEqual(self.df.shape[1],4)
        self.assertLess(self.df.value.min(),self.df.value.max())

    def test_stats(self):
        rate_load = loader.ratesLoader(self.flat_file)
        rate_load.get_stats()