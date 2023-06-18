import pandas as pd
import numpy as np

class ratesLoader:
    def __init__(self, flat_file):
        self.flat_file = flat_file
        self.rates = None

    def read_data(self):
        with open(self.flat_file, 'r') as file:
            f = file.readlines()
            lines = np.array([i.replace('\ufeff','').replace('\n','').replace('"','').split(',') for i in f])
            location = lines[1:,lines[0,:]=='LOCATION']
            freq = lines[1:,lines[0,:]=='FREQUENCY']
            year = lines[1:,lines[0,:]=='TIME']
            value = lines[1:,lines[0,:]=='Value'].astype(float)
            return location, freq, year, value

    def get_stats(self):
        loc, freq, year, value = self.read_data()
        rates_arr = np.concatenate([loc,year,value], axis=1)
        self.rates = pd.DataFrame(rates_arr, columns=['loc','year','value']).sort_values('year')
        self.rates.value = self.rates.value.astype(float)
        self.rates.loc[:,'pct_ch'] = self.rates.groupby('loc').value.pct_change()
        lower = self.rates.pct_ch.min()
        upper = self.rates.pct_ch.max()
        country_low = self.rates.query(f'pct_ch=={lower}')['loc'].values
        country_high = self.rates.query(f'pct_ch=={upper}')['loc'].values
        print(f'Country with lowest rate pct change is: {country_low}')
        print(f'Country with highest rate pct change is: {country_high}')
        print(f'Exchange rate (0, 0.25, 0.5, 0.75, 1) quantiles are: {np.quantile(value, [0,0.25,0.5,0.75,1])}')
        print(f'Min pct of exchange rate over counties: {lower}')
        print(f'Max pct of exchange rate over counties: {upper}')

    def prepare_ts(self):
        return self.rates.pivot_table(index='year',columns='loc',values='value')


