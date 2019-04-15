import numpy as np
import pandas as pd

if __name__ == '__main__':
    data_all = pd.read_csv("worldcities.csv").values
    countries = data_all[:, 3]
    sorting_indices = np.argsort(countries)

    countries_sorted = data_all[sorting_indices, :]
    print()








