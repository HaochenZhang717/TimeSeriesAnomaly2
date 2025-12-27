import matplotlib.pyplot as plt
import wfdb
import os
import numpy as np
from scipy.signal import decimate
import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path
from itertools import permutations
from sklearn.preprocessing import MinMaxScaler



import matplotlib.pyplot as plt
import numpy as np



def visualize_general():
    names = ["106"]
    for name in names:
        print('-' * 100)
        print(name)
        raw_data = np.load(f"./raw_data/{name}.npz")
        raw_signal = raw_data["signal"]
        scaler = MinMaxScaler()
        raw_signal = scaler.fit_transform(raw_signal)
        plt.plot(raw_signal)
        plt.show()
        # plot_12lead_ecg(
        #     signal=raw_signal,
        #     anomaly_label=None,
        #     title=name
        # )

if __name__ == '__main__':
    visualize_general()
