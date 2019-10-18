import numpy as np
import matplotlib.pyplot as plt
import glob

file_list = glob.glob('./log/*-TrialDurations-qlearning.txt')

def read_durations(path):
    data = []
    
    with open(path, 'r') as infile:
        for line in infile:
            data.append(float(line))
    
    return np.array(data).reshape((40))

all_data = []

for file in file_list:
    all_data.append(read_durations(file))

all_data = np.array(all_data)

first = all_data[:, :10]
last = all_data[:, -10:]


data = first
print(np.median(data))
print(np.percentile(data, 25))
print(np.percentile(data, 75))


data = last
print(np.median(data))
print(np.percentile(data, 25))
print(np.percentile(data, 75))