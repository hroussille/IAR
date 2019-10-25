import numpy as np

data = []

with open('log/1571383114.5987835-TrialDurations-qlearning.txt', 'r') as infile:
    for line in infile:
        data.append(float(line))
        
print(np.median(data))
print(np.percentile(data, 25))
print(np.percentile(data, 75))
