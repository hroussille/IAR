import matplotlib.pyplot as plt
import numpy as np
import glob

def get_pos(path, start=None, end=None):
    x = []
    y  = []
    data = np.load(path, allow_pickle=True)
    
    for trial in data[:10]:
        for pos in trial:
            x.append(pos.x())
            y.append(600 - pos.y())
        
    return x, y
            
pos_files = glob.glob('./log/*_pos.*')

x = []
y = []

for file in pos_files:
    _x, _y = get_pos(file, start=0, end=10)
    x += _x
    y += _y
    
plt.hist2d(x, y, bins=[12, 12], normed=True, cmap='plasma')
plt.show()