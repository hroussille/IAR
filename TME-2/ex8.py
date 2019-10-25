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


def smooth(x,window_len=8,window='hanning'):

    s=np.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_len,'d')
    else:
        w=eval('np.'+window+'(window_len)')

    y=np.convolve(w/w.sum(),s,mode='valid')
    return y



def smooth_demo():
    t=linspace(40)
    x=sin(t)
    xn=x+randn(len(t))*0.1
    y=smooth(x)


all_data = []

for file in file_list:
    all_data.append(read_durations(file))
    
all_data = np.array(all_data)
averaged = np.mean(all_data, axis=0)

print(averaged)


t=np.linspace(0, 40, 0.5)
x=np.sin(t)
xn=x+np.random.randn(len(t))*0.1
y=smooth(averaged)

print(y)

plt.plot(y)
plt.show()

