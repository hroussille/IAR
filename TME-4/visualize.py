import maze_plot
import sys
import glob
import random

print(sys.argv)
if len(sys.argv) < 3:
    exit()

path = sys.argv[1]
title = sys.argv[2]

files =  glob.glob(path + '/*traj*')

samples = random.sample(files, k=3)

for sample in samples:
    maze_plot.plot_traj_file(sample, title=title)
