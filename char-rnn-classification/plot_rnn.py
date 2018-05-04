import matplotlib.pyplot as plt
import numpy as np

dir = [
    'results/lr-0.005-optim-sgd-gpu-False-nhidden-128-bsize-1-seed-0',
    'results/lr-0.005-optim-sgd-gpu-True-nhidden-128-bsize-1-seed-0',
]

thisdir = dir[0]
figdir = thisdir+'/losses.png'
all_losses = np.loadtxt(thisdir+'/losses.txt')

plt.figure()
plt.plot(all_losses)
plt.savefig(figdir)
