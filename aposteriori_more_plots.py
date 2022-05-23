import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


legends = []

fname = "aposteriori_more_squarep3_p1dg_adaptive.csv"
data = genfromtxt(fname, delimiter=',')


data[:, 0] = np.sqrt(data[:, 0])
#data = data[:-2]

plt.figure(figsize=(5.4, 3), tight_layout=True)
plt.loglog(data[:, 0], data[:, 1], 'o-')
legends = legends + [
    "error estimator",
]

plt.loglog(data[:, 0], 800. / data[:, 0] ** 2, 'k:')
legends += ['quadratic order']

plt.legend(legends)

plt.xlabel('$\sqrt{N}$')
plt.ylabel('$\eta$')
plt.grid('both')
plt.savefig(fname + '.pdf')
#plt.show()
