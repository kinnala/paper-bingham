import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


legends = []

fname = "aposteriori_circle_p3_p1dg_uniform.csv"
data = genfromtxt(fname, delimiter=',')


data[:, 0] = np.sqrt(data[:, 0])

h1 = np.sqrt(data[:, 2] ** 2 + data[:, 1] ** 2)
#plt.loglog(data[:, 0], data[:, 1], 'o-')
plt.loglog(data[:, 0], h1, 'o-')
plt.loglog(data[:, 0], data[:, 3], 'o-')
legends = legends + [
    "$|| u - u_h ||_1$, order={:.1f}".format(np.polyfit(np.log(data[:, 0]), np.log(h1), 1)[0]),
#    "$|u - u_h|_1$, order={:.1f}".format(np.polyfit(np.log(data[:, 0]), np.log(data[:, 2]), 1)[0]),
    "$|| \mathrm{{div}}(\mathbf{{\lambda}} - \mathbf{{\lambda}}_h) ||_{{-1,h}}$, order={:.1f}".format(np.polyfit(np.log(data[:, 0]), np.log(data[:, 3]), 1)[0]),
]

#fname = "aposteriori_circle_p2b_p1dg_adaptive.csv"
fname = "aposteriori_circle_p3_p1dg_adaptive.csv"
data = genfromtxt(fname, delimiter=',')


data[:, 0] = np.sqrt(data[:, 0])

h1 = np.sqrt(data[:, 2] ** 2 + data[:, 1] ** 2)
plt.title("P3P1")
#plt.loglog(data[:, 0], data[:, 1], 's-')
plt.loglog(data[:, 0], h1, 's-')
plt.loglog(data[:, 0], data[:, 3], 's-')
legends = legends + [
    "$|| u - u_h ||_1$, order={:.1f}".format(np.polyfit(np.log(data[:, 0]), np.log(h1), 1)[0]),
#    "$|u - u_h|_1$, order={:.1f}".format(np.polyfit(np.log(data[:, 0]), np.log(data[:, 2]), 1)[0]),
    "$|| \mathrm{{div}}(\mathbf{{\lambda}} - \mathbf{{\lambda}}_h) ||_{{-1,h}}$, order={:.1f}".format(np.polyfit(np.log(data[:, 0]), np.log(data[:, 3]), 1)[0]),
]

plt.loglog(data[:, 0], 100. / data[:, 0] ** 2, 'k:')
legends += ['quadratic order']

plt.legend(legends)

plt.xlabel('$\sqrt{N}$')
plt.ylabel('error')
plt.grid('both')

#plt.show()
plt.savefig(fname + '.pdf')
