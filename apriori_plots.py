import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt

# plt.rcParams.update({
#   "text.usetex": True,
# })


def conv(fname, title):
    try:
        data = genfromtxt(fname, delimiter=',')
    except Exception:
        print("Run first all cases in apriori.py.")
        return

    plt.figure(figsize=(5.4, 3), tight_layout=True)
    plt.title(title)
    h = data[:, 0]
    plt.loglog(h, data[:, 1], 'o-')
    plt.loglog(h, data[:, 2], 'o-')
    plt.loglog(h, data[:, 3], 'o-')
    #plt.loglog(h, h * np.sqrt(np.log(1. / h)), 'k-')
    plt.legend([
        "$|| u - u_h ||_0$, order={:.1f}".format(np.polyfit(np.log(data[:, 0]), np.log(data[:, 1]), 1)[0]),
        "$|u - u_h|_1$, order={:.1f}".format(np.polyfit(np.log(data[:, 0]), np.log(data[:, 2]), 1)[0]),
        "$|| \mathrm{{div}}(\mathbf{{\lambda}} - \mathbf{{\lambda}}_h) ||_{{-1,h}}$, order={:.1f}".format(np.polyfit(np.log(data[:, 0]), np.log(data[:, 3]), 1)[0]),
    ])
    plt.xlabel('$h$')
    plt.ylabel('error')
    plt.grid('both')

conv("apriori_circle_p1b_p1.csv", "MINI")
conv("apriori_circle_p2_p0.csv", "P2P0")
conv("apriori_circle_p3_p1dg.csv", "P3P1")
conv("apriori_circle_p1_p0.csv", "P1P0")
plt.show()
