import numpy as np
import matplotlib as mpl
mpl.use("TkAgg")
import matplotlib.pyplot as plt

x = np.array([1, 2])
y = np.array([0, 1])

plt.figure()
plt.plot(x, y)
plt.show()
plt.savefig("easyplot.png")
