import numpy as np
import matplotlib.pyplot as plt
import os


# This file is used to generate a synthetic data for a line
m, c = 0.27, 2.7
yerr = 0.03

# Function to generate synthetic data
def line(x, m, c):
    return (m*x) + c

x = np.sort(np.random.rand(20))
y = np.array([np.random.normal(line(x[i], m, c), yerr) for i in range(20)])
ye = np.abs(np.random.normal(0., yerr, 20))

f1 = open(os.getcwd() + '/Data/line.dat', 'w')
f1.write('#y = mx + c for m = ' + str(m) + ' and c = ' + str(c) + '\n')
for i in range(20):
    f1.write(str(x[i]) + '\t' + str(y[i]) + '\t' + str(ye[i]) + '\n')
f1.close()

plt.errorbar(x, y, yerr=ye, fmt='.')
plt.plot(x, line(x, m, c), 'k-')
plt.show()