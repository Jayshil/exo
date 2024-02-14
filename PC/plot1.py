import numpy as np
import matplotlib.pyplot as plt

# Number of points (common for both samplers)
nos_pts = np.array([4552, 8915])

# Time taken for convergence (for dynesty)
tdyn = np.array([29.0479, 103.5327]) / 60.

# Time taken for 15000 iterations (for numpyro)
tnum15 = np.array([168.8001, 367.2480]) / 60.
tnum20 = np.array([216.1793, 473.5695]) / 60.

# Time taken by nautilus
tnat = np.array([8.8873, 23.6903]) / 60.

plt.figure(figsize=(16/1.5, 9/1.5))
plt.plot(nos_pts, tdyn, color='orangered', lw=1.5)
plt.plot(nos_pts, tnat, color='darkgreen', lw=1.5)
plt.plot(nos_pts, tnum15, color='cornflowerblue', lw=1.5, ls='-')
plt.plot(nos_pts, tnum20, color='cornflowerblue', lw=1.5, ls='--')
plt.show()

print('>>> --- Rate for Dynesty is: {:.4f} hour/point'.format((tdyn[1] - tdyn[0])/(nos_pts[1] - nos_pts[0])))
print('>>> --- Rate for Nautilus is: {:.4f} hour/point'.format((tnat[1] - tnat[0])/(nos_pts[1] - nos_pts[0])))
print('>>> --- Rate for numpyro (15k iter) is: {:.4f} hour/point'.format((tnum15[1] - tnum15[0])/(nos_pts[1] - nos_pts[0])))
print('>>> --- Rate for numpyro (20k iter) is: {:.4f} hour/point'.format((tnum20[1] - tnum20[0])/(nos_pts[1] - nos_pts[0])))