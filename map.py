import numpy as np
import healpy as hp
import matplotlib.pyplot as plt

npix = 12 * 1024 ** 2
i = [1,2,3,4,5,6,7] 

temp = np.zeros(npix)
occurance = np.zeros(npix)
avg_temp = np.zeros(npix)

# Load data from each file and store the fourth column in the list
for idx in i:
    filename = f"map{idx}.dat"
    print(idx)
    map_data = np.loadtxt(filename)
    temp += map_data[:, 1]
    occurance += map_data[:, 2]

for i in range(npix):
    if occurance[i] != 0:
        avg_temp[i] = 1000 * temp[i] / occurance[i]
    # else:
    #     avg_temp[i] = 10

map = avg_temp
hp.mollview(map)
plt.savefig("image.png")
