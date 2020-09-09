import numpy as np 
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def lats_lons():
	lat_grid = np.arange(-90,91,2)
	lats = np.load('lat_list.npy')
	lat_truth_list = np.array([x in lat_grid for x in lats])

	lon_grid = np.arange(-180,180,2)
	lons = np.load('lon_list.npy')
	lons[lons<-180]=lons[lons<-180]+360
	lon_truth_list = np.array([x in lon_grid for x in lons])

	truth_list = lat_truth_list&lon_truth_list
	lats = lats[truth_list]
	lons = lons[truth_list]
	return (lats,lons,truth_list)

lats,lons,truth_list = lats_lons()
file_list = ['pco2_surf_idxs.npy','o2_surf_idxs.npy','o2_100m_idxs.npy','dic_surf_idxs.npy','dic_100m_idxs.npy']
for file_ in file_list:
	array = np.load(file_)
	x,y = zip(*np.array(zip(lons,lats))[array])
	plt.scatter(x,y)
	plt.savefig(file_+'.png')