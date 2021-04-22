import os, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
sys.path.append(os.path.join(os.getenv("HOME"),'Projects'))
import transition_matrix
from transition_matrix.makeplots import plot_utils


def data_reshape(grid_lat,grid_lon,data):
	X,Y = np.meshgrid(grid_lon,grid_lat)
	data_holder = np.zeros(X.shape)
	for n,(lat,lon) in enumerate(zip(lat_list,lon_list)):
		lon_grid_index = grid_lon.index(lon)
		lat_grid_index = grid_lat.index(lat)
		data_holder[lat_grid_index,lon_grid_index]=data[n]
	return (X,Y,data_holder)


lat_list = np.load('lat_list.npy')
lon_list = np.load('lon_list.npy')
lon_list[lon_list<-180]=lon_list[lon_list<-180]+360
lon_list = lon_list
grid_lon = np.arange(-180,180,.5).tolist()
grid_lat = np.arange(-78.5,90,.5).tolist()

for file_name,title,units,vmax,vmin,save_name,color in [('subsampled_o2_100m.npy',\
	'$O_2$ (Temporal Variance, 100m Integrated)',\
	'(mol m$^{-2})^2$',\
	3,
	0,
	'o2_hires_matt',
	cm.BrBG),
('subsampled_pco2_surf.npy',\
'$pCO_2$ (Temporal Variance, Surface)',\
	'$(\mu atm)^2$',\
	75,
	800,
	'pco2_hires_matt',
	cm.PiYG)]:
	data = np.load(file_name)
	data = data.var(axis=0)
	X,Y,data_holder = data_reshape(grid_lat,grid_lon,data)
	XX,YY,m = plot_utils.basemap_setup(lat_grid=grid_lat,lon_grid=grid_lon,traj_type='Argo')
	m.pcolor(XX,YY,data_holder,cmap=color,vmax=vmax,vmin=vmin)
	plt.title(title)
	plt.colorbar(label=units)
	plt.savefig(save_name)
	plt.close()