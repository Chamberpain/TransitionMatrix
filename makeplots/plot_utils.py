from __future__ import print_function
import numpy as np
from plot_utilities.eulerian_plot import Basemap
from compute_utilities.list_utilities import find_nearest

def basemap_setup(lat_grid,lon_grid,traj_type,fill_color=True):
	X,Y = np.meshgrid(lon_grid,lat_grid)
	if traj_type == 'SOSE':
		print('I am plotting antarctic region')
		lon_0 = 0
		llcrnrlon=-180.
		llcrnrlat=-80.
		urcrnrlon=180.
		urcrnrlat=-25
		m = Basemap.auto_map(urcrnrlat,llcrnrlat,urcrnrlon,llcrnrlon,lon_0,aspect=True)
	elif traj_type == 'Crete':
		print('I am plotting Crete')
		lon_0 = 0
		llcrnrlon=20.
		llcrnrlat=30
		urcrnrlon=30
		urcrnrlat=40
		m = Basemap.auto_map(urcrnrlat,llcrnrlat,urcrnrlon,llcrnrlon,lon_0,aspect=False,resolution='h')
	elif traj_type == 'Moby':
		print('I am plotting Moby')
		lon_0 = 0
		center_lat = 20.8
		center_lon = -157.2
		llcrnrlon=(center_lon-2)
		llcrnrlat=(center_lat-2)
		urcrnrlon=(center_lon+2)
		urcrnrlat=(center_lat+2)
		m = Basemap.auto_map(urcrnrlat,llcrnrlat,urcrnrlon,llcrnrlon,lon_0,aspect=False,resolution='h')
		m.scatter(center_lon,center_lat,500,marker='*',color='Red',latlon=True,zorder=10)
	else:
		print('I am plotting global region')
		lon_0 = 0
		llcrnrlon=-180.
		llcrnrlat=-80.
		urcrnrlon=180.
		urcrnrlat=80
		m = Basemap.auto_map(urcrnrlat,llcrnrlat,urcrnrlon,llcrnrlon,lon_0,aspect=False)

	XX,YY = m(X,Y)
	return XX,YY,m

def transition_vector_to_plottable(lat_grid,lon_grid,index_list,vector):
	plottable = np.zeros([len(lon_grid),len(lat_grid)])
	plottable = np.ma.masked_equal(plottable,0)
	for n,tup in enumerate(index_list):
		ii_index = lon_grid.index(tup[1])
		qq_index = lat_grid.index(tup[0])
		plottable[ii_index,qq_index] = vector[n]
	return plottable.T

def plottable_to_transition_vector(lat_grid,lon_grid,index_list,plottable):
	vector = np.zeros([len(index_list)])
	for n,(lat,lon) in enumerate(index_list):
		lon_index = lon_grid.index(find_nearest(lon_grid,lon))
		assert abs(lon_grid[lon_index]-lon)<2
		lat_index = lat_grid.index(find_nearest(lat_grid,lat))
		assert abs(lat_grid[lat_index]-lat)<2
		vector[n] = plottable[lat_index,lon_index]
	return vector
