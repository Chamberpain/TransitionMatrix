from __future__ import print_function
import numpy as np
import cartopy.crs as ccrs
from GeneralUtilities.Compute.list import find_nearest
import cartopy.feature as cfeature
import matplotlib.pyplot as plt


def cartopy_setup(lat_grid,lon_grid,traj_type,fill_color=True,ax=False):
	XX,YY = np.meshgrid(lon_grid,lat_grid)
	if not ax:
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
	if traj_type == 'SOSE':
		print('I am plotting antarctic region')
		llcrnrlon=-180.
		llcrnrlat=-80.
		urcrnrlon=180.
		urcrnrlat=-25
		ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())

	elif traj_type == 'Crete':
		print('I am plotting Crete')
		lon_0 = 0
		llcrnrlon=20.
		llcrnrlat=30
		urcrnrlon=30
		urcrnrlat=40
		ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())
	elif traj_type == 'Moby':
		print('I am plotting Moby')
		center_lat = 20.8
		center_lon = -157.2
		llcrnrlon=(center_lon-3)
		llcrnrlat=(center_lat-3)
		urcrnrlon=(center_lon+3)
		urcrnrlat=(center_lat+3)
		ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())
		ax.scatter(center_lon,center_lat,500,marker='*',color='Red',zorder=10)
	else:
		print('I am plotting global region')
		lon_0 = 0
		llcrnrlon=-180.
		llcrnrlat=-80.
		urcrnrlon=180.
		urcrnrlat=80
		ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())
	ax.add_feature(cfeature.LAND)
	ax.add_feature(cfeature.COASTLINE)
	ax.set_aspect('auto')
	gl = ax.gridlines(draw_labels=True)
	gl.xlabels_top = False
	gl.ylabels_right = False
	return XX,YY,ax,fig

def transition_vector_to_plottable(lat_grid,lon_grid,index_list,vector):
	plottable = np.zeros([len(lon_grid),len(lat_grid)])
	plottable = np.ma.masked_equal(plottable,0)
	for n,tup in enumerate(index_list):
		ii_index = lon_grid.index(tup[0])
		qq_index = lat_grid.index(tup[1])
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
