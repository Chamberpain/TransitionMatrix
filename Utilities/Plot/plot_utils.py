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
	elif traj_type == 'GOM':
		print('I am plotting GOM')
		lon_0 = 0
		llcrnrlon=-100.
		llcrnrlat=20.5
		urcrnrlon=-81.5
		urcrnrlat=30.5
		ax.set_extent([llcrnrlon,urcrnrlon,llcrnrlat,urcrnrlat], crs=ccrs.PlateCarree())		
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
	return (XX,YY,ax,fig)


