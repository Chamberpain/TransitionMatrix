from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from transition_matrix_compute import transition_class_loader,find_nearest
from scipy.interpolate import griddata
import pandas as pd
import pyproj
from matplotlib.patches import Polygon
from itertools import groupby
import pickle
import scipy
import matplotlib.colors as colors
import pickle
import os
import datetime
import matplotlib.cm as cm
from sets import Set
import matplotlib.colors as mcolors         
from netCDF4 import Dataset
from landschutzer_plot import base_inversion_plot


class modis_plot(base_inversion_plot):
	def __init__(self,**kwds):
		super(modis_plot,self).__init__(**kwds)

		file_ = self.info.base_file+'../../Raw/MODIS/'
		datalist = []
		for _ in os.listdir(file_):
			if _ == '.DS_Store':
				continue
			nc_fid = Dataset(file_ + _)
			datalist.append(nc_fid.variables['chlor_a'][::12,::12])
		y = nc_fid.variables['lat'][::12]
		x = nc_fid.variables['lon'][::12]
		dat = np.ma.stack(datalist)
		temporal_var = np.ma.var(dat,axis=0)
		dummy = np.ma.mean(dat,axis=0)
		X,Y = np.gradient(dummy)
		spatial_gradient = X+Y
		self.spatial_gradient = np.zeros([len(self.transition.list),1])
		self.temporal_variance = np.zeros([len(self.transition.list),1])
		for n,(lat,lon) in enumerate(self.transition.list):
			lon_index = x.tolist().index(find_nearest(x,lon))
			lat_index = y.tolist().index(find_nearest(y,lat))
			self.spatial_gradient[n] = spatial_gradient[lat_index,lon_index]
			self.temporal_variance[n] = temporal_var[lat_index,lon_index]
		self.matrix.get_direction_matrix()
		self.corr = np.exp(-(traj_class.matrix.east_west**2)/(14/2)-(traj_class.matrix.north_south**2)/(7/2)) # glodap uses a 7 degree zonal correlation and 14 degree longitudinal correlation
		self.corr[self.corr<0.3] = 0
		self.corr = scipy.sparse.csc_matrix(self.corr)

	def variable_plot(self):
		for name,plotting_vector in [('Log of Spatial Gradients',self.spatial_gradient),('Log of Temporal Variance',self.temporal_variance)]:
			plt.figure()
			chl_plot = abs(self.matrix.transition_vector_to_plottable(plotting_vector))
			m = Basemap(projection='cyl',fix_aspect=False)
			# m.fillcontinents(color='coral',lake_color='aqua')
			m.drawcoastlines()
			XX,YY = m(self.info.X,self.info.Y)
			m.pcolormesh(XX,YY,np.log(np.ma.masked_equal(chl_plot,0)),cmap=plt.cm.PRGn)
			plt.title(name)
		plt.show()

	def inverse_plot(self):
		for title,field_vector in [('Chlorophyl A Spatial Gradients',self.spatial_gradient),('Cholorophyl A Temporal Variance',self.temporal_variance)]:

			field_plot = abs(self.matrix.transition_vector_to_plottable(field_vector))
			x,y,desired_vector =  self.get_optimal_float_locations(field_vector,self.corr,float_number=500)
			plt.figure()
			k = Basemap(projection='cea',llcrnrlon=-180.,llcrnrlat=-80.,urcrnrlon=180.,urcrnrlat=80,fix_aspect=False)
			k.drawcoastlines()
			XX,YY = k(self.info.X,self.info.Y)
			k.fillcontinents(color='coral',lake_color='aqua')
			k.pcolormesh(XX,YY,np.ma.masked_equal(field_plot,0),cmap=plt.cm.PRGn)
			plt.colorbar()
			plt.title(title.capitalize())
			k.scatter(x,y,marker='*',color='g',s=34,latlon=True)
			k = self.plot_latest_soccom_locations(k)		
			plt.savefig('-'.join(title.split(' ')))
			plt.close()