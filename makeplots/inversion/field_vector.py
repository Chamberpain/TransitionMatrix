import pandas as pd
import numpy as np
from plot_utils import basemap_setup,transition_vector_to_plottable,plottable_to_transition_vector
import matplotlib.pyplot as plt
import os, sys
# get an absolute path to the directory that contains mypackage
try:
	make_plot_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
	make_plot_dir = os.path.dirname(os.path.join(os.getcwd(),'dummy'))
sys.path.append(os.path.normpath(os.path.join(make_plot_dir, '../..compute/')))
from netCDF4 import Dataset
from compute_utils import find_nearest
from inverse_plot import InverseBase

def gradient_calc(data):
		XX,YY = np.gradient(data)
		return XX+YY

class TargetVector(InverseBase):
	def __init__(self,**kwds):
		super(TargetVector,self).__init__(**kwds)

	def load_vector(self,load_file):
		return np.load(load_file+'.npy')

	def plot(self,XX,YY,m=False):
		plt.figure()
		plot_vector = abs(transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.index_list,self.vector))
		if not m:
			XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type)
		m.pcolormesh(XX,YY,np.ma.masked_equal(plot_vector,0),cmap=self.cm)
		plt.colorbar(label=self.label)
		plt.title(self.title)
		return m

class LandschutzerCO2Flux(TargetVector):
	def __init__(self,var=False,**kwds):
		super(LandschutzerCO2Flux,self).__init__(**kwds)
		self.cm = plt.cm.PRGn

		self.label = 'CO2 Flux $gm C/m^2/yr$'
		if var:
			file_path = './data/cm2p6_vector_time_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		else:
			file_path = './data/cm2p6_vector_space_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		try:
			self.vector = self.load_vector(file_path)
		except IOError:
			print 'landchutzer target vector file not found, recompiling'
			self.vector = self.compile_vector(var,file_path)

	def compile_vector(self,var,save_file):

		file_ = self.base_file+'../spco2_MPI_SOM-FFN_v2018.nc'
		nc_fid = Dataset(file_)
		y = nc_fid['lat'][:]
		x = nc_fid['lon'][:]
		
		data = np.ma.masked_greater(nc_fid['fgco2_smoothed'][:],10**19) 
		if var:
			data = np.nanvar(data,axis=0)
		else:
			XX,YY = np.gradient(np.nanmean(data,axis=0)) # take the time mean, then take the gradient of the 2d array
			data = np.abs(XX+YY)
		x[0] = -180
		x[-1] = 180
		vector = plottable_to_transition_vector(y,x,self.list,data)
		np.save(save_file,vector)
		return vector

class MODISVector(TargetVector):
	def __init__(self,var='space',**kwds):
		super(MODISVector,self).__init__(**kwds)
		self.cm = plt.cm.PRGn
		if var=='time':
			self.label = '*** MODIS TIME VARIANCE ***'
		else:
			self.label = '*** MODIS SPACE VARIANCE ***'			
		file_path = './data/modis_vector_'+str(var)+'_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		try:
			self.vector = self.load_vector(file_path)
		except IOError:
			print 'modis file not found, recompiling'
			self.vector = self.compile_vector(var,file_path)

	def compile_vector(self,var,save_file):
		file_ = self.base_file+'../MODIS/'
		datalist = []
		for _ in os.listdir(file_):
			if _ == '.DS_Store':
				continue
			nc_fid = Dataset(file_ + _)
			datalist.append(nc_fid.variables['chlor_a'][::12,::12])
		y = nc_fid.variables['lat'][::12]
		x = nc_fid.variables['lon'][::12]
		dat = np.ma.stack(datalist)

		if var=='time':
			data = np.ma.var(dat,axis=0)
		else:		
			dummy = np.ma.mean(dat,axis=0)
			data = gradient_calc(dummy)

		vector = plottable_to_transition_vector(self.bins_lat,self.bins_lon,self.list,data)
		np.save(save_file,vector)
		return vector

class GlodapVector(TargetVector):
	def __init__(self,flux,**kwds):
		super(GlodapVector,self).__init__(**kwds)
		file_path = './data/glodap_vector_'+str(flux)+'_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		try:
			self.vector = self.load_vector(file_path)
		except IOError:
			print 'glodap file not found, recompiling'
			self.vector = self.compile_vector(flux,file_path)

	def compile_vector(self,flux,save_file):
		file_ = self.base_file+'../GLODAP/'
		for _ in os.listdir(file_):
			variable = _.split('.')[-2]
			if variable in [flux]:
				nc_fid = Dataset(file_ + _)
				data = nc_fid[variable][0,:,:]
				data = gradient_calc(data)
		y = nc_fid['lat'][:]
		x = nc_fid['lon'][:]
		x[x>180] = x[x>180]-360
		x[159] = 180
		x[160] = -180
		datalist = [data[:,_] for _ in range(data.shape[1])]
		sorted_data = [_ for __,_ in sorted(zip(x,datalist))]
		sorted_data = np.ma.stack(sorted_data).T
		sorted_x = sorted(x)
		print 'I am working on ',flux
		vector = plottable_to_transition_vector(y,sorted_x,self.list,sorted_data)
		vector[np.where(np.isnan(vector))] = max(vector)
		assert ~np.isnan(vector).any()
		np.save(save_file,vector)
		return vector

class CM2p6Vector(TargetVector):
	def __init__(self,variable,variance='time',**kwds):
		super(CM2p6Vector,self).__init__(**kwds)
		file_path = './data/cm2p6_vector_'+str(variable)+'_'+str(variance)+'_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		try:
			self.vector = self.load_vector(file_path)
		except IOError:
			print 'cm2p6 file not found, recompiling'
			self.vector = self.compile_vector(variable,variance,file_path)

	def compile_vector(self,variable,variance,save_file):	
		self.lat_list = np.load(self.base_file+'../lat_list.npy')
		lon_list = np.load(self.base_file+'../lon_list.npy')
		lon_list[lon_list<-180]=lon_list[lon_list<-180]+360
		self.lon_list = lon_list
		self.translation_list = []
		for x in self.list:
			mask = (self.lat_list==x[0])&(self.lon_list==x[1])
			if not mask.any():
				for lat in np.arange(x[0]-self.degree_bins[0],x[0]+self.degree_bins[0],0.5):
					for lon in np.arange(x[1]-self.degree_bins[1],x[1]+self.degree_bins[1],0.5):
						mask = (self.lat_list==lat)&(self.lon_list==lon)
						if mask.any():
							t = np.where(mask)
							break
					if mask.any():
						break
			else:
				t = np.where(mask)
			assert t[0]
			assert len(t[0])==1
			self.translation_list.append(t[0][0])
		if variable == 'o2':
			data = np.load(self.base_file+'../subsampled_o2.npy') 
		elif variable == 'pco2':
			data = np.load(self.base_file+'../subsampled_pco2.npy') 
		elif variable == 'hybrid':
			data_o2 = np.load(self.base_file+'../subsampled_o2.npy') 
			data_pco2 = np.load(self.base_file+'../subsampled_pco2.npy') 
			data = (data_o2+data_pco2)/2
		else:
			raise
		if variance=='time':
			vector = data.var(axis=0)[self.translation_list]
		else: 
			data = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.list,data.mean(axis=0)[self.translation_list])
			data = gradient_calc(data)
			vector = plottable_to_transition_vector(self.bins_lat,self.bins_lon,self.list,data)
		return vector