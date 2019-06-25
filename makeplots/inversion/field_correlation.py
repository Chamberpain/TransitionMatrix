import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, sys
# get an absolute path to the directory that contains mypackage
try:
	make_plot_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
	make_plot_dir = os.path.dirname(os.path.join(os.getcwd(),'dummy'))
sys.path.append(os.path.normpath(os.path.join(make_plot_dir, '../')))
sys.path.append(os.path.normpath(os.path.join(make_plot_dir, '../../compute/')))
from plot_utils import basemap_setup,transition_vector_to_plottable,plottable_to_transition_vector
from netCDF4 import Dataset
from compute_utils import find_nearest
from inverse_plot import InverseBase
from sets import Set
import scipy 
from compute_utils import save_sparse_csc, load_sparse_csc


class TargetCorrelation(InverseBase):
	def __init__(self,**kwds):
		super(TargetCorrelation,self).__init__(**kwds)

	def plot(self):
		plt.figure()
#this needs work
		plt.show()	

	def load_corr(self,load_file):
		return load_sparse_csc(load_file+'.npz')

	def save_corr(self,data_list,col_list,row_list,save_file):
		assert (np.array(data_list)<=1).all()
		sparse_matrix = scipy.sparse.csc_matrix((np.abs(data_list),(row_list,col_list)),shape=[len(self.list),len(self.list)])
		save_sparse_csc(save_file,sparse_matrix)
		return sparse_matrix

class CM2p6Correlation(TargetCorrelation):
	def __init__(self,variable,**kwds):
		super(CM2p6Correlation,self).__init__(**kwds)
		self.cm = plt.cm.PRGn
		file_path = './data/cm2p6_corr_'+str(variable)+'_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		try:
			self.corr = self.load_corr(file_path)
		except IOError:
			print file_path
			print 'cm2p6 file not found, recompiling'
			self.corr = self.compile_corr(variable,file_path)

	def compile_corr(self,variable,file_path):
		self.lat_list = np.load(self.base_file+'../lat_list.npy')
		lon_list = np.load(self.base_file+'../lon_list.npy')
		lon_list[lon_list<-180]=lon_list[lon_list<-180]+360
		self.lon_list = lon_list
		self.corr_translation_list = []
		mask = (self.lon_list%1==0)&(self.lat_list%1==0)
		lats = self.lat_list[mask]
		lons = self.lon_list[mask]
		# grab only the whole degree values
		for x in self.list:
			mask = (lats==x[0])&(lons==x[1])      
			if not mask.any():
				for lat in np.arange(x[0]-self.degree_bins[0],x[0]+self.degree_bins[0],0.5):
					for lon in np.arange(x[1]-self.degree_bins[1],x[1]+self.degree_bins[1],0.5):
						mask = (lats==lat)&(lons==lon)
						if mask.any():
							t = np.where(mask)
							break
					if mask.any():
						break
			else:
				t = np.where(mask)
			assert t[0]
			assert len(t[0])==1
			self.corr_translation_list.append(t[0][0])
		if variable == 'o2':
			data = np.load(self.base_file+'../o2_corr.npy')
		elif variable == 'pco2':
			data = np.load(self.base_file+'../pco2_corr.npy')
		elif variable == 'hybrid':
			data = (np.load(self.base_file+'../o2_corr.npy')+np.load(self.base_file+'../pco2_corr.npy'))/2

		total_set = Set([tuple(x) for x in self.list])
		row_list = []
		col_list = []
		data_list = []
		mask = (self.lon_list%1==0)&(self.lat_list%1==0) # we only calculated correlations at whole degrees because of memory
		lats = self.lat_list[mask][self.corr_translation_list] #mask out non whole degrees and only grab at locations that match to the self.transition.index
		lons = self.lon_list[mask][self.corr_translation_list] 
		corr_list = data[self.corr_translation_list]

		# test_Y,test_X = zip(*self.transition.list)
		for k,(base_lat,base_lon,corr) in enumerate(zip(lats,lons,corr_list)):
			if k % 100 ==0:
				print str(k/float(len(self.list)))+' done'
			lat_index_list = np.arange(base_lat-12,base_lat+12.1,0.5)
			lon_index_list = np.arange(base_lon-12,base_lon+12.1,0.5)
			Y,X = np.meshgrid(lat_index_list,lon_index_list) #we construct in this way to match how the correlation matrix was made
			test_set = Set(zip(Y.flatten(),X.flatten()))
			intersection_set = total_set.intersection(test_set)
			location_idx = [self.list.index(list(_)) for _ in intersection_set]
			data_idx = [zip(Y.flatten(),X.flatten()).index(_) for _ in intersection_set]

			data = corr.flatten()[data_idx]
			assert len(location_idx)==len(data)
			row_list += location_idx
			col_list += [k]*len(location_idx)
			data_list += data.tolist()
		corr = self.save_corr(data_list,col_list,row_list,file_path)
		return corr

class GlodapCorrelation(TargetCorrelation):
	def __init__(self,**kwds):
		super(GlodapCorrelation,self).__init__(**kwds)
		self.get_direction_matrix()
		self.corr = np.exp(-(self.east_west**2)/(14/2)-(self.north_south**2)/(7/2)) # glodap uses a 7 degree zonal correlation and 14 degree longitudinal correlation
		self.corr[self.corr<0.3] = 0
		self.corr = scipy.sparse.csc_matrix(self.corr)

