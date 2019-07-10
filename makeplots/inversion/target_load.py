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
from sets import Set
import scipy 
from compute_utils import save_sparse_csc, load_sparse_csc
from netCDF4 import Dataset
from compute_utils import find_nearest
from transition_matrix_plot import TransitionPlot
import matplotlib.cm as cm


class InverseBase(object):
	def __init__(self,transition_plot,**kwds):
		self.list = transition_plot.list
		self.base_file=transition_plot.base_file
		self.bins_lat = transition_plot.bins_lat
		self.bins_lon = transition_plot.bins_lon
		self.degree_bins = transition_plot.degree_bins
		self.quiver = transition_plot.quiver_plot
		self.traj_file_type = transition_plot.traj_file_type
		try:
			self.east_west = transition_plot.east_west
			self.north_south = transition_plot.north_south
			print 'I have succesfully loaded the east_west north_south matrices'
		except AttributeError:
			pass
	def gradient_calc(self,data):
			dy = 111.7*1000
			# bins_lat_holder = self.bins_lat
			# bins_lat_holder[0]=-89.5
			# bins_lat_holder[-1]=89.5
			# x = np.cos(np.deg2rad(bins_lat_holder))*dy #scaled meridional distance
			XX,YY = np.gradient(data,dy,dy)
			return XX+YY
	def data_return(self,variable):
		data = np.load(self.base_file+self.file_dictionary[variable]) 
		return data
####### target correlation ###########

class TargetCorrelation(InverseBase):
	def __init__(self,**kwds):
		super(TargetCorrelation,self).__init__(**kwds)

	def plot(self,m=False):
		if not m:
			m = self.quiver(trans_mat=self.matrix,arrows=False,scale_factor=self.scale_factor,degree_sep=8)
		else:
			self.quiver(trans_mat=self.matrix,arrows=False,scale_factor=self.scale_factor,degree_sep=8,m=m)
		return m

	def load_corr(self,load_file):
		return load_sparse_csc(load_file+'.npz')

	def save_corr(self,data_list,col_list,row_list,save_file):
		assert (np.array(data_list)<=1).all()
		sparse_matrix = scipy.sparse.csc_matrix((np.abs(data_list),(row_list,col_list)),shape=[len(self.list),len(self.list)])
		save_sparse_csc(save_file,sparse_matrix)
		return sparse_matrix

	def scale_direction(self):
		mask = np.array(self.matrix.todense()!=0)
		self.east_west = np.multiply(mask,self.east_west)
		self.north_south = np.multiply(mask,self.north_south)




class CM2p6Correlation(TargetCorrelation):
	def __init__(self,variable,**kwds):
		super(CM2p6Correlation,self).__init__(**kwds)
		self.scale_factor=.1
		self.file_dictionary = 	{'surf_o2':'../o2_surf_corr.npy','surf_dic':'../dic_surf_corr.npy',\
		'surf_pco2':'../pco2_surf_corr.npy','100m_dic':'../dic_100m_corr.npy','100m_o2':'../o2_100m_corr.npy'\
		}	

		corr_file_path = './data/cm2p6_corr_'+str(variable)+'_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		try:
			self.matrix = self.load_corr(corr_file_path)
		except IOError:

			print corr_file_path
			print 'cm2p6 file not found, recompiling'
			self.matrix = self.compile_corr(variable,corr_file_path)

		# self.scale_direction()

	def translation_list_construct(self):
		self.lat_list = np.load(self.base_file+'../lat_list.npy')
		lon_list = np.load(self.base_file+'../lon_list.npy')
		lon_list[lon_list<-180]=lon_list[lon_list<-180]+360
		self.lon_list = lon_list
		self.translation = []
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
			self.translation.append(t[0][0])


	def compile_corr(self,variable,file_path):
		translation_file_path = './data/cm2p6_corr_'+str(variable)+'_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		try:
			self.translation = np.load(translation_file_path)
		except IOError:
			print 'translation list not found, recompiling'			
			self.translation_list_construct()
			np.save(file_path,self.translation)		

		self.lat_list = np.load(self.base_file+'../lat_list.npy')
		lon_list = np.load(self.base_file+'../lon_list.npy')
		lon_list[lon_list<-180]=lon_list[lon_list<-180]+360
		self.lon_list = lon_list
		mask = (self.lon_list%1==0)&(self.lat_list%1==0)
		lats = self.lat_list[mask]
		lons = self.lon_list[mask]
		# grab only the whole degree values
		data = self.data_return(variable)
		total_set = Set([tuple(x) for x in self.list])
		row_list = []
		col_list = []
		data_list = []
		mask = (self.lon_list%1==0)&(self.lat_list%1==0) # we only calculated correlations at whole degrees because of memory
		lats = self.lat_list[mask][self.translation] #mask out non whole degrees and only grab at locations that match to the self.transition.index
		lons = self.lon_list[mask][self.translation] 
		corr_list = data[self.translation]

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
		self.matrix = np.exp(-(self.east_west**2)/(14/2)-(self.north_south**2)/(7/2)) # glodap uses a 7 degree zonal correlation and 14 degree longitudinal correlation
		self.matrix[self.matrix<0.3] = 0
		self.matrix = scipy.sparse.csc_matrix(self.matrix)
		self.scale_direction()

####### target vector #########

class TargetVector(InverseBase):
	def __init__(self,**kwds):
		super(TargetVector,self).__init__(**kwds)

	def load_vector(self,load_file):
		return np.load(load_file+'.npy')

	def plot(self,XX=False,YY=False,m=False):
		plot_vector = abs(transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.list,self.vector))
		print 'shape of plot vector is ',plot_vector.shape
		if not m:
			XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type)
		print 'shape of the coordinate matrix is ',XX.shape 
		m.pcolormesh(XX,YY,np.ma.masked_equal(plot_vector,0),cmap=self.cm,vmin=self.vmin,vmax=self.vmax)
		plt.colorbar(label=self.unit)
		plt.title(self.title)
		return (XX,YY,m)

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
			data = self.gradient_calc(dummy)

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
				data = self.gradient_calc(data)
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
	def __init__(self,**kwds):
		super(CM2p6Vector,self).__init__(**kwds)
		self.title = ''
		self.cm_dict = {'surf_o2':cm.BrBG,'surf_dic':cm.PRGn,\
		'surf_pco2':cm.PiYG,'100m_dic':cm.PRGn,'100m_o2':cm.BrBG\
		}	
		file_path = './data/cm2p6_vector_tranlation_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		self.unit_dict = {'surf_o2':'mol kg$^{-1}$','surf_dic':'mol m$^{-2}$ s$^{-1}$',\
		'surf_pco2':'$\mu$ atm','100m_dic':'mol m$^{-2}$','100m_o2':'mol m$^{-2}$'\
		}
		self.file_dictionary = 	{'surf_o2':'../subsampled_o2_surf.npy','surf_dic':'../subsampled_dic_surf.npy',\
		'surf_pco2':'../subsampled_pco2_surf.npy','100m_dic':'../subsampled_dic_100m.npy','100m_o2':'../subsampled_o2_100m.npy'\
		}	
		try:
			self.translation = np.load(file_path)
		except IOError:
			self.translation_list_construct()
			np.save(file_path,self.translation_list)

	def translation_list_construct(self):	
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

	def plot_setup(self,variable,plot_range_dict):
		self.vmin,self.vmax = plot_range_dict[variable]
		self.cm = self.cm_dict[variable]

class CM2p6VectorSpatialGradient(CM2p6Vector):	
	def __init__(self,variable,**kwds):
		super(CM2p6VectorSpatialGradient,self).__init__(**kwds)
		file_path = './data/cm2p6_vector_'+str(variable)+'_space_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		self.unit = self.unit_dict[variable]+' m$^{-1}$'
		try:
			self.vector = self.load_vector(file_path)
		except IOError:
			print 'cm2p6 file not found, recompiling'
			data = self.data_return(variable)
			data = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.list,data.mean(axis=0)[self.translation_list])
			data = self.gradient_calc(data)
			self.vector = plottable_to_transition_vector(self.bins_lat,self.bins_lon,self.list,data)
			np.save(file_path,self.vector)
		plot_range_dict = {'surf_o2':(0,10**-6),'surf_dic':(10**-6,10**-5),\
		'surf_pco2':(10**-4,10**-5),'100m_dic':(2*10**-5,5*10**-6),'100m_o2':(2*10**-5,10**-6)\
		}	
		self.plot_setup(variable,plot_range_dict)

class CM2p6VectorTemporalVariance(CM2p6Vector):	
	def __init__(self,variable,**kwds):
		super(CM2p6VectorTemporalVariance,self).__init__(**kwds)
		file_path = './data/cm2p6_vector_'+str(variable)+'_time_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
		self.unit = '('+self.unit_dict[variable]+')$^2$'
		try:
			self.vector = self.load_vector(file_path)
		except IOError:
			print 'cm2p6 file not found, recompiling'
			data = self.data_return(variable)
			self.vector = data.var(axis=0)[self.translation_list]
			np.save(file_path,self.vector)
		plot_range_dict = {'surf_o2':(0,3*10**-10),'surf_dic':(0,10**-14),\
		'surf_pco2':(200,1200),'100m_dic':(0,12),'100m_o2':(0,2)\
		}	
		self.plot_setup(variable,plot_range_dict)

class CM2p6VectorMean(CM2p6Vector):	
	def __init__(self,variable,**kwds):
		super(CM2p6VectorMean,self).__init__(**kwds)
		
		self.unit = self.unit_dict[variable]
		file_path = './data/cm2p6_vector_'+str(variable)+'_mean_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])

		try:
			self.vector = self.load_vector(file_path)
		except IOError:
			print 'cm2p6 file not found, recompiling'
			data = self.data_return(variable)
			self.vector = data.mean(axis=0)[self.translation_list]
			np.save(file_path,self.vector)
		plot_range_dict = {'surf_o2':(10,37),'surf_dic':(0,10**-7),\
		'surf_pco2':(200,400),'100m_dic':(195,225),'100m_o2':(15,35)\
		}	
		self.plot_setup(variable,plot_range_dict)

def plot_all_correlations():
	trans_plot = TransitionPlot()
	trans_plot.get_direction_matrix()
	for name,corr_class in [('CM2p6',CM2p6Correlation)]:
		for variable in ['o2','pco2','hybrid']:
			try:
				dummy_class = corr_class(transition_plot=trans_plot,variable=variable)
				m = dummy_class.plot()
				plt.title(name+' '+variable+' '+'correlation')
				plt.savefig('../../plots/'+name+'_'+variable+'_'+'correlation')
				plt.close()
			except AttributeError:
				pass