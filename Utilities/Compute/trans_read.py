from GeneralUtilities.Compute.list import find_nearest,flat_list
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list
from GeneralUtilities.Data.lagrangian.drifter_base_class import BaseRead
from TransitionMatrix.Utilities.Compute.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.Plot.plot_utils import cartopy_setup

import numpy as np
import scipy.sparse
import json
import scipy.sparse.linalg
import os 
from scipy.sparse import _sparsetools
from scipy.sparse.sputils import (get_index_dtype,upcast)
import pickle
import geopy

file_handler = FilePathHandler(ROOT_DIR,'trans_read')

class TransitionGeo(object):
	""" geo information and tools for transition matrices """
	file_type = 'argo'
	number_vmin=0
	number_vmax=300
	std_vmax=30
	def __init__(self,lat_sep=2,lon_sep=2,time_step=60):
		assert isinstance(lat_sep,int)
		assert isinstance(lon_sep,int)
		assert isinstance(time_step,int)

		self.lat_sep = lat_sep
		self.lon_sep = lon_sep
		self.time_step = time_step

	def plot_setup(self):
		XX,YY,ax,fig = cartopy_setup(self.get_lat_bins(),self.get_lon_bins(),self.file_type)
		return (XX,YY,ax,fig)

	def set_total_list(self,total_list):
		lats,lons,dummy = zip(*[tuple(x) for x in total_list])
		assert isinstance(total_list,list) 
		#total list must be a list
		assert all([isinstance(x,geopy.Point) for x in total_list]) 
		# total list must be composed of geopy.Points 
		assert (set(lats).issubset(set(self.get_lat_bins())))&(set(lons).issubset(set(self.get_lon_bins())))
		# total list must be a subset of the coordinate lists
		self.total_list = total_list #make sure they are unique

	def tuple_total_list(self):
		return [tuple(x)[:2] for x in self.total_list]

	def get_lat_bins(self):
		lat_bins = np.arange(-90,90.1,self.lat_sep)
		assert lat_bins.max()<=90
		assert lat_bins.min()>=-90
		return lat_bins

	def get_lon_bins(self):
		lon_bins = np.arange(-180,180.1,self.lon_sep)
		assert lon_bins.max()<=180
		assert lon_bins.min()>=-180
		return lon_bins

	def get_coords(self):
		XX,YY = np.meshgrid(self.get_lon_bins(),self.get_lat_bins())
		return (XX,YY)

	def transition_vector_to_plottable(self,vector):
		lon_grid = self.get_lon_bins().tolist()
		lat_grid = self.get_lat_bins().tolist()
		plottable = np.zeros([len(lon_grid),len(lat_grid)])
		plottable = np.ma.masked_equal(plottable,0)
		for n,pos in enumerate(self.total_list):
			ii_index = lon_grid.index(pos.longitude)
			qq_index = lat_grid.index(pos.latitude)
			plottable[ii_index,qq_index] = vector[n]
		return plottable.T

	def plottable_to_transition_vector(self,plottable):
		vector = np.zeros([len(index_list)])
		lon_grid = self.get_lon_bins().tolist()
		lat_grid = self.get_lat_bins().tolist()
		for n,pos in enumerate(self.total_list):
			lon_index = lon_grid.index(find_nearest(lon_grid,pos.longitude))
			assert abs(lon_grid[lon_index]-lon)<2
			lat_index = lat_grid.index(find_nearest(lat_grid,pos.longitude))
			assert abs(lat_grid[lat_index]-lat)<2
			vector[n] = plottable[lat_index,lon_index]
		return vector

	def make_filename(self):
		return file_handler.tmp_file(self.file_type+'-'+str(self.time_step)+'-'+str(self.lat_sep)+'-'+str(self.lon_sep))

	def get_direction_matrix(self):
		"""
		notes: this could be made faster by looping through unique values of lat and lon and assigning intelligently
		"""
		(lat_list,lon_list,dummy) = zip(*self.tuple_total_list())
		lat_list = np.array(lat_list)
		lon_list = np.array(lon_list)
		pos_max = 180/self.lon_sep #this is the maximum number of bins possible
		output_ns_list = []
		output_ew_list = []
		for (token_lat,token_lon,dummy) in self.tuple_total_list():
			token_ns = (token_lat-np.array(lat_list))/self.lat_sep
			token_ew = (token_lon-np.array(lon_list))/self.lon_sep
			token_ew[token_ew>pos_max]=token_ew[token_ew>pos_max]-2*pos_max #the equivalent of saying -360 degrees
			token_ew[token_ew<-pos_max]=token_ew[token_ew<-pos_max]+2*pos_max #the equivalent of saying +360 degrees
			output_ns_list.append(token_ns)
			output_ew_list.append(token_ew)
		self.east_west = np.array(output_ew_list)
		self.north_south = np.array(output_ns_list)
		assert (self.east_west<=180/self.lon_sep).all()
		assert (self.east_west>=-180/self.lon_sep).all()
		assert (self.north_south>=-180/self.lat_sep).all()
		assert (self.north_south<=180/self.lat_sep).all()

	@classmethod
	def new_from_old(cls,trans_geo):
		new_trans_geo = cls(lat_sep=trans_geo.lat_sep,lon_sep=trans_geo.lon_sep,time_step=trans_geo.time_step)
		new_trans_geo.set_total_list(trans_geo.total_list)
		return new_trans_geo

class SOSEGeo(TransitionGeo):
	file_type = 'SOSE'
	number_vmin=0
	number_vmax=900
	std_vmax=15

class SOSECaseGeo(SOSEGeo):
	def make_filename(self):
		return file_handler.tmp_file(self.description+'_'+self.file_type+'-'+str(self.time_step)+'-'+str(self.lat_sep)+'-'+str(self.lon_sep))

class SummerSOSEGeo(SOSECaseGeo):
	description = 'Summer'

class WinterSOSEGeo(SOSECaseGeo):
	description = 'Winter'

class CaseGeo(TransitionGeo):
	def make_filename(self):
		return file_handler.tmp_file(self.description+'_'+self.file_type+'-'+str(self.time_step)+'-'+str(self.lat_sep)+'-'+str(self.lon_sep))

class ARGOSGeo(CaseGeo):
	description = 'ARGOS_Positioning'

class GPSGeo(CaseGeo):
	description = 'GPS_Positioning'

class SummerGeo(CaseGeo):
	description = 'Summer'

class WinterGeo(CaseGeo):
	description = 'Winter'

class WithholdingGeo(CaseGeo):
	def __init__(self,percentage,idx_number,*args,**kwargs):
		self.description = str(percentage)+'_'+str(idx_number)
		super().__init__(*args,**kwargs)

	@classmethod
	def new_from_old(cls,trans_geo):
		return trans_geo

class SOSEWithholdingGeo(SOSECaseGeo):
	def __init__(self,percentage,idx_number,*args,**kwargs):
		self.description = str(percentage)+'_'+str(idx_number)
		super().__init__(*args,**kwargs)

	@classmethod
	def new_from_old(cls,trans_geo):
		return trans_geo

class BaseMat(scipy.sparse.csc_matrix):
	"""Base class for transition and correlation matrices, we include the timestep moniker for posterity. Time step for correlation matrices 
	means the L scaling """
	def __init__(self, arg1,shape=None,trans_geo=None,**kwargs):
		super(BaseMat,self).__init__(arg1,shape=shape)
		if trans_geo:
			self.set_trans_geo(trans_geo)

	@classmethod
	def load_from_type(cls,GeoClass=TransitionGeo,lat_spacing=None,lon_spacing=None,time_step=None):
		trans_geo = GeoClass(lat_sep=lat_spacing,lon_sep=lon_spacing,time_step=time_step)
		file_name = trans_geo.make_filename()
		return cls.load(file_name)

	@classmethod
	def load(cls,filename):
		with open(filename,'rb') as pickle_file:
			out_data = pickle.load(pickle_file)
		out_data = cls.new_from_old(out_data)
		return out_data

	@classmethod
	def new_from_old(cls,transition_matrix):
		old_trans_geo = transition_matrix.trans_geo
		trans_geo = old_trans_geo.new_from_old(old_trans_geo)
		row_idx,column_idx,data = scipy.sparse.find(transition_matrix)
		return cls((data,(row_idx,column_idx)),
			shape=(len(transition_matrix.trans_geo.total_list),
			len(transition_matrix.trans_geo.total_list)),
			trans_geo=trans_geo,
			number_data=transition_matrix.number_data) 

	def new_sparse_matrix(self,data):
		row_idx,column_idx,dummy = scipy.sparse.find(self)
		return scipy.sparse.csc_matrix((data,(row_idx,column_idx)),shape=(len(self.trans_geo.total_list),len(self.trans_geo.total_list)))             

	def _binopt(self, other, op):
		""" This is included so that when sparse matrices are added together, their instance variables are maintained this code was grabbed from the scipy source with the small addition at the end"""

		other = self.__class__(other)

		# e.g. csr_plus_csr, csr_minus_csr, etc.
		fn = getattr(_sparsetools, self.format + op + self.format)

		maxnnz = self.nnz + other.nnz
		idx_dtype = get_index_dtype((self.indptr, self.indices,
									 other.indptr, other.indices),
									maxval=maxnnz)
		indptr = np.empty(self.indptr.shape, dtype=idx_dtype)
		indices = np.empty(maxnnz, dtype=idx_dtype)

		bool_ops = ['_ne_', '_lt_', '_gt_', '_le_', '_ge_']
		if op in bool_ops:
			data = np.empty(maxnnz, dtype=np.bool_)
		else:
			data = np.empty(maxnnz, dtype=upcast(self.dtype, other.dtype))

		fn(self.shape[0], self.shape[1],
		   np.asarray(self.indptr, dtype=idx_dtype),
		   np.asarray(self.indices, dtype=idx_dtype),
		   self.data,
		   np.asarray(other.indptr, dtype=idx_dtype),
		   np.asarray(other.indices, dtype=idx_dtype),
		   other.data,
		   indptr, indices, data)
		if issubclass(type(self),TransMat):
			A = self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo
				,number_data=self.number_data)
		else:
			A = self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo
				,number_data=self.number_data)
		A.prune()

		return A


	def _mul_sparse_matrix(self, other):
		""" This is included so that when sparse matrices are multiplies together, 
		their instance variables are maintained this code was grabbed from the scipy 
		source with the small addition at the end"""
		M, K1 = self.shape
		K2, N = other.shape

		major_axis = self._swap((M, N))[0]
		other = self.__class__(other)  # convert to this format

		idx_dtype = get_index_dtype((self.indptr, self.indices,
									 other.indptr, other.indices))

		fn = getattr(_sparsetools, self.format + '_matmat_maxnnz')
		nnz = fn(M, N,
				 np.asarray(self.indptr, dtype=idx_dtype),
				 np.asarray(self.indices, dtype=idx_dtype),
				 np.asarray(other.indptr, dtype=idx_dtype),
				 np.asarray(other.indices, dtype=idx_dtype))

		idx_dtype = get_index_dtype((self.indptr, self.indices,
									 other.indptr, other.indices),
									maxval=nnz)

		indptr = np.empty(major_axis + 1, dtype=idx_dtype)
		indices = np.empty(nnz, dtype=idx_dtype)
		data = np.empty(nnz, dtype=upcast(self.dtype, other.dtype))

		fn = getattr(_sparsetools, self.format + '_matmat')
		fn(M, N, np.asarray(self.indptr, dtype=idx_dtype),
		   np.asarray(self.indices, dtype=idx_dtype),
		   self.data,
		   np.asarray(other.indptr, dtype=idx_dtype),
		   np.asarray(other.indices, dtype=idx_dtype),
		   other.data,
		   indptr, indices, data)

		if issubclass(type(self),TransMat):
			return self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo
				,number_data=self.number_data)
		else:
			return self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo
				,number_data=self.number_data)



class TransMat(BaseMat):
	def __init__(self, arg1,number_data=None,rescale=False,**kwargs):
		super(TransMat,self).__init__(arg1,**kwargs)
		self.number_data = number_data
		if rescale:
			self.rescale()

	def set_trans_geo(self,trans_geo):
		self.trans_geo = trans_geo.__class__.new_from_old(trans_geo)

	def save(self,filename=False):
		if not filename:
			filename = self.trans_geo.make_filename()
		with open(filename, 'wb') as pickle_file:
			pickle.dump(self,pickle_file)
		pickle_file.close()

	def return_mean(self):
		row_list, column_list, data_array = scipy.sparse.find(self)
		self.get_direction_matrix()
		east_west_data = self.east_west[row_list,column_list]*data_array
		north_south_data = self.north_south[row_list,column_list]*data_array
		east_west = self.new_sparse_matrix(east_west_data)
		north_south = self.new_sparse_matrix(north_south_data)        
		return(east_west,north_south)

	def add_noise(self,noise=0.05):
		"""
		Adds guassian noise to the transition matrix
		The appropriate level of noise has not been worked out and is kind of ad hock
		"""
		print('adding matrix noise')
		self.get_direction_matrix()
		direction_mat = (-abs(self.east_west)**2-abs(self.north_south)**2)
		noise_mat = abs(np.random.normal(0,noise,direction_mat.shape[0]*direction_mat.shape[1])).reshape(self.east_west.shape)
		direction_mat = noise*np.exp(direction_mat)
		direction_mat[direction_mat<noise/200]=0
		row_idx,column_idx,data = scipy.sparse.find(self)
		self.data+= direction_mat[row_idx,column_idx]
		self.rescale()

	def new_coordinate_list(self,new_total_list):
		trans_geo = self.trans_geo
		matrix_idx = [self.trans_geo.total_list.index(x) for x in new_total_list]
		new_matrix = self[matrix_idx,:]
		new_matrix = new_matrix[:,matrix_idx]
		new_matrix.set_trans_geo(trans_geo)
		new_matrix.trans_geo.set_total_list(new_total_list)
		new_matrix.rescale()
		return new_matrix

	def rescale(self,checksum=10**-2):
		div_array = np.abs(self.sum(axis=0)).tolist()[0]
		row_idx,column_idx,data = scipy.sparse.find(self)
		col_count = []
		for col in column_idx:
			col_count.append(float(div_array[col]))
		self.data = np.array(data)/np.array(col_count)
		zero_idx = np.where(np.abs(self.sum(axis=0))==0)[1]
		self[zero_idx,zero_idx]=1
		self.matrix_column_check(checksum=checksum)

	def matrix_eig_check(self,checksum=10**-5,bool_return=False):
		eig_vals,eig_vecs = scipy.sparse.linalg.eigs(self,k=30)
		if bool_return:
			return bool(np.where((eig_vecs>checksum).sum(axis=0)<=3)[0].tolist())
		else:
			assert not np.where((eig_vecs>checksum).sum(axis=0)<=3)[0].tolist()

	def matrix_column_check(self,checksum):
		assert (np.abs(self.sum(axis=0)-1)<checksum).all()

	def reduce_resolution(self,lat_sep,lon_sep):
		lat_mult = lat_sep/self.trans_geo.lat_sep
		lon_mult = lon_sep/self.trans_geo.lon_sep

		new_trans_geo = self.trans_geo.__class__(lat_sep=lat_sep,lon_sep=lon_sep,time_step=self.trans_geo.time_step)
		reduced_res_lat_bins = new_trans_geo.get_lat_bins()
		reduced_res_lon_bins = new_trans_geo.get_lon_bins()
		lat_bins,lon_bins = zip(*self.trans_geo.tuple_total_list())
		lat_idx = np.digitize(lat_bins,reduced_res_lat_bins)-1
		lon_idx = np.digitize(lon_bins,reduced_res_lon_bins)-1

		reduced_res_bins=[(reduced_res_lat_bins[i],reduced_res_lon_bins[j]) for i,j in zip(lat_idx,lon_idx)]
		reduced_res_total_list = [geopy.Point(x) for x in list(set((reduced_res_bins)))]
		new_trans_geo.set_total_list(reduced_res_total_list)
		translation_list = [new_trans_geo.tuple_total_list().index(x) for x in reduced_res_bins]
		check = [(np.array(translation_list)==x).sum()<=(lat_mult*lon_mult) for x in range(len(reduced_res_total_list))]
		assert all(check)
		translation_dict = dict(zip(range(len(self.trans_geo.tuple_total_list())),translation_list))
		
		old_row_idx,old_column_idx,old_data = scipy.sparse.find(self)
		new_row_idx = np.array([translation_dict[ii] for ii in old_row_idx])
		new_col_idx = np.array([translation_dict[ii] for ii in old_column_idx])
		out_data = []
		out_col = []
		out_row = []
		for row_dummy, col_dummy in list(set((zip(new_row_idx,new_col_idx)))):
			mask = (new_row_idx==row_dummy)&(new_col_idx==col_dummy)
			data_dummy = old_data[mask]
			assert len(data_dummy)<=(lat_mult*lon_mult)**2
#squared because of the possibilty of inter box exchange 
			out_data.append(data_dummy.sum())
			out_col.append(col_dummy)
			out_row.append(row_dummy)
		mat_dim = len(reduced_res_total_list)
		return self.__class__((out_data,(out_row,out_col)),shape=(mat_dim,mat_dim),trans_geo=new_trans_geo
				,number_data=None,rescale=True)	

	def save_trans_matrix_to_json(self,foldername):
		for column in range(self.shape[1]):
			print('there are ',(self.shape[1]-column),'columns remaining')
			p_lat,p_lon = tuple(self.trans_geo.total_list[column])
			data = self[:,column].data
			lat,lon = zip(*[tuple(self.trans_geo.total_list[x]) for x in self[:,column].indices.tolist()])
			feature_list = zip(lat,lon,data)
			geojson = {'type':'FeatureCollection', 'features':[]}
			for token in feature_list:
				lat,lon,prob = token
				feature = {'type':'Feature',
					'properties':{},
					'geometry':{'type':'Point',
					'coordinates':[]}}
				feature['geometry']['coordinates'] = [lon,lat]
				feature['properties']['Probability'] = prob
				geojson['features'].append(feature)
			output_filename = './'+str(foldername)+'/lat_'+str(p_lat)+'_lon_'+str(p_lon)+'.js'
			with open(output_filename,'wb') as output_file:
				output_file.write('var dataset = ')
				json.dump(geojson, output_file, indent=2) 