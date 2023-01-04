from GeneralUtilities.Compute.list import flat_list,LonList,LatList,GeoList
from GeneralUtilities.Data.Lagrangian.Argo.array_class import ArgoArray
from TransitionMatrix.Utilities.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.TransGeo import TransitionGeo

import numpy as np
import scipy.sparse
import json
import scipy.sparse.linalg
import os 
from scipy.sparse import _sparsetools
from scipy.sparse.sputils import (get_index_dtype,upcast)
import pickle
import geopy

file_handler = FilePathHandler(ROOT_DIR,'TransMat')

class BaseMat(scipy.sparse.csc_matrix):
	"""Base class for transition and correlation matrices"""
	def __init__(self, *args,trans_geo=None,**kwargs):
		super().__init__(*args,**kwargs)
		if trans_geo:
			self.set_trans_geo(trans_geo)

	def set_trans_geo(self,trans_geo):
		self.trans_geo = trans_geo.__class__.new_from_old(trans_geo)

	@classmethod
	def load_from_type(cls,GeoClass=TransitionGeo,lat_spacing=None,lon_spacing=None,time_step=None):
		trans_geo = GeoClass(lat_sep=lat_spacing,lon_sep=lon_spacing,time_step=time_step)
		file_name = trans_geo.make_filename()
		return cls.load(file_name,GeoClass)

	@classmethod
	def load(cls,filename,GeoClass=TransitionGeo):
		with open(filename,'rb') as pickle_file:
			out_data = pickle.load(pickle_file)
		out_data = cls.new_from_old(out_data,GeoClass)
		return out_data

	def save(self,filename=False):
		if not filename:
			filename = self.trans_geo.make_filename()
		with open(filename, 'wb') as pickle_file:
			pickle.dump(self,pickle_file)
		pickle_file.close()

	@classmethod
	def new_from_old(cls,transition_matrix,GeoClass=TransitionGeo):
		old_trans_geo = transition_matrix.trans_geo
		trans_geo = GeoClass.new_from_old(old_trans_geo)
		assert isinstance(trans_geo.total_list,GeoList) 
		row_idx,column_idx,data = scipy.sparse.find(transition_matrix)
		return cls((data,(row_idx,column_idx)),
			shape=(len(transition_matrix.trans_geo.total_list),
			len(transition_matrix.trans_geo.total_list)),
			trans_geo=trans_geo,
			number_data=transition_matrix.number_data) 

	def new_sparse_matrix(self,data):
		row_idx,column_idx,dummy = scipy.sparse.find(self)
		return BaseMat((data,(row_idx,column_idx)),shape=(len(self.trans_geo.total_list),len(self.trans_geo.total_list)))             

	def mean(self,axis=0):
		return np.array(self.sum(axis=axis)/(self!=0).sum(axis=axis)).flatten()

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
			try:
				A = self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo
					,number_data=self.number_data)
			except AttributeError:
				A = self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo)
		else:
			try:
				A = self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo
					,number_data=self.number_data)
			except AttributeError:
				A = self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo)				
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
			try:
				return self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo
					,number_data=self.number_data)
			except AttributeError:
				return self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo)
		else:
			try:
				return self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo
					,number_data=self.number_data)
			except AttributeError:
				return self.__class__((data, indices, indptr), shape=self.shape,trans_geo=self.trans_geo)


class TransMat(BaseMat):
	def __init__(self, *args,number_data=None,rescale=False,**kwargs):
		super().__init__(*args,**kwargs)
		self.number_data = number_data
		if rescale:
			self.rescale()

	def mean_of_column(self,geo_point):
		col_idx = self.trans_geo.total_list.index(geo_point)
		start_lat,start_lon,dummy = tuple(geo_point)
		idx_lat,idx_lon,dummy = tuple(self.trans_geo.total_list[col_idx])
		east_west, north_south = self.return_mean()
		x_mean = east_west[col_idx]+idx_lon
		y_mean = north_south[col_idx]+idx_lat
		return geopy.Point(y_mean,x_mean)


	def remove_small_values(self,value):
		row_idx,column_idx,data = scipy.sparse.find(self)
		mask = data>value
		row_idx = row_idx[mask]
		column_idx = column_idx[mask]
		data = data[mask]
		return self.__class__((data,(row_idx,column_idx)),shape=self.shape,trans_geo=self.trans_geo
				,number_data=self.number_data,rescale=True)	

	def multiply(self,mult,value=0.02):
		mat1 = self.remove_small_values(self.data.mean()/25)
		mat2 = self.remove_small_values(self.data.mean()/25)
		for k in range(mult):
			print('I am at ',k,' step in the multiplication')
			mat_holder = mat1.dot(mat2)
			mat1 = mat_holder.remove_small_values(mat_holder.data.mean()/25)
		return mat1

	def return_mean(self):
		row_list, column_list, data_array = scipy.sparse.find(self)
		try:
			self.trans_geo.east_west
		except AttributeError:
			self.trans_geo.get_direction_matrix()
		ew_scaled = self.new_sparse_matrix(self.trans_geo.east_west[row_list, column_list]*data_array*self.trans_geo.lon_sep).sum(axis=0)
		ew_scaled = np.array(ew_scaled).flatten()
		ns_scaled = self.new_sparse_matrix(self.trans_geo.north_south[row_list, column_list]*data_array*self.trans_geo.lat_sep).sum(axis=0)
		ns_scaled = np.array(ns_scaled).flatten()
		return(ew_scaled,ns_scaled)

	def return_std(self):
		ew_out = []
		ns_out = []
		row_list, column_list, data_array = scipy.sparse.find(self)
		try:
			self.trans_geo.east_west
		except AttributeError:
			self.trans_geo.get_direction_matrix()
		ew = self.new_sparse_matrix(self.trans_geo.east_west[row_list, column_list]*data_array*self.trans_geo.lon_sep)
		ns = self.new_sparse_matrix(self.trans_geo.north_south[row_list, column_list]*data_array*self.trans_geo.lon_sep)
		for k in range(ew.shape[0]):
			ew_out.append(ew[:,k].data.std())
			ns_out.append(ns[:,k].data.std())
		ew_out = np.array(ew_out).flatten()
		ns_out = np.array(ns_out).flatten()
		return (ew_out,ns_out)

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
		number_matrix = self.new_sparse_matrix(self.number_data)
		number_matrix = number_matrix[matrix_idx,:]
		number_matrix = number_matrix[:,matrix_idx]
		DUM,DUM,data = scipy.sparse.find(number_matrix)
		new_matrix.number_data = data
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
		lat_bins,lon_bins = self.trans_geo.total_list.lats_lons()
		lat_idx = np.digitize(lat_bins,reduced_res_lat_bins)-1
		lon_idx = np.digitize(lon_bins,reduced_res_lon_bins)-1

		reduced_res_bins=[(reduced_res_lat_bins[i],reduced_res_lon_bins[j]) for i,j in zip(lat_idx,lon_idx)]
		reduced_res_total_list = [geopy.Point(x) for x in list(set((reduced_res_bins)))]
		new_trans_geo.set_total_list(reduced_res_total_list)
		translation_list = [new_trans_geo.total_list.tuple_total_list().index(x) for x in reduced_res_bins]
		check = [(np.array(translation_list)==x).sum()<=(lat_mult*lon_mult) for x in range(len(reduced_res_total_list))]
		assert all(check)
		translation_dict = dict(zip(range(len(self.trans_geo.total_list.tuple_total_list())),translation_list))
		
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