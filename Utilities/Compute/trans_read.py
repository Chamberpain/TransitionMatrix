from __future__ import print_function
from compute_utilities.list_utilities import find_nearest,flat_list
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import scipy.sparse
import json
import scipy.sparse.linalg
import pandas as pd
import scipy.spatial as spatial
import os 
from scipy.sparse import _sparsetools
from scipy.sparse.sputils import (get_index_dtype,upcast)
from data_save_utilities.lagrangian.drifter_base_class import BaseRead
from random import shuffle


class TransitionCalc(object):
	"""class that takes the argo read instance as input. This is a cludgy way of doing things because
	it should probably just be inherited, but helped the debug process considerably. 

	Performs parsing logic to determine the proper transitions 
	"""
	def __init__(self,read_instance,lat_list,lon_list,time_step=60,checksum=10**-3):
		self.checksum=checksum
		print(read_instance.meta.id)
		self.instance = read_instance
		self.time_step = time_step
		start_pos_indexes, end_pos_indexes = self.get_start_and_end_indexes()
		self.start_bin_list = list(self.instance.prof.pos.return_pos_bins(lat_list,lon_list,index_values=start_pos_indexes,index_return=False))
		self.end_bin_list = list(self.instance.prof.pos.return_pos_bins(lat_list,lon_list,index_values=end_pos_indexes,index_return=False))


	def is_bin_contained(self,masked_start_values,masked_end_values):
		dummy_start = [list(_) for _ in self.start_bin_list]
		dummy_end = [list(_) for _ in self.end_bin_list]
		start_mask = (np.array(dummy_start)[:,None] == masked_start_values).all(2).any(1)
		end_mask = (np.array(dummy_end)[:,None] == masked_end_values).all(2).any(1)
		return start_mask|end_mask	

	def return_masked_start_and_end_bins(self,masked_start_values=None,masked_end_values=None):
		mask = self.is_bin_contained(masked_start_values,masked_end_values)
		dummy_start = np.array(self.start_bin_list)[~mask].tolist()
		dummy_end = np.array(self.end_bin_list)[~mask].tolist()
		return (dummy_start,dummy_end)

	def plot(self):
		""" plots basic map of profile locations as well as bin values. Mostly used during debugging"""

		start_bin_lat, start_bin_lon = zip(*self.start_bin_list)
		end_bin_lat, end_bin_lon = zip(*self.end_bin_list)
		plt.scatter(start_bin_lon,start_bin_lat, s=20)
		plt.scatter(end_bin_lon,end_bin_lat, s=10)

	def get_start_and_end_indexes(self):
		""" finds the start and end indexes for the transition matrix positions
			Performs logic to eliminate values that are out of diff_check tollerances

			Parameters
			----------
			None

			Returns
			-------
			List of start indexes and list of end indexes
			 """

		def get_index_of_decorrelated(self,time_delta=10):
			""" Argo trajectories are correlated on time scales of about 30 days (Gille et al. 2003). This subsampling removes the possibiity
				of dependent data 
				
				Parameters
				----------
				time delta: float that describes the time difference you prescribe as decorrelation (in days)

				Returns
				-------
				list of the indexes of the decorrelated start positions
				 """

			idx_list = []
			seconds_to_days = 1/(3600*24.)
			diff_list = [(_-self.instance.prof.date._list[0]).total_seconds()*seconds_to_days for _ in self.instance.prof.date._list]
			diff_array = np.array(diff_list)
			time_list = np.arange(0,max(diff_list),time_delta)
			for time in time_list:
				idx_list.append(diff_list.index(diff_array[diff_array>=time][0]))
			return idx_list

		def find_next_time_index(self,start_index,time_delta,diff_check=None):
			""" finds the index cooresponding to the nearest time delta from a start index

				Parameters
				----------
				start_index: the index at which you are starting your calculation from
				time delta: float that describes the time difference you want in your list (in days)
				diff_check (optional) : float that describes the acceptable difference away from your time delta

				Returns
				-------
				Index of the next time
				 """
			if not diff_check:
				diff_check = time_delta/3.
			seconds_to_days = 1/(3600*24.)
			diff_list = [(_-self.instance.prof.date._list[start_index]).total_seconds()*seconds_to_days for _ in self.instance.prof.date._list[(start_index+1):]]
			if not diff_list:
				return None
			closest_diff = find_nearest(diff_list,time_delta,test=False)
			if abs(closest_diff-time_delta)>diff_check:
				return None
			else:
				return start_index+diff_list.index(closest_diff)+1
		start_indexes = get_index_of_decorrelated(self)
		end_indexes = [find_next_time_index(self,_,self.time_step) for _ in start_indexes]
		mask=[i for i,v in enumerate(end_indexes) if v != None]
		start_indexes = np.array(start_indexes)[mask].tolist()
		end_indexes = np.array(end_indexes)[mask].tolist()
		return (start_indexes,end_indexes)

	def construct_trans_and_number_matrix(self,total_list):
		"""
		Function that creates the transition matrix from the available start index and end index lists

		Parameters
		----------
		Total list of the ordering of the unique position bins

		Returns
		-------
		Transition matrix in CSC form
		 """


		def index_classes(total_list):
			"""
			Function that gets the indexes of the start and end bin lists

			Parameters
			----------
			total list of the ordering of the position bins

			Returns
			-------
			2 lists that show the start and end indexes of the bins
			 """

			start_index_list = []
			end_index_list = []
			for dummy in zip(self.start_bin_list,self.end_bin_list):
				try: 
					start_index = total_list.index(dummy[0])
					end_index = total_list.index(dummy[1])
				except ValueError:
					continue
				start_index_list.append(start_index)
				end_index_list.append(end_index)
			return (start_index_list,end_index_list)			

		start_index_list, end_index_list = index_classes(total_list)
		if not start_index_list: # if none of the start or end bins were in the total list
			return (scipy.sparse.csc_matrix(([],([],[])),shape=(len(total_list),len(total_list))),
					scipy.sparse.csc_matrix(([],([],[])),shape=(len(total_list),len(total_list))))
		mix_count = Counter(zip(start_index_list,end_index_list)) 
		(cells,cell_count) = zip(*mix_count.items())
		col_list,row_list = zip(*cells)
		
		start_count = Counter(start_index_list)
		col_count = []
		for col in col_list:
			col_count.append(float(start_count[col]))
		trans_data = np.array(cell_count)/np.array(col_count)
		transition_matrix = scipy.sparse.csc_matrix((trans_data,(row_list,col_list)),shape=(len(total_list),len(total_list)))
		number_matrix = scipy.sparse.csc_matrix((cell_count,(row_list,col_list)),shape=(len(total_list),len(total_list)))
		assert (np.abs(transition_matrix.sum(axis=0).T[list(col_list)]-1)<self.checksum).all()
		return (number_matrix,transition_matrix)

class SubclassedDataFrame(pd.DataFrame):
	def return_index_of_not_enough_numbers(self):
		number_start = self.groupby(['StartIDX']).count()
		return number_start.index[(number_start<3).EndIDX].values.tolist()

	def return_count_df(self):
		df_ = self.groupby(['StartIDX','EndIDX']).size().reset_index(name='Count')
		row_idx = df_['EndIDX'].values.tolist()
		col_idx = df_['StartIDX'].values.tolist()
		data = df_['Count'].values.tolist()
		return (row_idx,col_idx,data)

def TransitionClassAggPrep(drifter_class,lat_spacing=2,lon_spacing=2,time_step=90,eig_vecs_flag=True):

		lat_bins = np.arange(-90,90.1,lat_spacing)
		lon_bins = np.arange(-180,180.1,lon_spacing)
		time_step = time_step
		all_dict = drifter_class.all_dict
		data_description = drifter_class.data_description 
		dummy_list = [TransitionCalc(_,lat_bins,lon_bins,time_step=time_step) for _ in all_dict.values()]
		return dummy_list

def TransitionClassAgg(drifter_class,fresh_dummy_list,lat_spacing=2,lon_spacing=2,time_step=90,eig_vecs_flag=True,percentage=None):
		def remove_percentage(list_a, percentage):
		    shuffle(list_a)
		    count = int(len(list_a) * percentage)
		    if not count: return []  # edge case, no elements removed
		    list_a[-count:], list_b = [], list_a[-count:]
		    return list_b		

		if percentage:
			dummy_list = remove_percentage(fresh_dummy_list.copy(),percentage)
			assert len(dummy_list)<len(fresh_dummy_list)
		else:
			dummy_list = fresh_dummy_list


		start_list = flat_list([_.return_masked_start_and_end_bins()[0] for _ in dummy_list])
		start_list = [tuple(x) for x in start_list]
		start_set = set((start_list))
		end_list = flat_list([_.return_masked_start_and_end_bins()[1] for _ in dummy_list])
		end_list = [tuple(x) for x in end_list]
		end_set = set((end_list))

		set_overlap = list(start_set.intersection(end_set))
		lat_sep = np.diff(lat_bins)[0]
		lon_sep = np.diff(lon_bins)[0]
		flag = False
		while not flag:
			overlap_dict = dict(zip(set_overlap,range(len(set_overlap))))
			return_list = []
			for start_idx,end_idx in zip(start_list,end_list):  
				try:              
					start_idx = overlap_dict[start_idx]
					end_idx = overlap_dict[end_idx]
					return_list.append((start_idx,end_idx))
				except KeyError:
					continue
			print('finished the index list calculations')
			df = SubclassedDataFrame(return_list,columns=['StartIDX','EndIDX'])

			if df.return_index_of_not_enough_numbers():
				print('there were '+str(len(df.return_index_of_not_enough_numbers()))+' indexes that were not large enough')
				not_vacant_cells = set(([set_overlap[_] for _ in df.StartIDX.unique().tolist()]))
				set_overlap = set((set_overlap)).difference(set(([set_overlap[_] for _ in df.return_index_of_not_enough_numbers()])))
				set_overlap = set_overlap.intersection(not_vacant_cells)
				set_overlap = list(set_overlap)
				print('length of set overlap is ',len(set_overlap))
				print(set_overlap)
				continue
			print('length of set overlap is ',len(set_overlap))
			tree = spatial.KDTree(set_overlap)
			closest_bins_spacing,_ = tree.query(set_overlap,k=2)
			idx_list = np.where(closest_bins_spacing[:,1]>np.sqrt(lat_sep**2+lon_sep**2))[0].tolist()
			if idx_list:
				print('there were isolated grid points')
				set_overlap = set((set_overlap)).difference(set(([set_overlap[_] for _ in idx_list])))
				set_overlap = list(set_overlap)
				continue

			row_idx,col_idx,data = df.return_count_df()
			mat = BaseMat((data,(row_idx,col_idx)),shape=(len(set_overlap)
				,len(set_overlap)))
			div_array = np.abs(mat.sum(axis=0)).tolist()[0]
			div_list = np.where(np.array(div_array)==0)[0].tolist()
			if div_list:
				set_overlap = set((set_overlap)).difference(set(([set_overlap[_] for _ in div_list])))
				set_overlap = list(set_overlap)
				continue
			col_count = []
			for col in col_idx:
				col_count.append(float(div_array[col]))
			scaled_data = np.array(data)/np.array(col_count)

			transition_matrix = TransMat((scaled_data,(row_idx,col_idx)),shape=(len(set_overlap)
				,len(set_overlap)),total_list=set_overlap
				,lat_spacing = np.diff(self.lat_bins[:2])[0],lon_spacing=np.diff(self.lon_bins[:2])[0]
				,time_step = self.time_step,number_data=data,traj_file_type=self.data_description)
			if eig_vecs_flag:
				checksum=10**-5
				print('begin calculating eigen vectors')
				eig_val,eig_vecs = scipy.sparse.linalg.eigs(transition_matrix,k=20)
				print('calculated the eigen vectors')
				checksum = [abs(eig_vecs[:,k]).max()-5*abs(eig_vecs[:,k]).std() for k in range(eig_vecs.shape[1])]
				problem_idx = [k for k in range(eig_vecs.shape[1]) if (abs((eig_vecs[:,k]))>checksum[k]).sum()<=3]
				if problem_idx:
					print('these were the problem idx: '+str(problem_idx))
					masked_values = []
					for idx in problem_idx:
						masked_values += [set_overlap[i] for i in np.where(eig_vecs[:,idx]>checksum[idx])[0].tolist()]
					set_overlap = set((set_overlap)).difference(set((masked_values)))
					set_overlap = list(set_overlap)
					continue
			flag = True
		if percentage:
			return transition_matrix
		else:
			transition_matrix.save()


	# def overlap_calc(self,unique_end_list,unique_start_list):
	# 	"""
	# 	Take the unique start and end lists and figure out where they dont overlap (which means that the transition matrix would
	# 	have cells that have no start or exit)
	# 	"""
	# 	unique_end_list = [tuple(_) for _ in flat_list(unique_end_list)]
	# 	unique_start_list = [tuple(_) for _ in flat_list(unique_start_list)]
	# 	unique_end_list = set((unique_end_list))
	# 	unique_start_list = set((unique_start_list))
	# 	start_list_overlap = list(unique_start_list.difference(unique_end_list))
	# 	end_list_overlap = list(unique_end_list.difference(unique_start_list))
	# 	return (start_list_overlap, end_list_overlap)




	# def overlap_iterate(self,unique_end_list,unique_start_list,start_list_overlap,end_list_overlap):
	# 	masked_start_values = []
	# 	masked_end_values = []

	# 	while (end_list_overlap!=[])|(start_list_overlap!=[]):
	# 		masked_start_values += start_list_overlap
	# 		masked_end_values += end_list_overlap
	# 		masked_start_values = list(set((masked_start_values)))
	# 		masked_end_values = list(set((masked_end_values)))
	# 		print(len(start_list_overlap))
	# 		print(len(end_list_overlap))
	# 		print(len(masked_start_values))
	# 		print(len(masked_end_values))

	# 		id_list = [i[1].instance.meta.id for i in self.all_dict.iteritems()]
	# 		bool_list = [i[1].is_bin_contained(masked_start_values,masked_end_values).any() for i in self.all_dict.iteritems()]
	# 		idx_to_fix = np.where(bool_list)
	# 		for idx in idx_to_fix[0]:
	# 			dummy_id = id_list[idx]
	# 			dummy_start,dummy_end = self.all_dict[dummy_id].return_masked_start_and_end_bins(masked_start_values=masked_start_values,masked_end_values=masked_end_values)
	# 			unique_start_list[idx]=dummy_start
	# 			unique_end_list[idx]=dummy_end

	# 		start_list_overlap,end_list_overlap = self.overlap_calc(unique_end_list,unique_start_list)
	# 	start_list = set(([tuple(i) for i in flat_list(unique_start_list)]))
	# 	end_list = set(([tuple(i) for i in flat_list(unique_end_list)]))
	# 	return (list(start_list.intersection(end_list)),masked_start_values,masked_end_values)

	# def index_transcalc_classes(self):
	# 	masked_end_values = []
	# 	masked_start_values = []
	# 	continue_flag = True

	# 	"""this creates the total list of transitions, because of edge cases where a float trajectory may have a departure from a cell
	# 	and no arrival, we have to iteratively mask our results"""

	# 	unique_start_list = []
	# 	unique_end_list = []
	# 	unique_start_list = []
	# 	unique_end_list = []
	# 	id_list = []
	# 	for dummy in self.all_dict.itervalues():
	# 		"""
	# 		Loop through all items in the class dictionary. Grab the masked start and end bins and make a nested list			
	# 		"""
	# 		assert len(dummy.start_bin_list)==len(dummy.end_bin_list)
	# 		dummy_start,dummy_end = dummy.return_masked_start_and_end_bins()
	# 		unique_start_list.append(dummy_start)
	# 		unique_end_list.append(dummy_end)
		
	# 	start_list_remove,end_list_remove = self.overlap_calc(unique_end_list,unique_start_list)
	# 	temp_total_list,masked_start_values,masked_end_values = self.overlap_iterate(unique_end_list,unique_start_list,start_list_remove,end_list_remove)



	# 	def create_trans_matrix_from_list(temp_total_list):
	# 		matrix_list = []
	# 		num_matrix_list = []
	# 		for k,dummy in enumerate(self.all_dict.values()):
	# 			if (k%1000)==0:
	# 				print(k)
	# 			num_matrix,matrix = dummy.construct_trans_and_number_matrix(temp_total_list)
	# 			matrix_list.append(matrix)
	# 			num_matrix_list.append(num_matrix)
	# 		transition_matrix = np.sum([matrix_list])
	# 		row_idx,column_idx,data = scipy.sparse.find(transition_matrix)
	# 		num_matrix = np.sum([num_matrix_list])
	# 		transition_matrix = TransMat((data,(row_idx,column_idx)),shape=(len(temp_total_list)
	# 			,len(temp_total_list)),total_list=temp_total_list
	# 			,lat_spacing = np.diff(self.lat_bins[:2])[0],lon_spacing=np.diff(self.lon_bins[:2])[0]
	# 			,time_step = self.time_step,number_data=num_matrix.data,traj_file_type=self.data_description)
	# 		return (transition_matrix,num_matrix)

	# 	transition_matrix,num_matrix = create_trans_matrix_from_list(temp_total_list)
	# 	checksum=10**-5


	# 	while transition_matrix.matrix_eig_check(bool_return=True,checksum=checksum):
	# 		eig_vals,eig_vecs = scipy.sparse.linalg.eigs(transition_matrix,k=50)
	# 		problem_idx = np.where((eig_vecs>checksum).sum(axis=0)<=3)[0]
	# 		for idx in problem_idx.tolist():
	# 			masked_end_values += [temp_total_list[i] for i in np.where(eig_vecs[:,idx]>checksum)[0]]
	# 			print([temp_total_list[i] for i in np.where(eig_vecs[:,idx]>checksum)[0]])
	# 			masked_start_values += [temp_total_list[i] for i in np.where(eig_vecs[:,idx]>checksum)[0]]
	# 			print([temp_total_list[i] for i in np.where(eig_vecs[:,idx]>checksum)[0]])
	# 		# temp_total_list,masked_start_values,masked_end_values,unique_start_list,unique_end_list = self.overlap_iterate(unique_start_list,unique_end_list,masked_start_values,masked_end_values)
			
	# 		start_list_overlap = masked_start_values
	# 		end_list_overlap = masked_end_values

	# 		while (end_list_overlap!=[])|(start_list_overlap!=[]):
	# 			masked_start_values += start_list_overlap
	# 			masked_end_values += end_list_overlap
	# 			masked_start_values = list(set((masked_start_values)))
	# 			masked_end_values = list(set((masked_end_values)))
	# 			print('length of the start list is', len(start_list_overlap))
	# 			print( 'length of the end list is', len(end_list_overlap))
	# 			print('length of the masked start list is', len(masked_start_values))
	# 			print('length of the masked end list is', len(masked_end_values))

	# 			id_list = [i[1].instance.meta.id for i in self.all_dict.iteritems()]
	# 			bool_list = [i[1].is_bin_contained(masked_start_values,masked_end_values).any() for i in self.all_dict.iteritems()]
	# 			idx_to_fix = np.where(bool_list)
	# 			for idx in idx_to_fix[0]:
	# 				dummy_id = id_list[idx]
	# 				dummy_start,dummy_end = self.all_dict[dummy_id].return_masked_start_and_end_bins(masked_start_values=masked_start_values,masked_end_values=masked_end_values)
	# 				unique_start_list[idx]=dummy_start
	# 				unique_end_list[idx]=dummy_end

	# 			start_list_overlap,end_list_overlap = self.overlap_calc(unique_end_list,unique_start_list)
	# 		start_list = set(([tuple(i) for i in flat_list(unique_start_list)]))
	# 		end_list = set(([tuple(i) for i in flat_list(unique_end_list)]))
	# 		print(len(set((masked_start_values))))
	# 		print(len(set((masked_end_values))))
	# 		temp_total_list = list(start_list.intersection(end_list))


	# 		matrix_list = []
	# 		num_matrix_list = []
	# 		for k,dummy in enumerate(self.all_dict.values()):
	# 			if (k%1000)==0:
	# 				print(k)
	# 			num_matrix,matrix = dummy.construct_trans_and_number_matrix(temp_total_list)
	# 			matrix_list.append(matrix)
	# 			num_matrix_list.append(num_matrix)
	# 		transition_matrix = np.sum([matrix_list])
	# 		row_idx,column_idx,data = scipy.sparse.find(transition_matrix)
	# 		num_matrix = np.sum([num_matrix_list])
	# 		transition_matrix = TransMat((data,(row_idx,column_idx)),shape=(len(temp_total_list)
	# 			,len(temp_total_list)),total_list=temp_total_list
	# 			,lat_spacing = np.diff(self.lat_bins[:2])[0],lon_spacing=np.diff(self.lon_bins[:2])[0]
	# 			,time_step = self.time_step,number_data=num_matrix.data,traj_file_type=self.data_description)

	# 		print('shape of matrix is', transition_matrix.shape)

	# 	self.transition_matrix = transition_matrix
	# 	self.transition_matrix.save()

class BaseMat(scipy.sparse.csc_matrix):
	"""Base class for transition and correlation matrices, we include the timestep moniker for posterity. Time step for correlation matrices 
	means the L scaling """
	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None,traj_file_type=None,**kwargs):
		super(BaseMat,self).__init__(arg1, shape=shape,**kwargs)
		if total_list is not None:
			total_list = [list(x) for x in total_list]
			total_list = np.array(total_list)
			total_list[:,0][total_list[:,0]>180] = total_list[:,0][total_list[:,0]>180]-360
			assert total_list[:,0].min()>=-180
			assert total_list[:,0].max()<=180
			assert total_list[:,1].max()<=90
			assert total_list[:,1].min()>=-90
			total_list = total_list.tolist()
		self.total_list = total_list
		if type(lat_spacing)==np.ndarray:
			self.degree_bins = [float(lon_spacing),float(lat_spacing)]
		else:
			self.degree_bins = [lon_spacing,lat_spacing]
		self.traj_file_type = str(traj_file_type)

	@staticmethod
	def bins_generator(degree_bins):
		bins_lat = np.arange(-90,90.1,degree_bins[0]).tolist()
		bins_lon = np.arange(-180,180.1,degree_bins[1]).tolist()
		return (bins_lat,bins_lon)

	def new_sparse_matrix(self,data):
		row_idx,column_idx,dummy = scipy.sparse.find(self)
		return scipy.sparse.csc_matrix((data,(row_idx,column_idx)),shape=(len(self.total_list),len(self.total_list)))             

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
			A = self.__class__((data, indices, indptr), shape=self.shape,total_list=self.total_list,
				lat_spacing=self.degree_bins[1],lon_spacing=self.degree_bins[0],
				traj_file_type=self.traj_file_type,time_step=self.time_step,number_data=self.number_data)
		else:
			A = self.__class__((data, indices, indptr), shape=self.shape,total_list=self.total_list,
				lat_spacing=self.degree_bins[1],lon_spacing=self.degree_bins[0],
				traj_file_type=self.traj_file_type,variable_list=self.variable_list)
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
			return self.__class__((data, indices, indptr), shape=self.shape,
				total_list=self.total_list,lat_spacing=self.degree_bins[1],
				lon_spacing=self.degree_bins[0],traj_file_type=self.traj_file_type,
				time_step=self.time_step,number_data=self.number_data)
		else:
			return self.__class__((data, indices, indptr), shape=self.shape,
				total_list=self.total_list,lat_spacing=self.degree_bins[1],
				lon_spacing=self.degree_bins[0],traj_file_type=self.traj_file_type)



class TransMat(BaseMat):
	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
		,time_step=None,number_data=None,traj_file_type=None,rescale=False,**kwargs):
		super(TransMat,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing,traj_file_type=traj_file_type,**kwargs)
		self.time_step = time_step
		self.number_data = number_data
		if rescale:
			self.rescale()

	@classmethod
	def load_from_type(cls,lat_spacing=None,lon_spacing=None,time_step=None, traj_type='argo'):
		degree_bins = [np.array(float(lat_spacing)),np.array(float(lon_spacing))]
		file_name = TransMat.make_filename(traj_type=traj_type,degree_bins=degree_bins,time_step=time_step)
		return cls.load(file_name)

	@staticmethod
	def make_filename(traj_type=None,degree_bins=None,time_step=None):
		from transition_matrix.definitions import ROOT_DIR
		base = ROOT_DIR+'/output/'+traj_type+'/'
		if not os.path.isdir(base):
			os.mkdir(base)
		degree_bins = [float(degree_bins[0]),float(degree_bins[1])]
		return base+str(time_step)+'-'+str(degree_bins)+'.npz'

	@classmethod
	def load(cls,filename):
		loader = np.load(filename,allow_pickle=True)
		return cls((loader['data'], loader['indices'], loader['indptr']),shape=loader['shape'],
			total_list = loader['total_list'],lat_spacing=loader['lat_spacing'],
			lon_spacing=loader['lon_spacing'],time_step=loader['time_step'],
			number_data=loader['number_data'],traj_file_type=loader['traj_file_type'])

	def save(self,filename=False):
		if not filename:
			filename = self.make_filename(traj_type=self.traj_file_type,degree_bins=self.degree_bins,time_step=self.time_step)
		np.savez(filename, data=self.data, indices=self.indices,indptr=self.indptr, 
		shape=self.shape, total_list=self.total_list,lat_spacing=self.degree_bins[0],
		lon_spacing=self.degree_bins[1],time_step=self.time_step,number_data=self.number_data,
		traj_file_type=self.traj_file_type)



	def get_direction_matrix(self):
		"""

		notes: this could be made faster by looping through unique values of lat and lon and assigning intelligently
		"""

		lat_list, lon_list = zip(*self.total_list)
		lat_list = np.array(lat_list)
		lon_list = np.array(lon_list)
		pos_max = 180/self.degree_bins[1] #this is the maximum number of bins possible
		output_ns_list = []
		output_ew_list = []
		for (token_lat,token_lon) in self.total_list:
			token_ns = (token_lat-np.array(lat_list))/self.degree_bins[0]
			token_ew = (token_lon-np.array(lon_list))/self.degree_bins[1]
			token_ew[token_ew>pos_max]=token_ew[token_ew>pos_max]-2*pos_max #the equivalent of saying -360 degrees
			token_ew[token_ew<-pos_max]=token_ew[token_ew<-pos_max]+2*pos_max #the equivalent of saying +360 degrees
			output_ns_list.append(token_ns)
			output_ew_list.append(token_ew)
		self.east_west = np.array(output_ew_list)
		self.north_south = np.array(output_ns_list)
		assert (self.east_west<=180/self.degree_bins[1]).all()
		assert (self.east_west>=-180/self.degree_bins[1]).all()
		assert (self.north_south>=-180/self.degree_bins[0]).all()
		assert (self.north_south<=180/self.degree_bins[0]).all()

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

	def reduce_resolution(self,new_degree_bins):
		lat_mult = new_degree_bins[0]/self.degree_bins[0]
		lon_mult = new_degree_bins[1]/self.degree_bins[1]

		reduced_res_lat_bins,reduced_res_lon_bins = self.bins_generator(new_degree_bins)
		lat_bins,lon_bins = zip(*self.total_list)
		lat_idx = np.digitize(lat_bins,reduced_res_lat_bins)
		lon_idx = np.digitize(lon_bins,reduced_res_lon_bins)
		reduced_res_bins=[(reduced_res_lat_bins[i],reduced_res_lon_bins[j]) for i,j in zip(lat_idx,lon_idx)]
		reduced_res_total_list = list(set((reduced_res_bins)))
		translation_list = [reduced_res_total_list.index(x) for x in reduced_res_bins]
		check = [(np.array(translation_list)==x).sum()<=(lat_mult*lon_mult) for x in range(len(reduced_res_total_list))]
		assert all(check)
		translation_dict = dict(zip(range(len(self.total_list)),translation_list))
		
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
			out_data.append(data_dummy.sum()*1/(lat_mult*lon_mult))
			out_col.append(col_dummy)
			out_row.append(row_dummy)
		mat_dim = len(reduced_res_total_list)
		TransMat((out_data,(out_row,out_col
			)),shape=(mat_dim,mat_dim),total_list=reduced_res_total_list,
			lat_spacing=new_degree_bins[0],lon_spacing=new_degree_bins[1],time_step=self.time_step,number_data=None,
			traj_file_type=self.traj_file_type,rescale=True)		



	def save_trans_matrix_to_json(self,foldername):
		for column in range(self.shape[1]):
			print('there are ',(self.shape[1]-column),'columns remaining')
			p_lat,p_lon = tuple(self.total_list[column])
			data = self[:,column].data
			lat,lon = zip(*[tuple(self.total_list[x]) for x in self[:,column].indices.tolist()])
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