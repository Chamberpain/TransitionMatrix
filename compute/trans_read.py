from __future__ import print_function
from compute_utilities.list_utilities import find_nearest,flat_list
import numpy as np
import matplotlib.pyplot as plt
from sets import Set
from collections import Counter
import scipy.sparse
import json
import scipy.sparse.linalg


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
		self.start_bin_list = self.instance.prof.pos.return_pos_bins(lat_list,lon_list,index_values=start_pos_indexes,index_return=False)
		self.end_bin_list = self.instance.prof.pos.return_pos_bins(lat_list,lon_list,index_values=end_pos_indexes,index_return=False)


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

		def get_index_of_decorrelated(self,time_delta=30):
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
				diff_check = time_delta/6.
			seconds_to_days = 1/(3600*24.)
			diff_list = [(_-self.instance.prof.date._list[start_index]).total_seconds()*seconds_to_days for _ in self.instance.prof.date._list[(start_index+1):]]
			if not diff_list:
				return None
			closest_diff = find_nearest(diff_list,time_delta)
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

class TransitionClassAgg(object):
	def __init__(self,lat_spacing=2,lon_spacing=2,time_step=60,drifter_class=None):
		self.lat_bins = np.arange(-90,90.1,lat_spacing)
		self.lon_bins = np.arange(-180,180.1,lon_spacing)
		self.time_step = time_step
		self.class_dict = drifter_class.all_dict
		self.drifter_type = drifter_class.data_description 
		dummy_list = [TransitionCalc(_,self.lat_bins,self.lon_bins,time_step=time_step) for _ in self.class_dict.values()]
		self.class_dict = dict(zip(self.class_dict.keys(),dummy_list))
		# self.create_total_list()
		self.index_transcalc_classes()

	def overlap_calc(self,unique_end_list,unique_start_list):
		"""
		Take the unique start and end lists and figure out where they dont overlap (which means that the transition matrix would
		have cells that have no start or exit)
		"""
		unique_end_list = [tuple(_) for _ in flat_list(unique_end_list)]
		unique_start_list = [tuple(_) for _ in flat_list(unique_start_list)]
		unique_end_list = Set(unique_end_list)
		unique_start_list = Set(unique_start_list)
		start_list_overlap = list(unique_start_list.difference(unique_end_list))
		end_list_overlap = list(unique_end_list.difference(unique_start_list))
		return (start_list_overlap, end_list_overlap)




	def overlap_iterate(self,unique_end_list,unique_start_list,start_list_overlap,end_list_overlap):
		masked_start_values = []
		masked_end_values = []

		while (end_list_overlap!=[])|(start_list_overlap!=[]):
			masked_start_values += start_list_overlap
			masked_end_values += end_list_overlap
			masked_start_values = list(Set(masked_start_values))
			masked_end_values = list(Set(masked_end_values))
			print(len(start_list_overlap))
			print(len(end_list_overlap))
			print(len(masked_start_values))
			print(len(masked_end_values))

			id_list = [i[1].instance.meta.id for i in self.class_dict.iteritems()]
			bool_list = [i[1].is_bin_contained(masked_start_values,masked_end_values).any() for i in self.class_dict.iteritems()]
			idx_to_fix = np.where(bool_list)
			for idx in idx_to_fix[0]:
				dummy_id = id_list[idx]
				dummy_start,dummy_end = self.class_dict[dummy_id].return_masked_start_and_end_bins(masked_start_values=masked_start_values,masked_end_values=masked_end_values)
				unique_start_list[idx]=dummy_start
				unique_end_list[idx]=dummy_end

			start_list_overlap,end_list_overlap = self.overlap_calc(unique_end_list,unique_start_list)
		start_list = Set([tuple(i) for i in flat_list(unique_start_list)])
		end_list = Set([tuple(i) for i in flat_list(unique_end_list)])
		return (list(start_list.intersection(end_list)),masked_start_values,masked_end_values)

	def index_transcalc_classes(self):
		masked_end_values = []
		masked_start_values = []
		continue_flag = True

		"""this creates the total list of transitions, because of edge cases where a float trajectory may have a departure from a cell
		and no arrival, we have to iteratively mask our results"""

		unique_start_list = []
		unique_end_list = []
		unique_start_list = []
		unique_end_list = []
		id_list = []
		for dummy in self.class_dict.itervalues():
			"""
			Loop through all items in the class dictionary. Grab the masked start and end bins and make a nested list			
			"""
			assert len(dummy.start_bin_list)==len(dummy.end_bin_list)
			dummy_start,dummy_end = dummy.return_masked_start_and_end_bins()
			unique_start_list.append(dummy_start)
			unique_end_list.append(dummy_end)
		
		start_list_remove,end_list_remove = self.overlap_calc(unique_end_list,unique_start_list)
		temp_total_list,masked_start_values,masked_end_values = self.overlap_iterate(unique_end_list,unique_start_list,start_list_remove,end_list_remove)



		def create_trans_matrix_from_list(temp_total_list):
			matrix_list = []
			num_matrix_list = []
			for k,dummy in enumerate(self.class_dict.values()):
				if (k%1000)==0:
					print(k)
				num_matrix,matrix = dummy.construct_trans_and_number_matrix(temp_total_list)
				matrix_list.append(matrix)
				num_matrix_list.append(num_matrix)
			transition_matrix = np.sum([matrix_list])
			row_idx,column_idx,data = scipy.sparse.find(transition_matrix)
			num_matrix = np.sum([num_matrix_list])
			transition_matrix = TransMat((data,(row_idx,column_idx)),shape=(len(temp_total_list)
				,len(temp_total_list)),total_list=temp_total_list
				,lat_spacing = np.diff(self.lat_bins[:2])[0],lon_spacing=np.diff(self.lon_bins[:2])[0]
				,time_step = self.time_step,number_data=num_matrix.data,traj_file_type=self.drifter_type)
			return (transition_matrix,num_matrix)

		transition_matrix,num_matrix = create_trans_matrix_from_list(temp_total_list)
		checksum=10**-5


		while transition_matrix.matrix_eig_check(bool_return=True,checksum=checksum):
			eig_vals,eig_vecs = scipy.sparse.linalg.eigs(transition_matrix,k=50)
			problem_idx = np.where((eig_vecs>checksum).sum(axis=0)<=3)[0]
			for idx in problem_idx.tolist():
				masked_end_values += [temp_total_list[i] for i in np.where(eig_vecs[:,idx]>checksum)[0]]
				print([temp_total_list[i] for i in np.where(eig_vecs[:,idx]>checksum)[0]])
				masked_start_values += [temp_total_list[i] for i in np.where(eig_vecs[:,idx]>checksum)[0]]
				print([temp_total_list[i] for i in np.where(eig_vecs[:,idx]>checksum)[0]])
			# temp_total_list,masked_start_values,masked_end_values,unique_start_list,unique_end_list = self.overlap_iterate(unique_start_list,unique_end_list,masked_start_values,masked_end_values)
			
			start_list_overlap = masked_start_values
			end_list_overlap = masked_end_values

			while (end_list_overlap!=[])|(start_list_overlap!=[]):
				masked_start_values += start_list_overlap
				masked_end_values += end_list_overlap
				masked_start_values = list(Set(masked_start_values))
				masked_end_values = list(Set(masked_end_values))
				print('length of the start list is', len(start_list_overlap))
				print( 'length of the end list is', len(end_list_overlap))
				print('length of the masked start list is', len(masked_start_values))
				print('length of the masked end list is', len(masked_end_values))

				id_list = [i[1].instance.meta.id for i in self.class_dict.iteritems()]
				bool_list = [i[1].is_bin_contained(masked_start_values,masked_end_values).any() for i in self.class_dict.iteritems()]
				idx_to_fix = np.where(bool_list)
				for idx in idx_to_fix[0]:
					dummy_id = id_list[idx]
					dummy_start,dummy_end = self.class_dict[dummy_id].return_masked_start_and_end_bins(masked_start_values=masked_start_values,masked_end_values=masked_end_values)
					unique_start_list[idx]=dummy_start
					unique_end_list[idx]=dummy_end

				start_list_overlap,end_list_overlap = self.overlap_calc(unique_end_list,unique_start_list)
			start_list = Set([tuple(i) for i in flat_list(unique_start_list)])
			end_list = Set([tuple(i) for i in flat_list(unique_end_list)])
			print(len(Set(masked_start_values)))
			print(len(Set(masked_end_values)))
			temp_total_list = list(start_list.intersection(end_list))


			matrix_list = []
			num_matrix_list = []
			for k,dummy in enumerate(self.class_dict.values()):
				if (k%1000)==0:
					print(k)
				num_matrix,matrix = dummy.construct_trans_and_number_matrix(temp_total_list)
				matrix_list.append(matrix)
				num_matrix_list.append(num_matrix)
			transition_matrix = np.sum([matrix_list])
			row_idx,column_idx,data = scipy.sparse.find(transition_matrix)
			num_matrix = np.sum([num_matrix_list])
			transition_matrix = TransMat((data,(row_idx,column_idx)),shape=(len(temp_total_list)
				,len(temp_total_list)),total_list=temp_total_list
				,lat_spacing = np.diff(self.lat_bins[:2])[0],lon_spacing=np.diff(self.lon_bins[:2])[0]
				,time_step = self.time_step,number_data=num_matrix.data,traj_file_type=self.drifter_type)

			print('shape of matrix is', transition_matrix.shape)

		self.transition_matrix = transition_matrix
		self.transition_matrix.save()

class BaseMat(scipy.sparse.csc_matrix):
	"""Base class for transition and correlation matrices, we include the timestep moniker for posterity. Time step for correlation matrices 
	means the L scaling """
	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None,traj_file_type=None):
		super(BaseMat,self).__init__(arg1, shape=shape)
		self.total_list = total_list
		self.degree_bins = [lon_spacing,lat_spacing]
		self.traj_file_type = traj_file_type

	@staticmethod
	def bins_generator(degree_bins):
		bins_lat = np.arange(-90,90.1,degree_bins[0]).tolist()
		bins_lon = np.arange(-180,180.1,degree_bins[1]).tolist()
		return (bins_lat,bins_lon)

	def new_sparse_matrix(self,data):
		row_idx,column_idx,dummy = scipy.sparse.find(self)
		return scipy.sparse.csc_matrix((data,(row_idx,column_idx)),shape=(len(self.total_list),len(self.total_list)))             


class TransMat(BaseMat):
	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
		,time_step=None,number_data=None,traj_file_type=None,rescale=True):
		super(TransMat,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing,traj_file_type=None)
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
		base = ROOT_DIR+'/output/'
		degree_bins = [float(degree_bins[0]),float(degree_bins[1])]
		return base+traj_type+'/'+str(time_step)+'-'+str(degree_bins)+'.npz'

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


	def rescale(self,checksum=10**-3):
		div_array = np.abs(self.sum(axis=0)).tolist()[0]
		row_idx,column_idx,data = scipy.sparse.find(self)
		col_count = []
		for col in column_idx:
			col_count.append(float(div_array[col]))
		self.data = np.array(data)/np.array(col_count)
		self.matrix_column_check(checksum)

	def matrix_eig_check(self,checksum=10**-5,bool_return=False):
		eig_vals,eig_vecs = scipy.sparse.linalg.eigs(self,k=30)
		if bool_return:
			return bool(np.where((eig_vecs>checksum).sum(axis=0)<=3)[0].tolist())
		else:
			assert not np.where((eig_vecs>checksum).sum(axis=0)<=3)[0].tolist()

	def matrix_column_check(self,checksum):
		assert (np.abs(self.sum(axis=0)-1)<checksum).all()

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