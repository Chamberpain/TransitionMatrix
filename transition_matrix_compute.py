import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os
import glob
import pickle
import datetime
import scipy.sparse
import scipy.optimize
from itertools import groupby  
import random
import math
import copy

def find_nearest(items, pivot):
	return min(items, key=lambda x: abs(x - pivot))

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csc(filename):
    loader = np.load(filename)
    return scipy.sparse.csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

__file__ = os.getenv("HOME")+'/Projects/transition_matrix/transition_matrix_compute.py' #this is a hack to get this thing to run in terminal

class argo_traj_data:
	def __init__(self,degree_bin_lat=2,degree_bin_lon=2,date_span_limit=60,traj_file_type='Argo'):
		self.__traj_file_type = traj_file_type
		if self.__traj_file_type=='Argo':
			self.__base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/'	#this is a total hack to easily change the code to run sose particles
			print 'I am loading Argo'
		elif self.__traj_file_type=='SOSE':
			self.__base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose/'
			print 'I am loading SOSE data'

		print 'I have started argo traj data'
		self.__degree_bins = (degree_bin_lat,degree_bin_lon)
		self.__date_span_limit = date_span_limit
		self.__bins_lat = np.arange(-90,90.1,self.__degree_bins[0]).tolist()
		self.__bins_lon = np.arange(-180,180.1,self.__degree_bins[1]).tolist()
		self.__end_bin_string = 'end bin '+str(self.__date_span_limit)+' day' # we compute the transition df for many different date spans, here is where we find that column

		if 180.0 not in self.__bins_lon:
			print '180 is not divisable by the degree bins chosen'		#need to add this logic for the degree bin choices that do not end at 180.
			raise
		self.__X,self.__Y = np.meshgrid(self.__bins_lon,self.__bins_lat)    #these are necessary private variables for the plotting routines
		self.load_trajectory_df(pd.read_pickle(self.__base_file+'all_argo_traj.pickle'))
		try:
			self.__transition_df = self.load_transition_df_and_total_list(pd.read_pickle(self.base_file+'transition_df_degree_bins_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])+'_modified.pickle'))
		except IOError: #this is the case that the file could not load
			print 'i could not load the transition df, I am recompiling with degree step size', self.degree_bins,''
			print 'file was '+self.base_file+'transition_df_degree_bins_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])+'_modified.pickle'
			self.recompile_transition_df()
		try: # try to load the transition matrix 
			self.transition_matrix = load_sparse_csc(self.base_file+'transition_matrix/transition_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
		except IOError: # if the matrix cannot load, recompile
			print 'i could not load the transition matrix, I am recompiling with degree step size', self.degree_bins,' and time step ',self.date_span_limit
			self.recompile_transition_matrix()
		assert (np.abs(self.transition_matrix.sum(axis=0)-1)<10**-10).all()
		assert (self.transition_matrix>0).data.all() 
		assert len(self.total_list)==self.transition_matrix.shape[0]
		print 'Transition matrix passed all necessary tests. Initial load complete'


	def load_trajectory_df(self,dataframe):
		if self.__traj_file_type=='SOSE':
			dataframe=dataframe[dataframe.Lat<-36] # this cuts off floats that wander north of 35
		dataframe = dataframe.sort_values(by=['Cruise','Date'])
		dataframe['bins_lat'] = pd.cut(dataframe.Lat,bins = self.__bins_lat,labels=self.__bins_lat[:-1])
		dataframe['bins_lon'] = pd.cut(dataframe.Lon,bins = self.__bins_lon,labels=self.__bins_lon[:-1])
		dataframe['bin_index'] = zip(dataframe['bins_lat'].values,dataframe['bins_lon'].values)
		dataframe = dataframe.reset_index(drop=True)
		#implement tests of the dataset so that all values are within the known domain
		assert dataframe.Lon.min() >= -180
		assert dataframe.Lon.max() <= 180
		assert dataframe.Lat.max() <= 90
		assert dataframe.Lat.min() >=-90
		print 'Trajectory dataframe passed necessary tests'
		self.__trajectory_df = dataframe

	def load_transition_df_and_total_list(self,dataframe,dump=False):
		assert ~np.isnan(dataframe['start bin'].unique().tolist()).any()
		dataframe = dataframe.dropna(subset=[self.__end_bin_string]) 
		while len(dataframe)!=len(dataframe[dataframe[self.__end_bin_string].isin(dataframe['start bin'].unique())]):	#this eliminates end bin strings that do not have a corresponding start bin which would create holes in the transition matrix
			print 'Removing end bin strings that do not have a start bin string associated'
			dataframe = dataframe[dataframe[self.__end_bin_string].isin(dataframe['start bin'].unique())]
		print 'Transition dataframe passed all necessary tests'
		token = dataframe[dataframe['start bin']!=dataframe[self.__end_bin_string]].drop_duplicates().groupby('start bin').count() 
		total_list = [list(x) for x in token[token[self.__end_bin_string]>0].index.unique().values.tolist()] # this will make a unique "total list" for every transition matrix, but is necessary so that we dont get "holes" in the transition matrix
		token = [tuple(x) for x in total_list]
		dataframe = dataframe[dataframe[self.__end_bin_string].isin(token)&dataframe['start bin'].isin(token)]

		self.__transition_df = dataframe
		self.__total_list = total_list
		if dump:
			self.__transition_df.to_pickle(self.__base_file+'transition_df_degree_bins_'+str(self.__degree_bins[0])+'_'+str(self.__degree_bins[1])+'_modified.pickle')


	def recompile_transition_df(self,dump=True):
		"""
		from self.df, calculate the dataframe that is used to create the transition matrix

		input: dump - logical variable that is used to determine whether to save the dataframe
		"""
		cruise_list = []
		start_bin_list = []
		final_bin_list = []
		end_bin_list = []
		date_span_list = []
		start_date_list = []
		position_type_list = []
		time_delta_list = np.arange(20,300,20)
		k = len(self.df.Cruise.unique())
		for n,cruise in enumerate(self.df.Cruise.unique()):
			print 'cruise is ',cruise,'there are ',k-n,' cruises remaining'
			print 'start bin list is ',len(start_bin_list),' long'
			mask = self.df.Cruise==cruise  		#pick out the cruise data from the df
			df_holder = self.df[mask]
			time_lag = 30	#we assume a decorrelation timescale of 30 days
			time_bins = np.arange(-0.001,(df_holder.Date.max()-df_holder.Date.min()).days,time_lag).tolist()
			df_holder['Time From Deployment'] = [(dummy-df_holder.Date.min()).days + (dummy-df_holder.Date.min()).seconds/float(3600*24) for dummy in df_holder.Date]
			#time from deployment is calculated like this to have the fractional day component
			assert (df_holder['Time From Deployment'].diff().tail(len(df_holder)-1)>0).all() # test that these are all positive and non zero

			max_date = df_holder.Date.max()
			df_holder['time_bins'] = pd.cut(df_holder['Time From Deployment'],bins = time_bins,labels=time_bins[:-1])
			#cut df_holder into a series of time bins, then drop duplicate time bins and only keep the first, this enforces the decorrelation criteria
			for row in df_holder.dropna(subset=['time_bins']).drop_duplicates(subset=['time_bins'],keep='first').iterrows():
				dummy, row = row
				cruise_list.append(row['Cruise']) # record cruise information
				start_date_list.append(row['Date']) # record date information
				start_bin_list.append(row['bin_index']) # record which bin was started in  
				position_type_list.append(row['position type']) # record positioning information
				location_tuple = []
				for time_addition in [datetime.timedelta(days=x) for x in time_delta_list]: # for all the time additions that we need to calculate
					final_date = row['Date']+time_addition # final date is initial date plus the addition
					if final_date>max_date:
						location_tuple.append(np.nan) #if the summed date is greater than the record, do not include in any transition matrix calculations
					else:
						final_dataframe_date = find_nearest(df_holder.Date,final_date) #find the nearest record to the final date
						if abs((final_dataframe_date-final_date).days)>30: 
							location_tuple.append(np.nan) #if this nearest record is greater than 30 days away from positioning, exclude
						else:
							location_tuple.append(df_holder[df_holder.Date==final_dataframe_date].bin_index.values[0]) # find the nearest date and record the bin index
				final_bin_list.append(tuple(location_tuple))
				if any(len(lst) != len(position_type_list) for lst in [start_date_list, start_bin_list, final_bin_list]):
					print 'position list length ,',len(position_type_list)
					print 'start date list ',len(start_date_list)
					print 'start bin list ',len(start_bin_list)
					raise
		df_dict = {}
		df_dict['Cruise']=cruise_list
		df_dict['position type']=position_type_list
		df_dict['date'] = start_date_list
		df_dict['start bin'] = start_bin_list
		for time_delta,bin_list in zip(time_delta_list,zip(*final_bin_list)):
			bins_string = 'end bin '+str(time_delta)+' day'
			df_dict[bins_string]=bin_list
		df_dict['end bin'] = final_bin_list
		dataframe = pd.DataFrame(df_dict)
		if dump:
			dataframe.to_pickle(self.base_file+'transition_df_degree_bins_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])+'_raw.pickle')
		load_transition_df_and_total_list(dataframe,dump=dump)

	def load_transition_and_number_matrix(self,t_matrix,n_matrix,dump=False):
		assert (np.abs(matrix.todense().sum(axis=0)-1)<0.01).all() #this is in the dump subroutine for the case that it is recompiled for the data withholding experiment.
		self.__transition_matrix = t_matrix
		self.__number_matrix = n_matrix
		if dump:
			save_sparse_csc(self.__base_file+'transition_matrix/transition_matrix_degree_bins_'+str(self.__degree_bins)+'_time_step_'+str(self.__date_span_limit)+'.npz',self.__transition_matrix)
			save_sparse_csc(self.base_file+'transition_matrix/number_matrix_degree_bins_'+str(self.__degree_bins)+'_time_step_'+str(self.__date_span_limit)+'.npz',self.__number_matrix)


	def column_compute(self,ii,total_list=self.__total_list):
#this algorithm has problems with low data density because the frame does not drop na values for the self.end_bin_string
		token_row_list = []
		token_column_list = []
		token_num_list = []
		token_num_test_list = []
		token_data_list = []

		ii_index = total_list.index(list(ii))
		frame = self.__transition_df[self.__transition_df['start bin']==ii]	# data from of floats that start in looped grid cell
		frame_cut = frame[frame[self.end_bin_string]!=ii].dropna(subset=[self.end_bin_string]) #this is all floats that go out of the looped grid cell
		if frame_cut.empty:
			print 'the frame cut was empty'
			return (token_num_list, token_data_list, token_row_list,token_column_list)
		token_row_list = []
		token_column_list = []
		token_num_list = []
		token_num_test_list = []
		token_data_list = []
		token_row_list.append(ii_index)	#compute the on diagonal elements
		token_column_list.append(ii_index)	#compute the on diagonal elemnts
		data = (len(frame)-len(frame_cut))/float(len(frame)) # this is the percentage of floats that stay in the looped grid cell
		token_num_list.append(len(frame)-len(frame_cut)) # this is where we save the data density of every cell
		token_data_list.append(data)
		for qq in frame_cut[self.end_bin_string].unique():
			qq_index = total_list.index(list(qq))
			token_row_list.append(qq_index)	#these will be the off diagonal elements
			token_column_list.append(ii_index)
			data = (len(frame_cut[frame_cut[self.end_bin_string]==qq]))/float(len(frame))
			token_data_list.append(data)
			token_num_list.append(len(frame_cut[frame_cut[self.end_bin_string]==qq])) # this is where we save the data density of every cell
		assert abs(sum(token_data_list)-1)<0.01	#ensure that all columns scale to 1
		assert ~np.isnan(token_data_list).any()
		assert (np.array(token_data_list)<=1).all()
		assert (np.array(token_data_list)>=0).all()
		assert sum(token_num_list)==len(frame)
		return (token_num_list, token_data_list, token_row_list,token_column_list)

	def recompile_transition_matrix_and_number_matrix(self,dump=True,plot=False):
		"""
		Recompiles transition matrix from __transition_df based on set timestep
		"""
		num_list = []
		data_list = []
		row_list = []
		column_list = []

		k = len(self.__total_list)	# sets up the total number of loops
		for n,index in enumerate(self.__total_list): #loop through all values of total_list
			print 'made it through ',n,' bins. ',(k-n),' remaining' 
			dummy_num_list, dummy_data_list,dummy_row_list,dummy_column_list = self.column_compute(tuple(index))
			num_list += dummy_num_list
			data_list += dummy_data_list
			row_list += dummy_row_list
			column_list += dummy_column_list
			assert len(column_list)==len(row_list)
			assert len(column_list)==len(data_list)
			assert len(column_list)==len(num_list)
		assert sum(num_list)==len(self.__transition_df[self.__transition_df['start bin'].isin([tuple(self.__total_list[x]) for x in np.unique(column_list)])])
		transition_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(self.__total_list),len(self.__total_list)))
		number_matrix = scipy.sparse.csc_matrix((num_list,(row_list,column_list)),shape=(len(self.__total_list),len(self.__total_list)))

		# self.transition_matrix = self.add_noise(self.transition_matrix)

		full_matrix_list = range(transition_matrix.shape[0])
		eig_vals,eig_vecs = scipy.linalg.eig(transition_matrix.todense())
		orig_len = len(transition_matrix.todense())
		idx = np.where((eig_vecs>0).sum(axis=0)<=2) #find the indexes of all eigenvectors with less than 2 entries
		index_list = []
		for index in idx[0]:
			eig_vec = eig_vecs[:,index]		
			index_list+=np.where(eig_vec>0)[0].tolist() # identify the specific grid boxes where the eigen values are 1 
		index_list = np.unique(index_list) #these may be duplicated so take unique values
		if index_list.tolist():	#if there is anything in the index list 
			not_in_index_list = np.array(full_matrix_list)[~np.isin(full_matrix_list,index_list)]

			row_list = transition_matrix.tocoo().row.tolist()
			column_list = transition_matrix.tocoo().col.tolist()
			data_list = transition_matrix.tocoo().data.tolist()
			number_list = number_matrix.tocoo().data.tolist()

			row_truth = np.isin(row_list,not_in_index_list) #these are all rows that are not at the bad locations
			column_truth = np.isin(column_list,not_in_index_list) # these are all columns that are not at the bad locations
			mask = row_truth&column_truth # create mask of rows and columns in the proper locations

			row_list = np.array(row_list)[mask].tolist()
			column_list = np.array(column_list)[mask].tolist()
			data_list = np.array(data_list)[mask].tolist()
			number_list = np.array(number_list)[mask].tolist()

			if plot:
				disfunction_df = self.__transition_df[self.__transition_df[self.end_bin_string].isin([tuple(x) for x in np.array(self.total_list)[index_list]])]
				for index in index_list:
					for FLOAT in disfunction_df[disfunction_df[self.end_bin_string]==tuple(self.total_list[index])].Cruise.unique()[2:5]:
						token = self.df[self.df.Cruise==FLOAT]
						plt.plot(token.Lat.tolist(),token.Lon.tolist())
						lat,lon = self.total_list[index]
						plt.plot(lat,lon,'r*',markersize=15)
						plt.show()

			transition_df_holder.loc[self.__transition_df[self.end_bin_string].isin([tuple(x) for x in np.array(self.__total_list)[index_list]]),self.___end_bin_string]=np.nan # make all of the end string values nan in these locations
			old_total_list = copy.deepcopy(self.__total_list)
			self.load_transition_df_and_total_list(transition_df_holder,dump=dump)

			dummy, column_redo = np.where(transition_matrix[index_list,:].todense()!=0)	#find all of the row locations where the transition matrix is non zero
			column_redo = np.unique(column_redo)
			column_redo = column_redo[~np.isin(column_redo,index_list)] # no need to redo columns that are in the index list because they will be rejected from matrix

			k = len(column_redo)
			for n,index in enumerate(column_redo):
				if n%10==0:
					print 'made it through ',n,' bins. ',(k-n),' remaining'
				dummy_num_list, dummy_data_list,dummy_row_list,dummy_column_list = self.column_compute(tuple(old_total_list[index]),total_list=old_total_list)
				data_list += dummy_data_list
				row_list += dummy_row_list
				column_list += dummy_column_list
				number_list += dummy_num_list

				assert len(column_list)==len(row_list)
				assert len(column_list)==len(data_list)
				assert len(column_list)==len(num_list)

			new_col_list = [new_total_list.index(self.__total_list[x]) for x in column_list]
			new_row_list = [new_total_list.index(self.__total_list[x]) for x in row_list]
			
			transition_matrix = scipy.sparse.csc_matrix((data_list,(new_row_list,new_col_list)),shape=(len(self.__total_list),len(self.__total_list)))
			number_matrix = scipy.sparse.csc_matrix((num_list,(new_row_list,new_col_list)),shape=(len(self.__total_list),len(self.__total_list)))
		self.load_transition_and_number_matrix(transition_matrix,number_matrix,dump=dump)


	def add_noise(self,matrix,noise=0.05):
		"""
		Adds guassian noise to the transition matrix
		The appropriate level of noise has not been worked out and is kind of ad hock
		"""
		print 'adding matrix noise'
		east_west,north_south = self.get_direction_matrix()
		direction_mat = -abs(east_west)**2-abs(north_south)**2
		direction_mat = noise*np.exp(direction_mat)
		direction_mat[direction_mat<noise*np.exp(-20)]=0
		matrix += scipy.sparse.csc_matrix(direction_mat)
		return self.rescale_matrix(matrix)

	def rescale_matrix(self,matrix_):
		print 'rescaling the matrix'
		mat_sum = matrix_.todense().sum(axis=0)
		scalefactor,dummy = np.meshgrid(1/mat_sum,1/mat_sum)
		return scipy.sparse.csc_matrix(mat_sum*scalefactor)

	def delete_w(self):
		"""
		deletes all w matrices in data directory
		"""
		for w in glob.glob(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_*.npz'):
			os.remove(w)

	def load_w(self,number,dump=True):
		"""
		recursively multiplies the transition matrix by itself 
		"""
		print 'in w matrix, current number is ',number
		try:
			self.w = load_sparse_csc(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz')
			assert self.transition_matrix.shape==self.w.shape
 			assert ~np.isnan(self.w).any()
			assert (np.array(self.w)<=1).all()
			assert (np.array(self.w)>=0).all()
			assert (self.w.todense().sum(axis=0)-1<0.01).all()
			print 'w matrix successfully loaded'
		except IOError:
			print 'w matrix could not be loaded and will be recompiled'
			if number == 0:
				self.w = self.transition_matrix 
			else:
				self.load_w(number-1,dump=True)		# recursion to get to the first saved transition matrix 
				self.w = self.w.dot(self.transition_matrix)		# advance the transition matrix once
			assert self.transition_matrix.shape==self.w.shape
 			assert ~np.isnan(self.w).any()
			assert (np.array(self.w)<=1).all()
			assert (np.array(self.w)>=0).all()
			assert (self.w.todense().sum(axis=0)-1<0.01).all()
			if dump:
				save_sparse_csc(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz',self.w)	#and save