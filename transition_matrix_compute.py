import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import shiftgrid
import sys,os
import glob
import pickle
import datetime
import scipy.sparse
import scipy.optimize
from itertools import groupby  
import random
import math
      
def find_nearest(items, pivot):
	return min(items, key=lambda x: abs(x - pivot))

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape']).T #this transpose is very important because otherwise these transition matrices load wrong and send flow the wrong way.

__file__ = os.getenv("HOME")+'/Projects/transition_matrix/transition_matrix_compute.py' #this is a hack to get this thing to run in terminal
bad_cruise_list = ['1900857','1900978','1901149','1901552','3900743','3901062','4901246','5903883','6900938','6901004']

class argo_traj_data:
	def __init__(self,degree_bins=1,date_span_limit=60):
		print 'I have started argo traj data'
		self.base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/'
		self.degree_bins = int(degree_bins)
		self.date_span_limit = date_span_limit
		self.bins_lat = np.arange(-90,90.1,self.degree_bins).tolist()
		self.bins_lon = np.arange(-180,180.1,self.degree_bins).tolist()
		if 180.0 not in self.bins_lon:
			print '180 is not divisable by the degree bins chosen'		#need to add this logic for the degree bin choices that do not end at 180.
			raise
		self.X,self.Y = np.meshgrid(self.bins_lon,self.bins_lat)
		self.df = pd.read_pickle(self.base_file+'all_argo_traj.pickle').sort_values(by=['Cruise','Date'])
		self.df = self.df[~self.df.Cruise.isin(bad_cruise_list)]
		self.df['bins_lat'] = pd.cut(self.df.Lat,bins = self.bins_lat,labels=self.bins_lat[:-1])
		self.df['bins_lon'] = pd.cut(self.df.Lon,bins = self.bins_lon,labels=self.bins_lon[:-1])
		self.df = self.df.reset_index(drop=True)
		self.df['bin_index'] = zip(self.df['bins_lat'].values,self.df['bins_lon'].values)

		#implement tests of the dataset so that all values are within the known domain
		assert self.df.Lon.min() >= -180
		assert self.df.Lon.max() <= 180
		assert self.df.Lat.max() <= 90
		assert self.df.Lat.min() >=-90

		try:
			self.df_transition = pd.read_pickle(self.base_file+'transition_df_degree_bins_'+str(self.degree_bins)+'.pickle')
			assert (self.df_transition['date span']>=0).all() #require time to go in the right direction

		except IOError: #this is the case that the file could not load
			print 'i could not load the transition df, I am recompiling with degree step size', self.degree_bins,''
			self.recompile_transition_df()
		self.end_bin_string = 'end bin '+str(self.date_span_limit)+' day' # we compute the transition df for many different date spans, here is where we find that column
		self.df_transition = self.df_transition.dropna(subset=[self.end_bin_string])
		self.identify_problems_df_transition()
		while len(self.df_transition)!=len(self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]):	#need to loop this
			self.df_transition = self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]
		self.total_list = [list(x) for x in self.df_transition['start bin'].unique()] 

		try: # try to load the transition matrix
			self.transition_matrix = load_sparse_csr(self.base_file+'transition_matrix/transition_matrix_degree_bins_'+str(degree_bins)+'_time_step_'+str(date_span_limit)+'.npz')
		except IOError: # if the matrix cannot load, recompile
			print 'i could not load the transition matrix, I am recompiling with degree step size', self.degree_bins,' and time step ',self.date_span_limit
			self.recompile_transition_matrix()

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
			max_date = df_holder.Date.max()
			df_holder['diff']= (df_holder.bins_lat.diff().apply(lambda x: 1 if abs(x) else 0)+df_holder.bins_lon.diff().apply(lambda x: 1 if abs(x) else 0)).cumsum()
			position_type = df_holder['position type'].values[0]
			group_library = df_holder.groupby('diff').groups
			diff_group = np.sort(df_holder.groupby('diff').groups.keys()).tolist()
			diff_group.pop() #this is the last grid box the float moved into and we can assume it died
			for g in diff_group:
				df_token = df_holder[df_holder.index.isin(group_library[g])]
				start_date = df_token.Date.min()
				cruise_list.append(cruise)
				start_date_list.append(start_date)
				start_bin_list.append(df_token.bin_index.values[0])
				position_type_list.append(position_type)
				location_tuple = []
				for time_addition in [datetime.timedelta(days=x) for x in time_delta_list]:
					final_date = start_date+time_addition
					if final_date>max_date:
						location_tuple.append(np.nan)
					else:
						location_tuple.append(df_holder[df_holder.Date==find_nearest(df_holder.Date,final_date)].bin_index.values[0]) # find the nearest date and record the bin index
				final_bin_list.append(tuple(location_tuple))
				
				try:
					advance = df_holder[df_holder.index==(df_token.tail(1).index.values[0]+1)]
					date_span_list.append((advance.Date.min()-df_token.Date.min()).days)
					end_bin_list.append(advance['bin_index'].values[0])
				except IndexError:
					date_span_list.append(np.nan)
					end_bin_list.append(np.nan)
				if any(len(lst) != len(position_type_list) for lst in [start_date_list, start_bin_list, end_bin_list, date_span_list, final_bin_list]):
					print 'position list length ,',len(position_type_list)
					print 'start date list ',len(start_date_list)
					print 'start bin list ',len(start_bin_list)
					print 'end bin list ',len(end_bin_list)
					print 'date span list ',len(final_bin_list)
					raise
		df_dict = {}
		df_dict['Cruise']=cruise_list
		df_dict['position type']=position_type_list
		df_dict['date'] = start_date_list
		df_dict['start bin'] = start_bin_list
		for time_delta,bin_list in zip(time_delta_list,zip(*final_bin_list)):
			bins_string = 'end bin '+str(time_delta)+' day'
			df_dict[bins_string]=bin_list
		df_dict['end bin'] = end_bin_list
		df_dict['date span'] = date_span_list
		self.df_transition = pd.DataFrame(df_dict)
		if dump:
			self.df_transition.to_pickle(self.base_file+'transition_df_degree_bins_'+str(self.degree_bins)+'.pickle')

	def identify_problems_df_transition(self,plot=False):
		"""
		Initiates tests to make sure the transition matrix has reasonable statistics

		todo: consider making this a check for generic matrices, and impliment this test at every level of the code
		"""
		degree_max = self.date_span_limit/3.
		lat1,lon1 = zip(*self.df_transition['start bin'].values)
		lat2,lon2 = zip(*self.df_transition[self.end_bin_string].values)
		lat_diff = abs(np.array(lat1)-np.array(lat2))
		lon_diff = abs(np.array(lon1)-np.array(lon2))
		lon_diff[lon_diff>180] = abs(lon_diff[lon_diff>180]-360)
		distance = np.sqrt((lon_diff*np.cos(lat1))**2+lat_diff**2)
		self.df_transition['distance_check'] = distance 
		cruise_list = self.df_transition[self.df_transition.distance_check>degree_max].Cruise.unique()
		if plot:
			for cruise in cruise_list:
				df_holder = self.df[self.df.Cruise==cruise]
				df_holder1 = self.df_transition[self.df_transition.Cruise==cruise]
				x = df_holder.Lon.values
				y = df_holder.Lat.values
				plt.plot(x,y)
				plt.scatter(x,y)
				plt.title('Max distance is '+str(df_holder1.distance_check.max()))
				plt.show()
		self.df_transition = self.df_transition[~self.df_transition.Cruise.isin(cruise_list)]


	def recompile_transition_matrix(self,dump=True):
		"""
		Recompiles transition matrix from df_transition based on set timestep
		"""
		num_list = []
		num_list_index = []
		data_list = []
		row_list = []
		column_list = []

		k = len(self.df_transition['start bin'].unique())
		for n,ii in enumerate(self.df_transition['start bin'].unique()):
			print 'made it through ',n,' bins. ',(k-n),' remaining'
			ii_index = self.total_list.index(list(ii))
			frame = self.df_transition[self.df_transition['start bin']==ii]	# data from of floats that start in looped grid cell
			frame_cut = frame[frame[self.end_bin_string]!=ii].dropna(subset=[self.end_bin_string]) #this is all floats that go out of the looped grid cell
			num_list_index.append(ii_index) # we only need to save this once, because we are only concerned about the diagonal
			num_list.append(len(frame_cut)) # this is where we save the data density of every cell
			if not frame_cut.empty:
				print 'the frame cut was not empty'
				test_list = []
				row_list.append(ii_index)	#compute the on diagonal elements
				column_list.append(ii_index)	#compute the on diagonal elemnts
				data = (len(frame)-len(frame_cut))/float(len(frame)) # this is the percentage of floats that stay in the looped grid cell
				data_list.append(data)
				test_list.append(data)
				print 'total number of row slots is ',len(frame_cut[self.end_bin_string].unique())
				for qq in frame_cut[self.end_bin_string].unique():
					qq_index = self.total_list.index(list(qq))
					row_list.append(qq_index)
					column_list.append(ii_index)
					data = (len(frame_cut[frame_cut[self.end_bin_string]==qq]))/float(len(frame))
					data_list.append(data)
					test_list.append(data)
			else:
				print 'the frame cut was empy, so I will calculate the scaled dispersion'
				test_list = []
				diagonal = 1
				for qq in frame['end bin'].unique():
					try:
						qq_index = self.total_list.index(list(qq))
					except ValueError:
						print qq,' was not in the list'
						continue
					frame_holder = frame[frame['end bin']==qq]
					percentage_of_floats = len(frame_holder)/float(len(frame))
					frame_holder['time step percent'] = self.date_span_limit/frame_holder['date span']
					frame_holder[frame_holder['time step percent']>1]=1
					off_diagonal = len(frame_holder)/float(len(frame))*frame_holder['time step percent'].mean()
					diagonal -= off_diagonal
					row_list.append(qq_index)
					column_list.append(ii_index)
					data_list.append(off_diagonal)
					test_list.append(off_diagonal)
					print off_diagonal
				print diagonal
				row_list.append(ii_index)
				column_list.append(ii_index)
				data_list.append(diagonal)
				test_list.append(diagonal)
			assert abs(sum(test_list)-1)<0.01	#ensure that all columns scale to 1
			assert ~np.isnan(data_list).any()
			assert (np.array(data_list)<=1).all()
			assert (np.array(data_list)>=0).all()
		
		self.transition_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(self.total_list),len(self.total_list)))
		# self.transition_matrix = self.add_noise(self.transition_matrix)
		self.number_matrix = scipy.sparse.csc_matrix((num_list,(num_list_index,num_list_index)),shape=(len(self.total_list),len(self.total_list)))
		if dump:
			assert (np.abs(self.transition_matrix.todense().sum(axis=0)-1)<0.01).all() #this is in the dump subroutine for the case that it is recompiled for the data withholding experiment.
			save_sparse_csr(self.base_file+'transition_matrix/transition_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',self.transition_matrix)
			save_sparse_csr(self.base_file+'transition_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',self.number_matrix)

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
			self.w = load_sparse_csr(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz')
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
				save_sparse_csr(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz',self.w)	#and save

for degree in [2,3,4]:
	for time in np.arange(20,300,20):
		traj_class = argo_traj_data(degree_bins=degree,date_span_limit=time)
