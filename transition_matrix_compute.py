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
from mpl_toolkits.basemap import Basemap

class Basemap(Basemap):
    def ellipse(self, x0, y0, a, b, n, phi=0, ax=None, **kwargs):
        """
        Draws a polygon centered at ``x0, y0``. The polygon approximates an
        ellipse on the surface of the Earth with semi-major-axis ``a`` and 
        semi-minor axis ``b`` degrees longitude and latitude, made up of 
        ``n`` vertices.

        For a description of the properties of ellipsis, please refer to [1].
        
        The polygon is based upon code written do plot Tissot's indicatrix
        found on the matplotlib mailing list at [2].
        Extra keyword ``ax`` can be used to override the default axis instance.

        Other \**kwargs passed on to matplotlib.patches.Polygon

        RETURNS
            poly : a maptplotlib.patches.Polygon object.

        REFERENCES
            [1] : http://en.wikipedia.org/wiki/Ellipse
        """
        ax = kwargs.pop('ax', None) or self._check_ax()
        g = pyproj.Geod(a=self.rmajor, b=self.rminor)
        # Gets forward and back azimuths, plus distances between initial
        # points (x0, y0)
        azf, azb, dist = g.inv([x0, x0], [y0, y0], [x0+a, x0], [y0, y0+b])
        tsid = dist[0] * dist[1] # a * b

        # Initializes list of segments, calculates \del azimuth, and goes on 
        # for every vertex
        seg = []
        AZ = np.linspace(azf[0], 360. + azf[0], n)
        for i, az in enumerate(AZ):
            # Skips segments along equator (Geod can't handle equatorial arcs).
            if np.allclose(0., y0) and (np.allclose(90., az) or
                np.allclose(270., az)):
                continue
            # In polar coordinates, with the origin at the center of the 
            # ellipse and with the angular coordinate ``az`` measured from the
            # major axis, the ellipse's equation  is [1]:
            #
            #                           a * b
            # r(az) = ------------------------------------------
            #         ((b * cos(az))**2 + (a * sin(az))**2)**0.5
            #
            # Azymuth angle in radial coordinates and corrected for reference
            # angle.
            azr = 2. * np.pi / 360. * (phi+az + 90.)
            A = dist[0] * np.sin(azr)
            B = dist[1] * np.cos(azr)
            r = tsid / (B**2. + A**2.)**0.5
            lon, lat, azb = g.fwd(x0, y0, az, r)
            x, y = self(lon, lat)

            # Add segment if it is in the map projection region.
            if x < 1e20 and y < 1e20:
                seg.append((x, y))
        # print seg
        poly = Polygon(seg, **kwargs)
        ax.add_patch(poly)

        # Set axes limits to fit map region.
        self.set_axes_limits(ax=ax)
        return poly



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

class TransMatrix():
	def __init__(self,degree_bin_lat=2,degree_bin_lon=2,date_span_limit=60,traj_file_type='Argo'):
		self.info = base_info(degree_bin_lat,degree_bin_lon,date_span_limit,traj_file_type)
		self.traj = trajectory(self.info)
		self.trans = transition(self.info,self.traj)
		self.matrix = matrix(self.info,self.trans)

class base_info():
	def __init__(self,degree_bin_lat,degree_bin_lon,date_span_limit,traj_file_type):
		self.traj_file_type = traj_file_type
		if self.traj_file_type=='Argo':
			self.base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/'	#this is a total hack to easily change the code to run sose particles
			print 'I am loading Argo'
		elif self.traj_file_type=='SOSE':
			self.base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose/'
			print 'I am loading SOSE data'

		print 'I have started argo traj data'
		self.degree_bins = (degree_bin_lat,degree_bin_lon)
		self.date_span_limit = date_span_limit
		self.bins_lat = np.arange(-90,90.1,self.degree_bins[0]).tolist()
		self.bins_lon = np.arange(-180,180.1,self.degree_bins[1]).tolist()
		self.end_bin_string = 'end bin '+str(self.date_span_limit)+' day' # we compute the transition df for many different date spans, here is where we find that column
		if 180.0 not in self.bins_lon:
			print '180 is not divisable by the degree bins chosen'		#need to add this logic for the degree bin choices that do not end at 180.
			raise
		self.X,self.Y = np.meshgrid(self.bins_lon,self.bins_lat)    #these are necessary private variables for the plotting routines

class trajectory():
	def __init__(self,information):
		self.info = information
		self.load_dataframe(pd.read_pickle(self.info.base_file+'all_argo_traj.pickle'))

	def load_dataframe(self,dataframe):
		if self.info.traj_file_type=='SOSE':
			dataframe=dataframe[dataframe.Lat<-36] # this cuts off floats that wander north of 35
		dataframe = dataframe.sort_values(by=['Cruise','Date'])
		dataframe['bins_lat'] = pd.cut(dataframe.Lat,bins = self.info.bins_lat,labels=self.info.bins_lat[:-1])
		dataframe['bins_lon'] = pd.cut(dataframe.Lon,bins = self.info.bins_lon,labels=self.info.bins_lon[:-1])
		dataframe['bin_index'] = zip(dataframe['bins_lat'].values,dataframe['bins_lon'].values)
		dataframe = dataframe.reset_index(drop=True)
		#implement tests of the dataset so that all values are within the known domain
		assert dataframe.Lon.min() >= -180
		assert dataframe.Lon.max() <= 180
		assert dataframe.Lat.max() <= 90
		assert dataframe.Lat.min() >=-90
		print 'Trajectory dataframe passed necessary tests'
		self.df = dataframe

################## plotting routines ###########################

	def profile_density_plot(self): #plot the density of profiles
	    ZZ = np.zeros([len(self.info.bins_lat),len(self.info.bins_lon)])
	    series = self.df.groupby('bin_index').count()['Cruise']
	    for item in series.iteritems():
	        tup,n = item
	        ii_index = self.info.bins_lon.index(tup[1])
	        qq_index = self.info.bins_lat.index(tup[0])
	        ZZ[qq_index,ii_index] = n           
	    plt.figure(figsize=(10,10))
	    m = Basemap(projection='cyl',fix_aspect=False)
	    # m.fillcontinents(color='coral',lake_color='aqua')
	    m.drawcoastlines()
	    XX,YY = m(self.info.X,self.info.Y)
	    ZZ = np.ma.masked_equal(ZZ,0)
	    m.pcolormesh(XX,YY,ZZ,vmin=0,cmap=plt.cm.magma)
	    plt.title('Profile Density',size=30)
	    plt.colorbar(label='Number of float profiles')
	    print 'I am saving argo dense figure'
	    plt.savefig(self.info.base_file+'argo_dense_data/number_matrix_degree_bins_'+str(self.info.degree_bins)+'.png')
	    plt.close()

	def deployment_density_plot(self): # plot the density of deployment locations
	    ZZ = np.zeros([len(self.info.bins_lat),len(self.info.bins_lon)])
	    series = self.df.drop_duplicates(subset=['Cruise'],keep='first').groupby('bin_index').count()['Cruise']
	    for item in series.iteritems():
	        tup,n = item
	        ii_index = self.info.bins_lon.index(tup[1])
	        qq_index = self.info.bins_lat.index(tup[0])
	        ZZ[qq_index,ii_index] = n  
	    plt.figure(figsize=(10,10))
	    m = Basemap(projection='cyl',fix_aspect=False)
	    # m.fillcontinents(color='coral',lake_color='aqua')
	    m.drawcoastlines()
	    XX,YY = m(self.info.X,self.info.Y)
	    ZZ = np.ma.masked_equal(ZZ,0)
	    m.pcolormesh(XX,YY,ZZ,vmin=0,vmax=30,cmap=plt.cm.magma)
	    plt.title('Deployment Density',size=30)
	    plt.colorbar(label='Number of floats deployed')
	    plt.savefig(self.info.base_file+'deployment_number_data/number_matrix_degree_bins_'+str(self.info.degree_bins)+'.png')
	    plt.close()

	def deployment_location_plot(self): # plot all deployment locations
	    plt.figure(figsize=(10,10))
	    m = Basemap(projection='cyl',fix_aspect=False)
	    # m.fillcontinents(color='coral',lake_color='aqua')
	    m.drawcoastlines()
	    series = self.df.drop_duplicates(subset=['Cruise'],keep='first')
	    m.scatter(series.Lon.values,series.Lat.values,s=0.2)
	    plt.title('Deployment Locations',size=30)
	    plt.savefig(self.info.base_file+'deployment_number_data/deployment_locations.png')
	    plt.close()

class transition():
	def __init__(self,information,traj):
		self.info = information
		self.trajectory = traj
		try:
			self.load_df_and_list(pd.read_pickle(self.info.base_file+'transition_df_degree_bins_'+str(self.info.degree_bins[0])+'_'+str(self.info.degree_bins[1])+'.pickle'))
		except IOError: #this is the case that the file could not load
			print 'i could not load the transition df, I am recompiling with degree step size', self.info.degree_bins,''
			print 'file was '+self.info.base_file+'transition_df_degree_bins_'+str(self.info.degree_bins[0])+'_'+str(self.info.degree_bins[1])+'.pickle'
			self.recompile_df(traj.df)

	def load_df_and_list(self,dataframe):
		dataframe = dataframe.dropna(subset=[self.info.end_bin_string]) 
		dataframe = dataframe.dropna(subset=['start bin']) 
		assert ~dataframe['start bin'].isnull().any()
		assert ~dataframe[self.info.end_bin_string].isnull().any()
		dataframe = dataframe[dataframe['start bin'].isin(dataframe[self.info.end_bin_string].unique())]
		token = dataframe[dataframe[self.info.end_bin_string]!=dataframe['start bin']].drop_duplicates().groupby(self.info.end_bin_string).count() 
		total_list = [list(x) for x in token[token['start bin']>0].index.unique().values.tolist()] # this will make a unique "total list" for every transition matrix, but is necessary so that we dont get "holes" in the transition matrix
		dataframe = dataframe[dataframe[self.info.end_bin_string].isin([tuple(x) for x in total_list])]
		print 'Transition dataframe passed all necessary tests'
		self.df = dataframe
		self.list = total_list

	def recompile_df(self,dataframe,dump=True):
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
		k = len(dataframe.Cruise.unique())
		for n,cruise in enumerate(dataframe.Cruise.unique()):
			print 'cruise is ',cruise,'there are ',k-n,' cruises remaining'
			print 'start bin list is ',len(start_bin_list),' long'
			mask = dataframe.Cruise==cruise  		#pick out the cruise data from the df
			df_holder = dataframe[mask]
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
			dataframe.to_pickle(self.info.base_file+'transition_df_degree_bins_'+str(self.info.degree_bins[0])+'_'+str(self.info.degree_bins[1])+'.pickle')
		self.load_df_and_list(dataframe)

class matrix():
	def __init__(self,information,transition):
		self.info = information
		self.transition = transition
		self.recompile_transition_and_number_matrix()

	def load_transition_and_number_matrix(self,transition_matrix,number_matrix):
		self.transition_matrix = transition_matrix
		self.number_matrix = number_matrix
		assert (np.abs(self.transition_matrix.sum(axis=0)-1)<10**-10).all()
		assert (self.transition_matrix>0).data.all() 
		assert len(self.transition.list)==self.transition_matrix.shape[0]
		print 'Transition matrix passed all necessary tests. Initial load complete'

	def column_compute(self,ii,total_list):
#this algorithm has problems with low data density because the frame does not drop na values for the self.end_bin_string
		token_row_list = []
		token_column_list = []
		token_num_list = []
		token_num_test_list = []
		token_data_list = []

		ii_index = total_list.index(list(ii))
		frame = self.transition.df[self.transition.df['start bin']==ii]	# data from of floats that start in looped grid cell
		frame_cut = frame[frame[self.info.end_bin_string]!=ii].dropna(subset=[self.info.end_bin_string]) #this is all floats that go out of the looped grid cell
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
		for qq in frame_cut[self.info.end_bin_string].unique():
			qq_index = total_list.index(list(qq))
			token_row_list.append(qq_index)	#these will be the off diagonal elements
			token_column_list.append(ii_index)
			data = (len(frame_cut[frame_cut[self.info.end_bin_string]==qq]))/float(len(frame))
			token_data_list.append(data)
			token_num_list.append(len(frame_cut[frame_cut[self.info.end_bin_string]==qq])) # this is where we save the data density of every cell
		assert abs(sum(token_data_list)-1)<0.01	#ensure that all columns scale to 1
		assert ~np.isnan(token_data_list).any()
		assert (np.array(token_data_list)<=1).all()
		assert (np.array(token_data_list)>=0).all()
		assert sum(token_num_list)==len(frame)
		return (token_num_list, token_data_list, token_row_list,token_column_list)

	def matrix_coordinate_change(self,matrix_,old_coord_list,new_coord_list):
		row_list = matrix_.tocoo().row.tolist()
		column_list = matrix_.tocoo().col.tolist()
		data_list = matrix_.tocoo().data.tolist()
		new_row_col_list = []
		new_data_list = []
		for col,row,data in zip(column_list,row_list,data_list):
			try: 
				token = (new_coord_list.index(old_coord_list[row]),new_coord_list.index(old_coord_list[col]))
				new_row_col_list.append(token)
				new_data_list.append(data)
				assert len(new_data_list)==len(new_row_col_list)
			except ValueError:
				continue
		new_row_list,new_col_list = zip(*new_row_col_list)
		return scipy.sparse.csc_matrix((new_data_list,(new_row_list,new_col_list)),shape=(len(new_coord_list),len(new_coord_list)))

	def matrix_recalc(self,index_list,old_total_list,transition_matrix):
		row_list = transition_matrix.tocoo().row.tolist()
		column_list = transition_matrix.tocoo().col.tolist()
		data_list = transition_matrix.tocoo().data.tolist()

		full_matrix_list = range(transition_matrix.shape[0])
		not_index_list = np.array(full_matrix_list)[np.isin(full_matrix_list,[old_total_list.index(x) for x in self.transition.list])] # all of the matrix that is not in the new transition list

		index_mask = np.isin(column_list,not_index_list) # this eliminates all columns that have bad data
		print 'the column list is '+str(len(column_list))+' long and we are rejecting '+str((len(index_mask)-sum(index_mask)))

		dummy, column_redo = np.where(transition_matrix[index_list,:].todense()!=0)	#find all of the row locations where the transition matrix is non zero
		column_redo = np.unique(column_redo)
		# column_redo = column_redo[~np.isin(column_redo,index_list)] # no need to redo columns that we are removing from the matrix 
		column_redo_mask = ~np.isin(column_list,column_redo)	# delete all parts of the matrix that need to be redone
		print 'the column list is '+str(len(column_list))+' long and we are recalculating '+str((len(column_redo_mask)-sum(column_redo_mask)))

		mask = index_mask&column_redo_mask
		row_list = np.array(row_list)[mask].tolist()
		column_list = np.array(column_list)[mask].tolist()
		data_list = np.array(data_list)[mask].tolist()
		k = len(column_redo)
		for n,index in enumerate(column_redo):
			if n%10==0:
				print 'made it through ',n,' bins. ',(k-n),' remaining'
			dummy_num_list, dummy_data_list,dummy_row_list,dummy_column_list = self.column_compute(tuple(old_total_list[index]),total_list=old_total_list)
			data_list += dummy_data_list
			row_list += dummy_row_list
			column_list += dummy_column_list

			assert len(column_list)==len(row_list)
			assert len(column_list)==len(data_list)

		transition_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(old_total_list),len(old_total_list)))
		transition_matrix = self.matrix_coordinate_change(transition_matrix,old_total_list,self.transition.list)
		transition_matrix += self.add_noise()
		transition_matrix = self.rescale_matrix(transition_matrix)
		return transition_matrix


	def recompile_transition_and_number_matrix(self,plot=False):
		"""
		Recompiles transition matrix from __transition_df based on set timestep
		"""
		org_number_list = []
		org_data_list = []
		org_row_list = []
		org_column_list = []

		k = len(self.transition.list)	# sets up the total number of loops
		for n,index in enumerate(self.transition.list): #loop through all values of total_list
			print 'made it through ',n,' bins. ',(k-n),' remaining' 
			dummy_num_list, dummy_data_list,dummy_row_list,dummy_column_list = self.column_compute(tuple(index),self.transition.list)
			org_number_list += dummy_num_list
			org_data_list += dummy_data_list
			org_row_list += dummy_row_list
			org_column_list += dummy_column_list
			assert len(org_column_list)==len(org_row_list)
			assert len(org_column_list)==len(org_data_list)
			assert len(org_column_list)==len(org_number_list)
		assert sum(org_number_list)==len(self.transition.df[self.transition.df['start bin'].isin([tuple(self.transition.list[x]) for x in np.unique(org_column_list)])])

		transition_matrix = scipy.sparse.csc_matrix((org_data_list,(org_row_list,org_column_list)),shape=(len(self.transition.list),len(self.transition.list)))
		number_matrix = scipy.sparse.csc_matrix((org_number_list,(org_row_list,org_column_list)),shape=(len(self.transition.list),len(self.transition.list)))

		# self.transition_matrix = self.add_noise(self.transition_matrix)

		eig_vals,eig_vecs = scipy.linalg.eig(transition_matrix.todense())
		orig_len = len(transition_matrix.todense())
		idx = np.where((eig_vecs>0).sum(axis=0)<=2) #find the indexes of all eigenvectors with less than 2 entries
		index_list = []
		for index in idx[0]:
			eig_vec = eig_vecs[:,index]		
			index_list+=np.where(eig_vec>0)[0].tolist() # identify the specific grid boxes where the eigen values are 1 
		dummy,column_index = np.where(np.abs(transition_matrix.sum(axis=0)-1)>10**-10)

		index_list = np.unique(index_list+column_index.tolist()) #these may be duplicated so take unique values

		if index_list.tolist():	#if there is anything in the index list 
			transition_df_holder = self.transition.df.copy()
			transition_df_holder.loc[self.transition.df[self.info.end_bin_string].isin([tuple(x) for x in np.array(self.transition.list)[index_list]]),self.info.end_bin_string]=np.nan # make all of the end string values nan in these locations
			transition_df_holder.loc[self.transition.df['start bin'].isin([tuple(x) for x in np.array(self.transition.list)[index_list]]),'start bin']=np.nan # make all of the start bin values nan in these locations				
			old_total_list = copy.deepcopy(self.transition.list)
			self.transition.load_df_and_list(transition_df_holder)
			assert ~np.isin(index_list,[old_total_list.index(list(x)) for x in self.transition.df['start bin'].unique()]).any()
			assert ~np.isin(index_list,[old_total_list.index(list(x)) for x in self.transition.df[self.info.end_bin_string].unique()]).any()
			transition_matrix = self.matrix_recalc(index_list,old_total_list,transition_matrix)
		self.load_transition_and_number_matrix(transition_matrix,number_matrix)

	def get_direction_matrix(self):
	    """
	    This creates a matrix that shows the number of grid cells north and south, east and west.
	    """
	    lat_list, lon_list = zip(*self.transition.list)
	    pos_max = 180/self.info.degree_bins[1] #this is the maximum number of bins possible
	    output_list = []
	    for n,position_list in enumerate([lat_list,lon_list]):
	        token_array = np.zeros([len(position_list),len(position_list)])
	        for token in np.unique(position_list):
	            index_list = np.where(np.array(position_list)==token)[0]
	            token_list = (np.array(position_list)-token)/self.info.degree_bins[n] #the relative number of degree bins
	            token_list[token_list>pos_max]=token_list[token_list>pos_max]-2*pos_max #the equivalent of saying -360 degrees
	            token_list[token_list<-pos_max]=token_list[token_list<-pos_max]+2*pos_max #the equivalent of saying +360 degrees
	            token_array[:,index_list]=np.array([token_list.tolist(),]*len(index_list)).transpose() 
	        output_list.append(token_array)
	    north_south,east_west = output_list
	    self.east_west = east_west
	    self.north_south = north_south  #this is because np.array makes lists of lists go the wrong way
	    assert (self.east_west<=180/self.info.degree_bins[1]).all()
	    assert (self.east_west>=-180/self.info.degree_bins[1]).all()
	    assert (self.north_south>=-180/self.info.degree_bins[0]).all()
	    assert (self.north_south<=180/self.info.degree_bins[0]).all()


	def add_noise(self,noise=0.05):
		"""
		Adds guassian noise to the transition matrix
		The appropriate level of noise has not been worked out and is kind of ad hock
		"""
		print 'adding matrix noise'
		self.get_direction_matrix()
		direction_mat = -abs(self.east_west)**2-abs(self.north_south)**2
		direction_mat = noise*np.exp(direction_mat)
		direction_mat[direction_mat<noise*np.exp(-20)]=0
		return scipy.sparse.csc_matrix(direction_mat)

	def rescale_matrix(self,matrix_):
		print 'rescaling the matrix'
		mat_sum = matrix_.todense().sum(axis=0)
		scalefactor,dummy = np.meshgrid(1/mat_sum,1/mat_sum)
		matrix_ = scipy.sparse.csc_matrix(np.array(matrix_.todense())*np.array(scalefactor)) #must make these arrays so that the multiplication is element wise
		assert (np.abs(matrix_.sum(axis=0)-1)<10**-5).all()
		return matrix_

	def trajectory_data_withholding_setup(self,percentage):
        # number_matrix = load_sparse_csc(self.base_file+'transition_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
        # standard_error_matrix = load_sparse_csc(self.base_file+'transition_matrix/standard_error_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
		old_coord_list = copy.deepcopy(self.transition.list)

		mylist = self.transition.df.Cruise.unique().tolist() # total list of floats
		remaining_float_list = [mylist[i] for i in sorted(random.sample(xrange(len(mylist)), int(round(len(mylist)*(1-percentage)))))] #randomly select floats to exclude
		subtracted_float_list = np.array(mylist)[~np.isin(mylist,remaining_float_list)]

		df_holder = self.transition.df[self.transition.df.Cruise.isin(subtracted_float_list)]
		
		index_list = []
		for x in (df_holder[self.info.end_bin_string].unique().tolist()+df_holder['start bin'].unique().tolist()):
			try:
				index_list.append(old_coord_list.index(list(x)))
			except ValueError:
				print 'there was a value error during withholding'
				continue
		index_list = np.unique(index_list)

		new_df = self.transition.df[self.transition.df.Cruise.isin(remaining_float_list)]        
		self.transition.load_df_and_list(new_df)
		transition_matrix = self.matrix_recalc(index_list,old_coord_list,self.transition_matrix)
		self.load_transition_and_number_matrix(transition_matrix,number_matrix)

	############## PLOTTING ROUTINES #############
	def transition_vector_to_plottable(self,vector):
	    plottable = np.zeros([len(self.info.bins_lat),len(self.info.bins_lon)])
	    for n,tup in enumerate(self.transition.list):
	        ii_index = self.info.bins_lon.index(tup[1])
	        qq_index = self.info.bins_lat.index(tup[0])
	        plottable[qq_index,ii_index] = vector[n]
	    return plottable

	def matrix_plot_setup(self):
		if self.info.traj_file_type == 'SOSE':
		    print 'I am plotting antarctic region'
		    m = Basemap(llcrnrlon=-180.,llcrnrlat=-80.,urcrnrlon=180.,urcrnrlat=-25,projection='cyl',fix_aspect=False)
		else:
		    print 'I am plotting global region'
		    m = Basemap(projection='cyl',fix_aspect=False)
		m.drawcoastlines()
		XX,YY = m(self.info.X,self.info.Y)
		return (m,XX,YY)        

	def number_plot(self): 
		k = self.number_matrix.sum(axis=0)
		k = k.T
		print k
		number_matrix_plot = self.transition_vector_to_plottable(k)
		plt.figure('number matrix',figsize=(10,10))
		m,XX,YY = self.matrix_plot_setup()
		number_matrix_plot = np.ma.masked_equal(number_matrix_plot,0)   #this needs to be fixed in the plotting routine, because this is just showing the number of particles remaining
		m.pcolormesh(XX,YY,number_matrix_plot,cmap=plt.cm.magma)
		plt.title('Transition Density',size=30)
		plt.colorbar(label='Number of float transitions')
		plt.savefig(self.info.base_file+'/number_matrix/number_matrix_degree_bins_'+str(self.info.degree_bins)+'_time_step_'+str(self.info.date_span_limit)+'.png')
		plt.close()

	def standard_error_plot(self):
		row_list = self.transition_matrix.tocoo().row.tolist()
		column_list = self.transition_matrix.tocoo().col.tolist()
		data_list = self.transition_matrix.tocoo().data.tolist()		
		sparse_ones = scipy.sparse.csc_matrix(([1 for x in range(len(data_list))],(row_list,column_list)),shape=(len(self.transition.list),len(self.transition.list)))
		standard_error = np.sqrt(self.transition_matrix*(sparse_ones-self.transition_matrix)/self.number_matrix)
		standard_error = scipy.sparse.csc_matrix(standard_error)
		k = np.diagonal(standard_error.todense())
		standard_error_plot = self.transition_vector_to_plottable(k)
		plt.figure('standard error')
		# m.fillcontinents(color='coral',lake_color='aqua')
		# number_matrix_plot[number_matrix_plot>1000]=1000
		m,XX,YY = self.matrix_plot_setup()
		q = self.number_matrix.sum(axis=0)
		q = q.T
		standard_error_plot = np.ma.array(standard_error_plot,mask=self.transition_vector_to_plottable(q)==0)
		m.pcolormesh(XX,YY,standard_error_plot,cmap=plt.cm.cividis)
		plt.title('Standard Error',size=30)
		plt.colorbar(label='Standard Error')
		plt.savefig(self.info.base_file+'/number_matrix/standard_error_degree_bins_'+str(self.info.degree_bins)+'_time_step_'+str(self.info.date_span_limit)+'.png')
		plt.close()

	def transition_matrix_plot(self,filename):
		plt.figure(figsize=(10,10))
		k = np.diagonal(self.transition_matrix.todense())
		transition_plot = self.transition_vector_to_plottable(k)
		m,XX,YY = self.matrix_plot_setup()
		k = self.number_matrix.sum(axis=0)
		k = k.T
		transition_plot = np.ma.array((1-transition_plot),mask=self.transition_vector_to_plottable(k)==0)
		m.pcolormesh(XX,YY,transition_plot,vmin=0,vmax=1) # this is a plot for the tendancy of the residence time at a grid cell
		plt.colorbar(label='% particles dispersed')
		plt.title('1 - diagonal of transition matrix',size=30)
		plt.savefig(self.info.base_file+'transition_plots/'+filename+'_diag_degree_bins_'+str(self.info.degree_bins)+'_time_step_'+str(self.info.date_span_limit)+'.png')
		plt.close()


date_span_limit = 60
coord_list = [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]
for outer in coord_list:
	for traj_file_type in ['Argo','SOSE']:
		lat_outer,lon_outer = outer
		trans_class = TransMatrix(lat_outer,lon_outer,date_span_limit,traj_file_type)
	# def delete_w(self):
	# 	"""
	# 	deletes all w matrices in data directory
	# 	"""
	# 	for w in glob.glob(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_*.npz'):
	# 		os.remove(w)

	# def load_w(self,number,dump=True):
	# 	"""
	# 	recursively multiplies the transition matrix by itself 
	# 	"""
	# 	print 'in w matrix, current number is ',number
	# 	try:
	# 		self.w = load_sparse_csc(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz')
	# 		assert self.transition_matrix.shape==self.w.shape
 # 			assert ~np.isnan(self.w).any()
	# 		assert (np.array(self.w)<=1).all()
	# 		assert (np.array(self.w)>=0).all()
	# 		assert (self.w.todense().sum(axis=0)-1<0.01).all()
	# 		print 'w matrix successfully loaded'
	# 	except IOError:
	# 		print 'w matrix could not be loaded and will be recompiled'
	# 		if number == 0:
	# 			self.w = self.transition_matrix 
	# 		else:
	# 			self.load_w(number-1,dump=True)		# recursion to get to the first saved transition matrix 
	# 			self.w = self.w.dot(self.transition_matrix)		# advance the transition matrix once
	# 		assert self.transition_matrix.shape==self.w.shape
 # 			assert ~np.isnan(self.w).any()
	# 		assert (np.array(self.w)<=1).all()
	# 		assert (np.array(self.w)>=0).all()
	# 		assert (self.w.todense().sum(axis=0)-1<0.01).all()
	# 		if dump:
	# 			save_sparse_csc(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz',self.w)	#and save