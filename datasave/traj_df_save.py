import datetime
import scipy.io
import pandas as pd
import numpy as np
import os
from utils import time_parser, wrap_lon180,loop_data
from netCDF4 import Dataset
"""file to create pandas dataframe from trajectory data"""
class BaseTraj(object):
	def __init__(self):
		self.degree_dist = 111.7	#The km distance of one degree
		self.f = open('../data/global_argo/traj_df_changelog.txt','w') #this is to keep a record of the floats that have been rejected

	def frame_checker(self,df_holder,file_name):
#performs a series of checks on every .nc record that gets passed into the dataset
		if df_holder.empty:
			self.f.write(file_name+' was empty \n')
			return df_holder

		df_holder['hour_diff'] = df_holder.Date.diff().dt.days*24+df_holder.Date.diff().dt.seconds/3600.
		if (df_holder.hour_diff.tail(len(df_holder)-1)<0).any():
			df_holder = df_holder.sort_values(['Date']).reset_index(drop=True)
			df_holder['hour_diff'] = df_holder.Date.diff().dt.days*24+df_holder.Date.diff().dt.seconds/3600.

			self.f.write(file_name+' resorted because date order not correct \n')
			print 'I am resorting the dates because they were not correct'
		assert (df_holder.hour_diff.tail(len(df_holder)-1)>0).all()	#the first hour diff will be zero and must be filtered out.
		df_holder['Lat Diff'] = df_holder.Lat.diff()
		df_holder['Lat Speed']= df_holder['Lat Diff']/df_holder['hour_diff']*self.degree_dist
		df_holder['Lon Diff']=df_holder.Lon.diff().abs()
		if not df_holder[df_holder['Lon Diff']>180].empty:
			df_holder.loc[df_holder['Lon Diff']>180,'Lon Diff']=(df_holder[df_holder['Lon Diff']>180]['Lon Diff']-360).abs()
		df_holder['Lon Speed']= np.cos(np.deg2rad(df_holder.Lat))*df_holder['Lon Diff']/df_holder['hour_diff']*self.degree_dist
		df_holder['Speed'] = np.sqrt(df_holder['Lat Speed']**2 + df_holder['Lon Speed']**2)
		if (df_holder.Speed>self.max_speed).any():
			self.f.write(file_name+' is rejected because the calculated speed is outside the '+str(self.max_speed)+' km/hour tolerance \n')
		assert (df_holder.dropna(subset=['Lon Diff'])['Lon Diff']<180).all()
		assert (df_holder.dropna(subset=['Lat Diff'])['Lat Diff']<180).all()
		assert (df_holder.dropna(subset=['Speed']).Speed>=0).all()
		assert df_holder.Lon.min() >= -180
		assert df_holder.Lon.max() <= 180
		assert df_holder.Lat.max() <= 90
		assert df_holder.Lat.min() >=-90
		assert (df_holder.dropna(subset=['hour_diff'])['hour_diff']>0).all()
		return df_holder

class MatlabTraj(BaseTraj):
	def __init__(self,matlab_directory='/iCloud/Data/Raw/Traj/mat/',**kwds):
		super(MatlabTraj,self).__init__(**kwds)
		self.dataframe_directory ='../data/global_argo/'
		self.file_path = self.dataframe_directory+'unprocessed_matlab_traj.pickle'
		try:
			self.traj_df = pd.read_pickle(self.file_path)
			print 'I have successfully loaded the matlab df'
		except IOError:
			self.matlab_df_save(os.getenv("HOME")+matlab_directory)

	def matlab_frame_return(self,file_name):
		traj_dict = scipy.io.loadmat(file_name)
		lon_i = traj_dict['subxlon_i'].flatten().tolist() 
#actual last longitude of previous cycle for subsurface velocity estimates
		lon_i = wrap_lon180(lon_i)
		lon_f = traj_dict['subxlon_f'].flatten().tolist() 
#actual first longitude of current cycle for subsurface velocity estimates
		lon_f = wrap_lon180(lon_f)
		lat_i = traj_dict['subxlat_i'].flatten().tolist() 
#actual last latitude of previous cycle for subsurface velocity estimates
		lat_f = traj_dict['subxlat_f'].flatten().tolist() 
#actual first latitude of current cycle for subsurface velocity estimates
		pres = traj_dict['subpres'].flatten().tolist() 
#best estimate of drift pressure
		date_i = time_parser(traj_dict['subxjday_i'].flatten().tolist()) 
#actual last time of previous cycle's surface period for subsurface velocity estimates
		date_f = time_parser(traj_dict['subxjday_f'].flatten().tolist()) 
#actual first time of current cycle's surface period for subsurface velocity estimates

		cruise = traj_dict['subwmo'][0]	
#this is the WMO id number of the cruise
		positioning_type = traj_dict['subtrans_sys'][0]	
#this is the positioning type used for the cruise
		assert ~all(v == len(lon_i) for v in [len(lon_i),len(lon_f),len(lat_i),len(lat_f),len(date_i),len(date_f),len(pres)])	
# all lists have to be the same length
		df_holder = pd.DataFrame({'position type':positioning_type,'Cruise':cruise,'lon_i':lon_i,'Lon':lon_f,'lat_i':lat_i,'Lat':lat_f,'date_i':date_i,'Date':date_f,'pres':pres}) # create dataframe of all data
		df_holder = self.frame_checker(df_holder,file_name)
		return df_holder

	def matlab_df_save(self,data_directory):
		print 'I am looking for Matlab files in ',data_directory		
		frames = loop_data(data_directory,'*.mat',self.matlab_frame_return)		
		df = pd.concat(frames)
		df.loc[df['position type'].isin(['GPS     ','IRIDIUM ']),'position type'] = 'GPS'
		df.loc[df['position type'].isin(['ARGOS   ']),'position type'] = 'ARGOS'
		assert ((df['position type']=='GPS')|(df['position type']=='ARGOS')).all()
		self.traj_df = df
		self.traj_df.to_pickle(self.file_path)

class GPSTraj(BaseTraj):
	def __init__(self,gps_directory = '/iCloud/Data/Raw/ARGO',**kwds):
		super(GPSTraj,self).__init__(**kwds)
		self.dataframe_directory ='../data/global_argo/'
		self.file_path = self.dataframe_directory+'unprocessed_gps_traj.pickle'
		try:
			self.gps_df = pd.read_pickle(self.dataframe_directory)
			print 'I have successfully loaded the GPS df'
		except IOError:
			self.gps_df_save(os.getenv("HOME")+gps_directory)

	def gps_frame_return(self,file_name):
		nc_fid = Dataset(file_name, 'r')
		pos_type = ''.join([x for x in nc_fid.variables['POSITIONING_SYSTEM'][:][0].tolist() if x is not None])
		cruise = ''.join([x for x in nc_fid.variables['PLATFORM_NUMBER'][:][0].tolist() if x is not None])
		if pos_type == 'ARGOS':
			return pd.DataFrame()
# we are only interested in the GPS tracked floats
		mode = nc_fid.variables['DATA_MODE'][:]
		truth_list = mode!='R'
# we are only interested in delayed mode data 
		lat = nc_fid.variables['LATITUDE'][truth_list].tolist()
		lon = nc_fid.variables['LONGITUDE'][truth_list].tolist()
		pos_qc = nc_fid.variables['POSITION_QC'][truth_list].tolist() 
		try:
			date = nc_fid.variables['JULD_ADJUSTED'][truth_list].tolist()
			date_qc = nc_fid.variables['JULD_ADJUSTED_QC'][truth_list]
		except KeyError:
			date = nc_fid.variables['JULD'][truth_list].tolist()
			date_qc = nc_fid.variables['JULD_QC'][truth_list]
		reference_date = ''.join(nc_fid.variables['REFERENCE_DATE_TIME'][:].tolist())
		reference_date = datetime.datetime(int(reference_date[:4]),int(reference_date[4:6]),int(reference_date[6:8]))
		if None in date:
			return pd.DataFrame()
		date = time_parser(date,ref_date=reference_date)
		df_holder = pd.DataFrame({'Date':date,'DateQC':date_qc,'Lon':lon,'Lat':lat,'PosQC':pos_qc})
		df_holder['Cruise']=cruise
		df_holder['position type']='GPS'
		df_holder = df_holder.drop_duplicates(subset='Date')
		df_holder = df_holder.dropna(subset=['Lat','Lon'])
		if df_holder.empty:
			return df_holder
		df_holder = df_holder[(df_holder.DateQC=='1')&(df_holder.PosQC=='1')]
		df_holder = self.frame_checker(df_holder,file_name)
		return df_holder

	def gps_df_save(self,data_directory):
		print 'I am looking for GPS files in ',data_directory
		frames = loop_data(data_directory,'*prof.nc',self.gps_frame_return)
		df = pd.concat(frames)
		df = df.drop_duplicates(subset=['Cruise','Date'])
		df = df[[u'Cruise', u'Date', u'Lat', u'Lat Diff', u'Lat Speed',
	       u'Lon', u'Lon Diff', u'Lon Speed', u'Speed', u'hour_diff',
	       u'position type']]
		self.gps_df = df
		self.gps_df.to_pickle(self.file_path)

class AllTraj(GPSTraj,MatlabTraj):
	def __init__(self):
		super(AllTraj,self).__init__()
		self.max_speed = 5
		self.percent_reject = 0.05
		self.lat_grid = np.arange(-90,90.1,4).tolist()
		self.lon_grid = np.arange(-180,180.1,4).tolist() 
		try:
			self.df = pd.read_pickle(self.dataframe_directory+'rejected_traj_df')
			self.speed_variance_matrix = np.load(self.dataframe_directory+'traj_speed_variance.npy')
			self.speed_mean_matrix = np.load(self.dataframe_directory+'traj_speed_mean.npy')
			print 'I have successfully loaded the rejected df'
		except IOError:
			self.gps_df = self.gps_df[self.gps_df.Cruise.isin(self.traj_df.Cruise.unique())]
			try:
				self.gps_df != self.traj_df
			except ValueError:
				pass
			self.df = pd.concat([self.gps_df,self.traj_df])
			self.rejection_criteria()
		self.df_rejected = df[df.Cruise.isin(df[df.reject==True].Cruise.unique())]
		self.df = df[~df.Cruise.isin(df[df.reject==True].Cruise.unique())]

		############ Show rejected statistics and redeem some floats ###########
		percentage_df = (df_rejected[df_rejected.reject==True].groupby('Cruise').count()/df_rejected.groupby('Cruise').count())['pres'] # calculate the percentage of bad locations for each displacement
		for item in percentage_df.iteritems():
			cruise,percentage = item
			self.df_rejected.loc[df_rejected.Cruise==cruise,'percentage'] = percentage


	def rejection_criteria(self):
		self.speed_variance_matrix = np.zeros([len(self.lat_grid),len(self.lon_grid)])
		self.speed_mean_matrix = np.zeros([len(self.lat_grid),len(self.lon_grid)])
		self.df['bins_lat'] = pd.cut(self.df.Lat,bins = self.lat_grid,labels=self.lat_grid[:-1])
		self.df['bins_lon'] = pd.cut(self.df.Lon,bins = self.lon_grid,labels=self.lon_grid[:-1])
		self.df['bin_index'] = zip(self.df['bins_lat'].values,self.df['bins_lon'].values)
		assert (~self.df['bins_lat'].isnull()).all()
		assert (~self.df['bins_lon'].isnull()).all()

		self.df['reject'] = False
		self.df.loc[self.df.Speed>5,'reject']=True # reject all floats with a speed above 5 km/hour
		k  = len (self.df.bin_index.unique())
		for n,index_bin in enumerate(self.df.bin_index.unique()):
			lat,lon = index_bin
			lat_index = self.lat_grid.index(lat)
			lon_index = self.lon_grid.index(lon)
			print 'we have ',k-n,' index bins remaining'
			df_token = self.df[self.df.bin_index==index_bin]

			self.speed_mean_matrix[lat_index,lon_index] = df_token.Speed.var()
			self.speed_variance_matrix[lat_index,lon_index] = df_token.Speed.mean()

			for cruise in df_token.Cruise.unique():
				test_value = 60*df_token[df_token.Cruise!=cruise].Speed.var()+df_token[df_token.Cruise!=cruise].Speed.mean()
		# this is to make some regionality to the rejection critera. 60 is chosen arbitrarily
				if (df_token[df_token.Cruise==cruise].Speed.dropna()>test_value).any():
					self.f.write(str(cruise)+' is rejected because the maximum calculated speed is '
						+str(df_token[df_token.Cruise==cruise].Speed.dropna().max())+
						' which is outside the '+str(test_value)+' km/hour tolerance for the grid box \n')
					self.df.loc[(self.df.Cruise==cruise)\
					&(self.df.Speed>test_value)&(self.df.bin_index==index_bin),'reject']=True
		self.df.to_pickle(self.dataframe_directory+'rejected_traj_df.pickle')
		np.save(self.dataframe_directory+'traj_speed_variance.npy',self.speed_variance_matrix)
		np.save(self.dataframe_directory+'traj_speed_mean.npy',self.speed_mean_matrix)

	def redemption(self):
#this is not included in the init portion, because the plotting routine plots pre and post redemption
		self.df = pd.concat([self.df_rejected[(self.df_rejected.percentage<self.percent_reject)&(self.df_rejected.reject==False)],self.df]) 
#We allow some redemption if less than 5% of displacements are messed up
		self.df_rejected = self.df_rejected[~self.df_rejected.Cruise.isin(self.df.Cruise.unique())]

	def save(self):
		assert (~df.reject).all()	# reject flag must be false for all floats
		assert (df.dropna(subset=['Speed'])['Lon Speed']<(self.max_speed-0.2)).all()
		assert (df.dropna(subset=['Speed'])['Lat Speed']<(self.max_speed-0.2)).all()
		self.f.close()
		self.df.to_pickle(self.dataframe_directory+'all_argo_traj.pickle')
