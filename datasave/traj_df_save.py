import fnmatch
import datetime
import scipy.io
import pandas as pd
import time # used for expressing how long each routine lasted.
import numpy as np
import os
from utils import time_parser, wrap_lon180


"""file to create pandas dataframe from trajectory data"""


max_speed = 5
percent_reject = 0.05



class BaseTrajSave(object):
	def __init__(self):
		self.degree_dist = 111.7	#The km distance of one degree
		self.f = open('../data/traj_df_changelog.txt','w') #this is to keep a record of the floats that have been rejected

	def frame_checker(df_holder):
#performs a series of checks on every .nc record that gets passed into the dataset
		if df_holder.empty:
			self.f.write(str(cruise)+' was empty \n')
			return df_holder

		df_holder['hour_diff'] = df_holder.Date.diff().dt.days*24+df_holder.Date.diff().dt.seconds/3600.
		if (df_holder.hour_diff.tail(len(df_holder)-1)<0).any():
			df_holder = df_holder.sort_values(['Date']).reset_index(drop=True)
			df_holder['hour_diff'] = df_holder.Date.diff().dt.days*24+df_holder.Date.diff().dt.seconds/3600.

			self.f.write(str(cruise)+' resorted because date order not correct \n')
			print 'I am resorting the dates because they were not correct'
		assert (df_holder.hour_diff.tail(len(df_holder)-1)>0).all()	#the first hour diff will be zero and must be filtered out.
		df_holder['Lat Diff'] = df_holder.Lat.diff()
		df_holder['Lat Speed']= df_holder['Lat Diff']/df_holder['hour_diff']*self.degree_dist
		df_holder['Lon Diff']=df_holder.Lon.diff().abs()
		if not df_holder[df_holder['Lon Diff']>180].empty:
			df_holder.loc[df_holder['Lon Diff']>180,'Lon Diff']=(df_holder[df_holder['Lon Diff']>180]['Lon Diff']-360).abs()
		df_holder['Lon Speed']= np.cos(np.deg2rad(df_holder.Lat))*df_holder['Lon Diff']/df_holder['hour_diff']*self.degree_dist
		df_holder['Speed'] = np.sqrt(df_holder['Lat Speed']**2 + df_holder['Lon Speed']**2)
		if (df_holder.Speed>max_speed).any():
			self.f.write(str(cruise)+' is rejected because the calculated speed is outside the '+str(max_speed)+' km/hour tolerance \n')
		assert (df_holder.dropna(subset=['Lon Diff'])['Lon Diff']<180).all()
		assert (df_holder.dropna(subset=['Lat Diff'])['Lat Diff']<180).all()
		assert (df_holder.dropna(subset=['Speed']).Speed>=0).all()
		assert df_holder.Lon.min() >= -180
		assert df_holder.Lon.max() <= 180
		assert df_holder.Lat.max() <= 90
		assert df_holder.Lat.min() >=-90
		assert (df_holder.dropna(subset=['hour_diff'])['hour_diff']>0).all()
		return df_holder

class MatlabTrajSave(BaseTrajSave):
	def __init__(self,directory='/iCloud/Data/Raw/Traj/mat/',**kwds):
		super(MatlabTrajSave,self).__init__(**kwds)
		self.df_return(os.getenv("HOME")+directory)

	def frame_return(self,file_name):
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
		df_holder = self.frame_checker(df_holder)
		return df_holder

	def df_return(self,data_directory):
		frames = []
		matches = []
		for root, dirnames, filenames in os.walk(data_directory):
			for filename in fnmatch.filter(filenames, '*.mat'):
				matches.append(os.path.join(root, filename))
		for n, match in enumerate(matches):
			print 'file is ',match,', there are ',len(matches[:])-n,'floats left'
			t = time.time()
			frames.append(self.frame_return(match))
			print 'Building and merging datasets took ', time.time()-t
		df = pd.concat(frames)
		df.loc[df['position type'].isin(['GPS     ','IRIDIUM ']),'position type'] = 'GPS'
		df.loc[df['position type'].isin(['ARGOS   ']),'position type'] = 'ARGOS'
		assert ((df['position type']=='GPS')|(df['position type']=='ARGOS')).all()
		self.df = df

class GPSTrajSave(BaseTrajSave):
	def __init__(self,**kwds)
		super(GPSTrajSave,self).__init(directory = '/iCloud/Data/Raw/ARGO', **kwds)
		self.gps_frame_return(os.getenv("HOME")+directory)

	def frame_return(self,match):
		nc_fid = Dataset(match, 'r')
		pos_type = ''.join([x for x in nc_fid.variables['POSITIONING_SYSTEM'][:][0].tolist() if x is not None])
		cruise = ''.join([x for x in nc_fid.variables['PLATFORM_NUMBER'][:][0].tolist() if x is not None])
		if pos_type == 'ARGOS':
			return pd.DataFrame()
		mode = nc_fid.variables['DATA_MODE'][:]
		truth_list = mode!='R'
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
			continue
		date = time_parser(date,ref_date=reference_date)
		df_holder = pd.DataFrame({'Date':date,'DateQC':date_qc,'Lon':lon,'Lat':lat,'PosQC':pos_qc})
		df_holder['Cruise']=cruise
		df_holder['position type']='GPS'
		df_holder = df_holder.drop_duplicates(subset='Date')
		df_holder = df_holder.dropna(subset=['Lat','Lon'])
		if df_holder.empty:
			return df_holder
		df_holder = df_holder[(df_holder.DateQC=='1')&(df_holder.PosQC=='1')]
		df_holder = frame_checker(df_holder)
		return df_holder

	def df_save(self,data_directory):
		frames = []
		matches = []
		for root, dirnames, filenames in os.walk(data_directory): #this should be a class generator for paralization (accoording to gui)
			for filename in fnmatch.filter(filenames, '*prof.nc'): #yield function should be used
				matches.append(os.path.join(root, filename))
		for n, match in enumerate(matches):
			print 'file is ',match,', there are ',len(matches[:])-n,'floats left'
			df_holder = self.frame_return(match)
			frames.append(df_holder)
		df = pd.concat(frames)
		df = df[~df_gps.Cruise.isin(df_matlab.Cruise.unique())]
		df = df.drop_duplicates(subset=['Cruise','Date'])
		df = df[[u'Cruise', u'Date', u'Lat', u'Lat Diff', u'Lat Speed',
	       u'Lon', u'Lon Diff', u'Lon Speed', u'Speed', u'hour_diff',
	       u'position type']]
		self.df = df

try:
	pd.read_pickle(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/all_argo_unprocessed_traj.pickle')
except IOError:
	df_matlab = matlab_df_return()
	df_gps = gps_df_return()

	df = pd.concat([df_matlab,df_gps])
	df.to_pickle(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/all_argo_unprocessed_traj.pickle')

dummy_lat = np.arange(-90,90.1,4).tolist()
dummy_lon = np.arange(-180,180.1,4).tolist()

X,Y = np.meshgrid(dummy_lon,dummy_lat)
speed_variance_matrix = np.zeros(X.shape)
speed_mean_matrix = np.zeros(X.shape)
df['bins_lat'] = pd.cut(df.Lat,bins = dummy_lat,labels=dummy_lat[:-1])
df['bins_lon'] = pd.cut(df.Lon,bins = dummy_lon,labels=dummy_lon[:-1])
df['bin_index'] = zip(df['bins_lat'].values,df['bins_lon'].values)
assert (~df['bins_lat'].isnull()).all()
assert (~df['bins_lon'].isnull(        )).all()

df['reject'] = False
df.loc[df.Speed>5,'reject']=True # reject all floats with a speed above 5 km/hour
k  = len (df.bin_index.unique())
for n,index_bin in enumerate(df.bin_index.unique()):
	lat,lon = index_bin
	lat_index = dummy_lat.index(lat)
	lon_index = dummy_lon.index(lon)
	print 'we have ',k-n,' index bins remaining'
	df_token = df[df.bin_index==index_bin]

	speed_mean_matrix[lat_index,lon_index] = df_token.Speed.var()
	speed_variance_matrix[lat_index,lon_index] = df_token.Speed.mean()

	for cruise in df_token.Cruise.unique():
		test_value = 60*df_token[df_token.Cruise!=cruise].Speed.var()+df_token[df_token.Cruise!=cruise].Speed.mean()
# this is to make some regionality to the rejection critera. 60 is chosen arbitrarily
		if (df_token[df_token.Cruise==cruise].Speed.dropna()>test_value).any():
			self.f.write(str(cruise)+' is rejected because the maximum calculated speed is '
				+str(df_token[df_token.Cruise==cruise].Speed.dropna().max())+
				' which is outside the '+str(test_value)+' km/hour tolerance for the grid box \n')
			df.loc[(df.Cruise==cruise)&(df.Speed>test_value)&(df.bin_index==index_bin),'reject']=True


plt.figure(figsize=(10,10))
m = Basemap(projection='cyl',fix_aspect=False)
# m.fillcontinents(color='coral',lake_color='aqua')
m.drawcoastlines()
XX,YY = m(X,Y)
m.pcolor(X,Y,np.log(np.ma.masked_equal(speed_mean_matrix,0)))
plt.title('Log Raw Mean Speed')
plt.colorbar(label='log(km/hr)')
plt.savefig(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/raw_data_statistics/map_of_raw_mean_speed.png')
plt.close()

plt.figure(figsize=(10,10))
m = Basemap(projection='cyl',fix_aspect=False)
# m.fillcontinents(color='coral',lake_color='aqua')
m.drawcoastlines()
XX,YY = m(X,Y)
m.pcolor(X,Y,np.ma.masked_equal(speed_variance_matrix,0))
plt.title('Raw Mean Variance')
plt.colorbar(label='$km^2/hr^2$')
plt.savefig(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/raw_data_statistics/map_of_raw_speed_variance.png')
plt.close()

df_rejected = df[df.Cruise.isin(df[df.reject==True].Cruise.unique())]
df = df[~df.Cruise.isin(df[df.reject==True].Cruise.unique())]

############ Show rejected statistics and redeem some floats ###########
percentage_df = (df_rejected[df_rejected.reject==True].groupby('Cruise').count()/df_rejected.groupby('Cruise').count())['pres'] # calculate the percentage of bad locations for each displacement
for item in percentage_df.iteritems():
	cruise,percentage = item
	df_rejected.loc[df_rejected.Cruise==cruise,'percentage'] = percentage
fig, ax = plt.subplots()
df_rejected.Speed.hist(ax=ax, bins=100, bottom=0.1,label='All Floats')
df_rejected[df_rejected.percentage<percent_reject].Speed.hist(ax=ax, color='r', bins=100, bottom=0.1,alpha=.5,label='5% Rejected')
ax.set_yscale('log')
ax.set_xscale('log')
plt.legend()
plt.xlabel('km/hr')
plt.ylabel('number of displacements')
plt.title('Histogram of rejected velocities')
plt.savefig(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/raw_data_statistics/histogram_of_rejected_velocities.png')
plt.close()

fig1, ax1 = plt.subplots()
argos_number = df_rejected.drop_duplicates(subset=['Cruise']).groupby('position type').count()['Cruise']['ARGOS']
iridium_number = df_rejected.drop_duplicates(subset=['Cruise']).groupby('position type').count()['Cruise']['GPS']
argos_number_low = df_rejected[df_rejected.percentage<percent_reject].drop_duplicates(subset=['Cruise']).groupby('position type').count()['Cruise']['ARGOS']
iridium_number_low = df_rejected[df_rejected.percentage<percent_reject].drop_duplicates(subset=['Cruise']).groupby('position type').count()['Cruise']['GPS']

ind = (1,2)
ax1.bar(ind,(argos_number,iridium_number),label='All Floats')
ax1.bar(ind,(argos_number_low,iridium_number_low),color='r',alpha=.5,label='5% Rejected')
ax1.set_ylabel('Number')
ax1.set_title('Number of rejected floats by positioning type')
ax1.set_xticks(ind)
ax1.set_xticklabels(('ARGOS', 'GPS'))
plt.legend()
plt.savefig(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/raw_data_statistics/number_of_floats_by_positioning_type.png')
plt.close()

fig2, ax2 = plt.subplots()
df_rejected.drop_duplicates(subset=['Cruise']).percentage.hist(bins=50,label='All floats')
df_rejected[df_rejected.percentage<percent_reject].drop_duplicates(subset=['Cruise']).percentage.hist(bins=5,color='r',alpha=0.5,label='5% Rejected')
plt.xlabel('Percentage of bad displacements')
plt.ylabel('Number of floats')
plt.title('Histogram of percentage of bad dispacements by float')
plt.legend()
plt.savefig(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/raw_data_statistics/percentage_of_bad_displacements.png')
plt.close()


if (df_token[df_token.Cruise==cruise].Speed.dropna()>test_value).any():
	self.f.write(str(cruise)+' is suspicious because the maximum calculated speed is '+str(df_token[df_token.Cruise==cruise].Speed.dropna().max())+' which is outside the '+str(test_value)+' km/hour tolerance for the grid box \n')

df = pd.concat([df_rejected[(df_rejected.percentage<percent_reject)&(df_rejected.reject==False)],df]) #We allow some redemption if less than 5% of displacements are messed up
df_rejected = df_rejected[~df_rejected.Cruise.isin(df.Cruise.unique())]
############ Plot the data ######################
fig, ax = plt.subplots()
for number in [(1,1),(2,2),(3,3),(4,4)]:
	dummy_lat = np.arange(-90,90.1,number[0]).tolist()
	dummy_lon = np.arange(-180,180.1,number[1]).tolist()
	df['bins_lat'] = pd.cut(df.Lat,bins = dummy_lat,labels=dummy_lat[:-1])
	df['bins_lon'] = pd.cut(df.Lon,bins = dummy_lon,labels=dummy_lon[:-1])
	assert (~df['bins_lat'].isnull()).all()
	assert (~df['bins_lon'].isnull()).all()
	df['bin_index'] = zip(df['bins_lat'].values,df['bins_lon'].values)
	df.groupby('bin_index').count()['Cruise'].hist(label='lat '+str(number[0])+', lon '+str(number[1])+' degree bins',alpha=0.5,bins=500)
plt.xlim([0,2000])
ax.set_yscale('log')
plt.ylabel('Number of bins')
plt.xlabel('Number of displacements')
plt.legend()
plt.savefig(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/raw_data_statistics/number_displacements_degree_bin_1.png')
plt.close()


fig, ax = plt.subplots()
for number in [(1,3),(1.5,4.5),(2,6),(2,4)]:
	dummy_lat = np.arange(-90,90.1,number[0]).tolist()
	dummy_lon = np.arange(-180,180.1,number[1]).tolist()
	df['bins_lat'] = pd.cut(df.Lat,bins = dummy_lat,labels=dummy_lat[:-1])
	df['bins_lon'] = pd.cut(df.Lon,bins = dummy_lon,labels=dummy_lon[:-1])
	assert (~df['bins_lat'].isnull()).all()
	assert (~df['bins_lon'].isnull()).all()
	df['bin_index'] = zip(df['bins_lat'].values,df['bins_lon'].values)
	df.groupby('bin_index').count()['Cruise'].hist(label='lat '+str(number[0])+', lon '+str(number[1])+' degree bins',alpha=0.5,bins=500)
plt.xlim([0,2000])
ax.set_yscale('log')
plt.ylabel('Number of bins')
plt.xlabel('Number of displacements')
plt.legend()
plt.savefig(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/raw_data_statistics/number_displacements_degree_bin_2.png')
plt.close()

############## Tests ################
# assert df.pres.min()>800
# assert df.pres.max()<1200
assert (~df.reject).all()	# reject flag must be false for all floats
assert (df.dropna(subset=['Speed'])['Lon Speed']<(max_speed-0.2)).all()
assert (df.dropna(subset=['Speed'])['Lat Speed']<(max_speed-0.2)).all()
# assert len(matches) == len(df.Cruise.unique())+len(df_rejected.Cruise.unique()) #total number of floats must be the same


############# Save ###############
f.close()
df.to_pickle(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/all_argo_traj.pickle')