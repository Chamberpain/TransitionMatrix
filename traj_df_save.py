import sys,os
import fnmatch
import datetime
import scipy.io
import pandas as pd
import time
import numpy as np
import oceans
import matplotlib.pyplot as plt

"""file to create pandas dataframe from trajectory data"""
max_speed = 5
percent_reject = 0.05

f = open('traj_df_changelog.txt','w') #this is to keep a record of the floats that have been rejected
degree_dist = 111.7	#The km distance of one degree

def time_parser(juld_list):
	ref_date = datetime.datetime(1950,1,1,1,1)
	return [ref_date + datetime.timedelta(days=x) for x in juld_list]

def frame_return(file_name):
	traj_dict = scipy.io.loadmat(file_name)
	lon_i = traj_dict['subxlon_i'].flatten().tolist() #actual last longitude of previous cycle for subsurface velocity estimates
	lon_i = oceans.wrap_lon180(lon_i)
	lon_f = traj_dict['subxlon_f'].flatten().tolist() #actual first longitude of current cycle for subsurface velocity estimates
	lon_f = oceans.wrap_lon180(lon_f)
	lat_i = traj_dict['subxlat_i'].flatten().tolist() #actual last latitude of previous cycle for subsurface velocity estimates
	lat_f = traj_dict['subxlat_f'].flatten().tolist() #actual first latitude of current cycle for subsurface velocity estimates
	pres = traj_dict['subpres'].flatten().tolist() #best estimate of drift pressure
	date_i = time_parser(traj_dict['subxjday_i'].flatten().tolist()) #actual last time of previous cycle's surface period for subsurface velocity estimates
	date_f = time_parser(traj_dict['subxjday_f'].flatten().tolist()) #actual first time of current cycle's surface period for subsurface velocity estimates

	cruise = traj_dict['subwmo'][0]	#this is the WMO id number of the cruise
	positioning_type = traj_dict['subtrans_sys'][0]	#this is the positioning type used for the cruise
	assert ~all(v == len(lon_i) for v in [len(lon_i),len(lon_f),len(lat_i),len(lat_f),len(date_i),len(date_f),len(pres)])	# all lists have to be the same length
	df_holder = pd.DataFrame({'position type':positioning_type,'Cruise':cruise,'lon_i':lon_i,'Lon':lon_f,'lat_i':lat_i,'Lat':lat_f,'date_i':date_i,'Date':date_f,'pres':pres})

	if df_holder.empty:
		f.write(str(cruise)+' was empty \n')
		return df_holder
	df_holder['hour_diff'] = df_holder.Date.diff().dt.days*24+df_holder.Date.diff().dt.seconds/3600.
	if (df_holder.hour_diff.tail(len(df_holder)-1)<0).any():
		df_holder = df_holder.sort_values(['Date']).reset_index(drop=True)
		df_holder['hour_diff'] = df_holder.Date.diff().dt.days*24+df_holder.Date.diff().dt.seconds/3600.

		f.write(str(cruise)+' resorted because date order not correct \n')
		print 'I am resorting the dates because they were not correct'
	assert (df_holder.hour_diff.tail(len(df_holder)-1)>0).all()	#the first hour diff will be zero and must be filtered out.
	df_holder['Lat Diff'] = df_holder.Lat.diff()
	df_holder['Lat Speed']= df_holder['Lat Diff']/df_holder['hour_diff']*degree_dist
	df_holder['Lon Diff']=df_holder.Lon.diff().abs()
	if not df_holder[df_holder['Lon Diff']>180].empty:
		df_holder.loc[df_holder['Lon Diff']>180,'Lon Diff']=(df_holder[df_holder['Lon Diff']>180]['Lon Diff']-360).abs()
	df_holder['Lon Speed']= np.cos(np.deg2rad(df_holder.Lat))*df_holder['Lon Diff']/df_holder['hour_diff']*degree_dist
	df_holder['Speed'] = np.sqrt(df_holder['Lat Speed']**2 + df_holder['Lon Speed']**2)
	if (df_holder.Speed>max_speed).any():
		f.write(str(cruise)+' is rejected because the calculated speed is outside the '+str(max_speed)+' km/hour tolerance \n')
	assert (df_holder.dropna(subset=['Lon Diff'])['Lon Diff']<180).all()
	assert (df_holder.dropna(subset=['Lat Diff'])['Lat Diff']<180).all()
	assert (df_holder.dropna(subset=['Speed']).Speed>=0).all()
	assert df_holder.Lon.min() >= -180
	assert df_holder.Lon.max() <= 180
	assert df_holder.Lat.max() <= 90
	assert df_holder.Lat.min() >=-90
	assert (df_holder.dropna(subset=['hour_diff'])['hour_diff']>0).all()
	return df_holder

data_directory = os.getenv("HOME")+'/iCloud/Data/Raw/Traj/mat/'
frames = []
matches = []
float_type = ['Argo']
for root, dirnames, filenames in os.walk(data_directory):
	for filename in fnmatch.filter(filenames, '*.mat'):
		matches.append(os.path.join(root, filename))
for n, match in enumerate(matches):
	print 'file is ',match,', there are ',len(matches[:])-n,'floats left'
	t = time.time()
	frames.append(frame_return(match))
	print 'Building and merging datasets took ', time.time()-t
df = pd.concat(frames)
df.loc[df['position type'].isin(['GPS     ','IRIDIUM ']),'position type'] = 'GPS'
df.loc[df['position type'].isin(['ARGOS   ']),'position type'] = 'ARGOS'
df['Lon'] = oceans.wrap_lon180(df.Lon) 


dummy_lat = np.arange(-90,90.1,4).tolist()
dummy_lon = np.arange(-180,180.1,4).tolist()
df['bins_lat'] = pd.cut(df.Lat,bins = dummy_lat,labels=dummy_lat[:-1])
df['bins_lon'] = pd.cut(df.Lon,bins = dummy_lon,labels=dummy_lon[:-1])
df['bin_index'] = zip(df['bins_lat'].values,df['bins_lon'].values)
assert (~df['bins_lat'].isnull()).all()
assert (~df['bins_lon'].isnull()).all()

df['reject'] = False
df.loc[df.Speed>5,'reject']=True # reject all floats with a speed above 6 km/hour
k  = len (df.bin_index.unique())
for n,index_bin in enumerate(df.bin_index.unique()):
	print 'we have ',k-n,' index bins remaining'
	df_token = df[df.bin_index==index_bin]
	for cruise in df_token.Cruise.unique():
		test_value = 60*df_token[df_token.Cruise!=cruise].Speed.var()+df_token[df_token.Cruise!=cruise].Speed.mean()
		if (df_token[df_token.Cruise==cruise].Speed.dropna()>test_value).any():
			f.write(str(cruise)+' is rejected because the maximum calculated speed is '+str(df_token[df_token.Cruise==cruise].Speed.dropna().max())+' which is outside the '+str(test_value)+' km/hour tolerance for the grid box \n')
			df.loc[(df.Cruise==cruise)&(df.Speed>test_value)&(df.bin_index==index_bin),'reject']=True

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
	f.write(str(cruise)+' is suspicious because the maximum calculated speed is '+str(df_token[df_token.Cruise==cruise].Speed.dropna().max())+' which is outside the '+str(test_value)+' km/hour tolerance for the grid box \n')

df = pd.concat([df_rejected[(df_rejected.percentage<percent_reject)&(df_rejected.reject==False)],df]) #We allow some redemption if less than 5% of displacements are messed up
df_rejected = df_rejected[~df_rejected.Cruise.isin(df.Cruise.unique())]
############ Plot the data ######################
fig, ax = plt.subplots()
for number in [1+x for x in range(4)]:
	dummy_lat = np.arange(-90,90.1,number).tolist()
	dummy_lon = np.arange(-180,180.1,number).tolist()
	df['bins_lat'] = pd.cut(df.Lat,bins = dummy_lat,labels=dummy_lat[:-1])
	df['bins_lon'] = pd.cut(df.Lon,bins = dummy_lon,labels=dummy_lon[:-1])
	assert (~df['bins_lat'].isnull()).all()
	assert (~df['bins_lon'].isnull()).all()
	df['bin_index'] = zip(df['bins_lat'].values,df['bins_lon'].values)
	df.groupby('bin_index').count()['Cruise'].hist(label=str(number)+' degree bins',alpha=0.5,bins=500)
plt.xlim([0,2000])
ax.set_yscale('log')
plt.ylabel('Number of bins')
plt.xlabel('Number of displacements')
plt.legend()
plt.savefig(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/raw_data_statistics/number_displacements_degree_bin.png')
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