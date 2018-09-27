import sys,os
import fnmatch
import datetime
import scipy.io
import pandas as pd
import time
import numpy as np
import oceans

"""file to create pandas dataframe from trajectory data"""

f = open('traj_df_changelog.txt','w') #this is to keep a record of the floats that have been rejected
degree_dist = 111.7	#The km distance of one degree

def time_parser(juld_list):
	ref_date = datetime.datetime(1950,1,1,1,1)
	return [ref_date + datetime.timedelta(days=x) for x in juld_list]

def frame_return(file_name):
	traj_dict = scipy.io.loadmat(file_name)
	lon_i = traj_dict['subxlon_i'].flatten().tolist() #actual last longitude of previous cycle for subsurface velocity estimates
	lon_f = traj_dict['subxlon_f'].flatten().tolist() #actual first longitude of current cycle for subsurface velocity estimates
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
	df_holder['Lat Speed']= df_holder.Lat.diff()/df_holder['hour_diff']*degree_dist
	df_holder['Lon Diff']=df_holder.Lon.diff().abs()
	if not df_holder[df_holder['Lon Diff']>180].empty:
		df_holder.loc[df_holder['Lon Diff']>180,'Lon Diff']=df_holder[df_holder['Lon Diff']>180]['Lon Diff']-360
	df_holder['Lon Speed']= np.cos(np.deg2rad(df_holder.Lat))*df_holder['Lon Diff']/df_holder['hour_diff']*degree_dist
	df_holder['Speed'] = np.sqrt(df_holder['Lat Speed']**2 + df_holder['Lon Speed']**2)
	if (df_holder.Speed>6).any():
		print df_holder
		print 'Min time diff in days is ',df_holder.Date.diff().dt.days.min()
		print 'Min time diff in seconds is ',df_holder.Date.diff().dt.seconds.min()
		print 'Min time diff in hours is ',(df_holder.Date.diff().dt.seconds/3600.).min()
		print 'Max lon diff is ',df_holder['Lon Diff'].max()		
		print 'Max lat diff is ',df_holder.Lat.diff().abs().max()
		print 'Max lat speed is ',df_holder['Lat Speed'].abs().max()
		print 'Max lon speed is ',df_holder['Lon Speed'].abs().max()
		print 'Max speed is ',df_holder['Speed'].abs().max()
		print 'I am returning the empty set'
		f.write(str(cruise)+' is rejected because the calculated speed is outside the 10 km/hour tolerance \n')
		return pd.DataFrame()
	else:
		return df_holder

data_directory = os.getenv("HOME")+'/iCloud/Data/Raw/Traj/mat/'
frames = []
matches = []
float_type = ['Argo']
for root, dirnames, filenames in os.walk(data_directory):
	for filename in fnmatch.filter(filenames, '*.mat'):
		matches.append(os.path.join(root, filename))
for n, match in enumerate(matches):
	try:
		print 'file is ',match,', there are ',len(matches[:])-n,'floats left'
		t = time.time()
		frames.append(frame_return(match))
		print 'Building and merging datasets took ', time.time()-t
	except: 
		raise
df = pd.concat(frames)
df.loc[df['position type'].isin(['GPS     ','IRIDIUM ']),'position type'] = 'GPS'
df.loc[df['position type'].isin(['ARGOS   ']),'position type'] = 'ARGOS'
df['Lon'] = oceans.wrap_lon180(df.Lon) 
############## Tests ################
# assert df.pres.min()>800
# assert df.pres.max()<1200

assert (df.dropna(subset=['Speed']).Speed>=0).all()
assert df.Lon.min() >= -180
assert df.Lon.max() <= 180
assert df.Lat.max() <= 90
assert df.Lat.min() >=-90

bins_lat = np.arange(-90,90.1,2).tolist()
bins_lon = np.arange(-180,180.1,2).tolist()
df['bins_lat'] = pd.cut(df.Lat,bins = bins_lat,labels=bins_lat[:-1])
df['bins_lon'] = pd.cut(df.Lon,bins = bins_lon,labels=bins_lon[:-1])


reject_list = []
for lat_bin in df.bins_lat.unique():
	print 'we are on lat bin ',lat_bin
	if (df.bins_lat==lat_bin).empty:
		continue
	for lon_bin in df.bins_lon.unique():
		df_token = df[(df.bins_lat==lat_bin)&(df.bins_lon==lon_bin)]
		if df_token.empty:
			continue
		for cruise in df_token.Cruise.unique():

			test_value = 100*df_token[df_token.Cruise!=cruise].Speed.var()+df_token[df_token.Cruise!=cruise].Speed.mean()
			# print 'test value is ',test_value
			# print 'max speed is ',df_token[df_token.Cruise==cruise].Speed.max()

			if (df_token[df_token.Cruise==cruise].Speed.dropna()>test_value).any():
				f.write(str(cruise)+' is rejected because the maximum calculated speed is '+str(df_token[df_token.Cruise==cruise].Speed.dropna().max())+' which is outside the '+str(test_value)+' km/hour tolerance for the grid box \n')
				reject_list.append(cruise)
df = df[~df.Cruise.isin(reject_list)]
############# Save ###############
f.close()
df.to_pickle(os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/all_argo_traj.pickle')