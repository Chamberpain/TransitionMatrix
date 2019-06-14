"""Script that gathers together all of the SOSE particle trajectory data and formats it into one dataframe

todo: Need to add routine to filter out floats in depths greater than 1000m (data isa provided did not do this)
"""

import os
import pandas as pd
import numpy as np
import oceans
import datetime

omega = 7.292*10**-5

data_file_name = os.getenv("HOME")+'/iCloud/Data/Raw/SOSE/particle_release/SO_RAND_0001.XYZ.0000000001.0003153601.data'
df_output_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose_particle_df.pickle'
npts= 10000
ref_date = datetime.date(year=1950,month=1,day=1)
opt=np.fromfile(data_file_name,'>f4').reshape(-1,3,npts)

print "data has %i records" %(opt.shape[0])

frames = []
for k in range(opt.shape[2]):
	x,y=opt[:,0,k],opt[:,1,k]#this is in model grid index coordinate, convert to lat-lon using x=x/6.0;y=y/6.0-77.875
	t = range(opt.shape[0])
	date=[ref_date+datetime.timedelta(days=j) for j in t]
	x=x/6.0;y=y/6.0-77.875
	x = x%360
	frames.append(pd.DataFrame({'Lat':y,'Lon':x,'Date':date}))
	frames[-1]['Cruise']=k
df = pd.concat(frames)
df['position type'] = 'SOSE'
df['Lon']=oceans.wrap_lon180(df.Lon) 

assert df.Lon.min() >= -180
assert df.Lon.max() <= 180
assert df.Lat.max() <= 90
assert df.Lat.min() >=-90
df.to_pickle(df_output_file_name)