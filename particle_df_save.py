import os
import pandas as pd
import numpy as np


omega = 7.292*10**-5

data_file_name = os.getenv("HOME")+'/iCloud/Data/Raw/SOSE/particle_release/SO_RAND_0001.XYZ.0000000001.0003153601.data'
df_output_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose_particle_df.pickle'
npts= 10000

opt=np.fromfile(data_file_name,'>f4').reshape(-1,3,npts)

print "data has %i records" %(opt.shape[0])

frames = []
for k in range(opt.shape[2]):
	x,y=opt[:,0,k],opt[:,1,k]#this is in model grid index coordinate, convert to lat-lon using x=x/6.0;y=y/6.0-77.875
	t = range(opt.shape[0])
	x=x/6.0;y=y/6.0-77.875
	x = x%360
	frames.append(pd.DataFrame({'Lat':y,'Lon':x,'Date':t}))
	frames[-1]['Cruise']=k
df = pd.concat(frames)
df['position type'] = 'SOSE'
df.to_pickle(df_output_file_name)
