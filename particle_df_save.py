import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

data_file_name = os.getenv("HOME")+'/iCloud/Data/Raw/SOSE/particle_release/SO_RAND_0001.XYZ.0000000001.0003153601.data'
grid_file_name = os.getenv("HOME")+'/iCloud/Data/Raw/SOSE/grid.mat'
output_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose_particle_df.pickle'
npts= 10000


mat = scipy.io.loadmat(grid_file_name)
XC = mat['XC'][:,0]
YC = mat['YC'][0,:]
Depth = mat['Depth']

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

####### this code is necessary because Isa accidentally deployed the particles on land ####
df['bins_lat'] = pd.cut(df.Lat,bins=YC,labels=YC[:-1])
df['bins_lon'] = pd.cut(df.Lon,bins=XC,labels=XC[:-1])

for i,lon in enumerate(XC[:-1]):
	print 'lon = ',lon
	for j,lat in enumerate(YC[:-1]):
		df.loc[(df.bins_lat==lat)&(df.bins_lon==lon),'Depth']=Depth[i,j]
