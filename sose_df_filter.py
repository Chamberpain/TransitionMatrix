import pandas as pd
import numpy as np
import scipy.io
import sys,os
import oceans

def visual_check():
	x = df.Lon.values
	y = df.Lat.values
	depth = df.Depth.values
	plt.scatter(x[::20],y[::20],c=depth[::20])
	plt.show()

sose_df_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose_particle_df.pickle'
df = pd.read_pickle(sose_df_file_name)
df['Lon'] = oceans.wrap_lon360(df['Lon'])
assert df.Lon.min()>=0

mat = scipy.io.loadmat(grid_file_name)
XC = mat['XC'][:,0]
YC = mat['YC'][0,:]
Depth = mat['Depth']
df['bins_lat'] = pd.cut(df.Lat,bins=YC,labels=YC[:-1])
df['bins_lon'] = pd.cut(df.Lon,bins=XC,labels=XC[:-1])
df.loc[df.bins_lon.isna(),'bins_lon'] = XC[0] # this is for the values greater than 359.8 and less than 0.166
assert df[df.bins_lon.isna()].empty
for i,lon in enumerate(XC[:-1]):
	print 'lon = ',lon
	for j,lat in enumerate(YC[:-1]):
		df.loc[(df.bins_lat==lat)&(df.bins_lon==lon),'Depth']=Depth[i,j]

visual_check()

