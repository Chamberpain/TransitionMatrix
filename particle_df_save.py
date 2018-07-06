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

# print "data has %i records" %(opt.shape[0])


#plot some trajectories
x,y=opt[:,0,:10],opt[:,1,:10] #this is in model grid index coordinate, convert to lat-lon using x=x/6.0;y=y/6.0-77.875

x=x/6.0;y=y/6.0-77.875
x = x%360
plt.plot(x,y,'-')
plt.xlabel('x')
plt.ylabel('y')
plt.show()