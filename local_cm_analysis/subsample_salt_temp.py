from netCDF4 import Dataset
import numpy as np 
import os

dir_path = '/data/SO12/pchamber/cm2p6/'

time_list = []
salt_list = []
temp_list = []

mask = np.load('mask.npy')
subsample_mask = np.load('subsample_mask.npy')


for fd in os.listdir(dir_path):
	nc_fid = Dataset(dir_path+fd)
	for k in range(nc_fid.variables['salt'].shape[0]):
#9 corresponds to 100m
		matrix_holder = nc_fid.variables['temp'][k,9,:,:][mask]
		temp_list.append(matrix_holder.flatten()[subsample_mask].data)
		matrix_holder = nc_fid.variables['salt'][k,9,:,:][mask]
		salt_list.append(matrix_holder.flatten()[subsample_mask].data)	
		time_list.append(nc_fid.variables['time'][:][k])

idx_list = [time_list.index(_) for _ in sorted(time_list)]
salt = np.array(salt_list)[idx_list]
temp = np.array(temp_list)[idx_list]
time = np.array(time_list)[idx_list]

np.save('subsampled_temp',temp)
np.save('subsampled_salt',salt)
np.save('time_list',time)