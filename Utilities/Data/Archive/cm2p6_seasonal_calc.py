import numpy as np
from netCDF4 import Dataset
import fnmatch
import sys,os
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib.pyplot as plt
import datetime
import pandas as pd
import pickle
from scipy.fftpack import fft


base_filepath = '/home/pchamber/sub_sampled_cm2p6/'
# nc_fid = Dataset('/data/SO12/CM2p6/ocean_scalar.static.nc')
# mask = nc_fid['ht'][:]<2000
surf_dir = '/data/SO12/CM2p6/ocean_minibling_surf_flux/'
hundred_m_dir = '/data/SO12/CM2p6/ocean_minibling_100m/'


def find_nearest(items, pivot):
	return min(items, key=lambda x: abs(x - pivot))

x = np.arange(-280,80)
y = np.arange(-90,91)

data_directory_list = [('pco2',surf_dir),('o2',hundred_m_dir),('dic',hundred_m_dir),('po4',hundred_m_dir),('biomass_p',hundred_m_dir)]
for variable,directory in data_directory_list:
	matches = []
	for root, dirnames, filenames in os.walk(directory):
		for filename in fnmatch.filter(filenames, '*.nc'):
			matches.append(os.path.join(root, filename))
	date = [int(_.split('/')[-1].split('.')[0]) for _ in matches]
	class_list = [Dataset(match) for _,match in sorted(zip(date,matches))]
	save_list = []
	for k,token in enumerate(class_list):
		print(k)
		print(variable)
		if k ==0:
			x_list = class_list[0]['xt_ocean'][:].data
			y_list = class_list[0]['yt_ocean'][:].data
			x_idx = [x_list.tolist().index(find_nearest(x_list,_)) for _ in x]
			y_idx = [y_list.tolist().index(find_nearest(y_list,_)) for _ in y]
			TT,XX,YY = np.meshgrid(range(365),y_idx,x_idx,indexing='ij')
			idx = np.ravel_multi_index(np.array([TT.ravel().tolist(),XX.ravel().tolist(),YY.ravel().tolist()]),class_list[0][variable].shape) 
		save_array = token[variable][:].ravel()[idx]
		save_list.append(save_array.reshape(TT.shape))
	with open(variable+'.pkl', "wb") as fp:   #Pickling
		pickle.dump(save_list, fp)