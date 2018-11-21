import numpy as np                                                                                                              
from netCDF4 import Dataset                                                         
import fnmatch
import sys,os                                                                                                                   
import pickle

############
########## IF THIS IS EVER REDONE, INSTEAD OF SAVING ARRAYS, JUST SAVE THE CORRELATION VALUES RELATIVE TO THE STARTING LOCATION ###########
############



nc_fid = Dataset('/data/soccom/CM2p6/ocean_scalar.static.nc')
mask = nc_fid['ht'][:]<2000
data_directory = '/data/soccom/CM2p6/ocean_minibling_surf_flux'
pco2_list = []
o2_list = []
matches = []

degree_cutoff = 8  #so that we dont calculate unphysical correlation scales
lon_list = np.arange(-279.5,80.5,1)
lat_list = np.arange(-80.5,89.5,1)
def find_nearest(items, pivot):                                                                                                 
	return min(items, key=lambda x: abs(x - pivot)) 
try:
	o2 = np.load('subsampled_o2')
	pco2 = np.load('subsampled_pco2')




except IOError:
	for root, dirnames, filenames in os.walk(data_directory):
		for filename in fnmatch.filter(filenames, '*.nc'):
			matches.append(os.path.join(root, filename))	
	pco2_list = []
	o2_list = []
	for n, match in enumerate(matches):
		print 'file is ',match,', there are ',len(matches[:])-n,'files left'
		nc_fid = Dataset(match, 'r')


		lat_index_list = [nc_fid['yt_ocean'][:].tolist().index(find_nearest(nc_fid['yt_ocean'][:],x)) for x in lat_list]
		lon_index_list = [nc_fid['xt_ocean'][:].tolist().index(find_nearest(nc_fid['xt_ocean'][:],x)) for x in lon_list]

		#need to include logic to mask 2000 meters

		holder_pco2 = nc_fid['pco2'][:].take(lon_index_list, axis=2, mode='clip').take(lat_index_list, axis=1, mode='clip')
		print 'pco2 loaded'
		# take is much faster than array slicing
		pco2_list.append(holder_pco2)

		holder_o2 = nc_fid['o2_saturation'][:].take(lon_index_list, axis=2, mode='clip').take(lat_index_list, axis=1, mode='clip')
		print 'o2 loaded'
		# take is much faster than array slicing
		o2_list.append(holder_pco2)
	o2 = np.ma.concatenate(o2_list)
	pco2 = np.ma.concatenate(pco2_list)

	o2.dump('subsampled_o2')
	pco2.dump('subsampled_pco2')


def correlation_calc(matrix_):
	position_list = []
	array_list = []
	for i,lat_l in enumerate(lat_list):
		for k,lon_l in enumerate(lon_list):
			print 'i is ',i,' k is ',k
			if matrix_[:,i,k].mask.any():
				print 'we found masked values in this array and are looping on'
				continue	
			base_list = matrix_[:,i,k]
			base_pos = (lat_l,lon_l)
			lat_difference_list = abs(lat_list - lat_l) 
			lon_difference_list = abs(lon_list - lon_l) 
			lon_difference_list[lon_difference_list>180] = abs(lon_difference_list[lon_difference_list>180]-360)
			lon_mask = np.where((lon_difference_list<degree_cutoff)&(lon_difference_list>0))[0]
			lat_mask = np.where((lat_difference_list<degree_cutoff)&(lat_difference_list>0))[0]
			lon_holder = lon_list[lon_mask]
			lat_holder = lat_list[lat_mask]
			token_matrix_ = matrix_.take(lon_mask, axis=2, mode='clip').take(lat_mask, axis=1, mode='clip')
			for q,lat_h in enumerate(lat_holder):
				for p,lon_h in enumerate(lon_holder):
						cor = np.corrcoef(base_list,token_matrix_[:,q,p])[0,1]
						if np.isnan(cor):
							print 'we found mask values in the correlation array and are looping on'
							continue
						array_list.append(cor)
						position_list.append((base_pos,(lat_h,lon_h)))
	return (array_list,position_list)
pco2_array_list, pco2_position_list = correlation_calc(pco2)
with open('pco2_array_list', 'wb') as fp:
    pickle.dump(pco2_array_list, fp)
with open('pco2_position_list', 'wb') as fp:
    pickle.dump(pco2_position_list, fp)

o2_array_list, o2_position_list = correlation_calc(o2)
with open('o2_array_list', 'wb') as fp:
    pickle.dump(o2_array_list, fp)
with open('o2_position_list', 'wb') as fp:
    pickle.dump(o2_position_list, fp)