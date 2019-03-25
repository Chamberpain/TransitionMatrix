import numpy as np
from netCDF4 import Dataset
import fnmatch
import sys,os
import pickle
import time
import matplotlib.pyplot as plt

def find_nearest(items, pivot):
	return min(items, key=lambda x: abs(x - pivot))


try:
	lat_list = np.load('lat_list.npy')
	lon_list = np.load('lon_list.npy')

except IOError:
	nc_fid_token = Dataset('/data/SO12/CM2p6/ocean_scalar.static.nc')
	depth = nc_fid_token.variables['ht'][:]
	mask = nc_fid_token.variables['ht'][:]>500
	lon_token = nc_fid_token.variables['xt_ocean'][:]
	lat_token = nc_fid_token.variables['yt_ocean'][:]
	X,Y = np.meshgrid(lon_token,lat_token)
	lon_list = X[mask]
	lat_list = Y[mask]
	rounded_lat_list = np.arange(-81,89.6,0.5)
	rounded_lon_list = np.arange(-279.5,80.1,0.5)
	target_lat_list = [find_nearest(lat_token,x) for x in rounded_lat_list]
	target_lon_list = [find_nearest(lon_token,x) for x in rounded_lon_list]
	subsample_mask = np.isin(lon_list,target_lon_list)&np.isin(lat_list,target_lat_list)
	lat_list = lat_list[subsample_mask]
	lon_list = lon_list[subsample_mask]
	lat_list = np.array([find_nearest(rounded_lat_list,x) for x in lat_list])
	lon_list = np.array([find_nearest(rounded_lon_list,x) for x in lon_list])

	data_directory = '/data/SO12/CM2p6/ocean_minibling_surf_flux'

	degree_cutoff = 12  #so that we dont calculate unphysical correlation scales


try:
	o2 = np.load('subsampled_o2.npy')
	pco2 = np.load('subsampled_pco2.npy')

except IOError:
	matches = []
	for root, dirnames, filenames in os.walk(data_directory):
		for filename in fnmatch.filter(filenames, '*.nc'):
			matches.append(os.path.join(root, filename))	
	pco2_list = []
	o2_list = []
	for n, match in enumerate(matches):
		print 'file is ',match,', there are ',len(matches[:])-n,'files left'
		nc_fid = Dataset(match, 'r')
		for k in range(nc_fid.variables['pco2'].shape[0]):
			print 'day ',k
			matrix_holder = nc_fid.variables['pco2'][k,:,:][mask]
			pco2_list.append(matrix_holder[subsample_mask].data)
			matrix_holder = nc_fid.variables['o2_saturation'][k,:,:][mask]
			o2_list.append(matrix_holder.flatten()[subsample_mask].data)

	o2 = np.vstack(o2_list)
	np.save('subsampled_o2',o2)
	pco2 = np.vstack(pco2_list)
	np.save('subsampled_pco2',pco2)


def recompile_array(file_name_,matrix_,n,lon_index_list,lat_index_list):
	np.save(file_name_,np.zeros([1]))
	base_timeseries = matrix_[:,n]
	array = np.zeros([49,49])

	for i,lon in enumerate(lon_index_list):
		lon_mask = (lon_list==lon)
		for j,lat in enumerate(lat_index_list):
			lat_mask = (lat_list==lat)
			mask = lat_mask&lon_mask
			if ~mask.any():
				continue
			token_timeseries = matrix_[:,mask.tolist().index(True)]
			cor = np.corrcoef(base_timeseries,token_timeseries)[0,1]
			array[i,j]=cor	
	np.save(file_name_,array)
	return array


def correlation_calc(matrix_,name_,plot=False):
	array_list = []
	len_= len(lat_list)
	for n in range(len(lat_list)):
		print 'I am ',float((n+1))/len_,' percent done'

		start = time.time()
		base_lat = lat_list[n]
		base_lon = lon_list[n]
		print 'lat is ',base_lat
		print 'lon is ',base_lon
		file_name_ = './'+name_+'/'+str(base_lat)+'_'+str(base_lon)+'.npy'

		lat_index_list = np.arange(base_lat-12,base_lat+12.1,0.5)
		assert len(lat_index_list)==49
		lon_index_list = np.arange(base_lon-12,base_lon+12.1,0.5)
		lon_index_list[lon_index_list<-279.5] = lon_index_list[lon_index_list<-279.5]+360 
		lon_index_list[lon_index_list>80] = lon_index_list[lon_index_list>80]-360 
		assert len(lon_index_list)==49

		if (base_lat % 1 != 0 )|(base_lon %1 !=0): # only consider whole degree values
			end = time.time()
			print 'the position was not a whole degree, looping'
			print 'subroutine took ',end-start
			continue
		try:
			array = np.load(file_name_)
		except IOError:
			print 'the array was not found, we are recompiling'
			array = recompile_array(file_name_,matrix_,n,lon_index_list,lat_index_list)
		if array.shape!=(49,49):
			array = recompile_array(file_name_,matrix_,n,lon_index_list,lat_index_list)
		assert array.shape==(49,49)
		assert ~(array==0).all()
		array_list.append(array)
		end = time.time()
		print 'subroutine took ',end-start
		if plot:
			X,Y = np.meshgrid(lon_index_list,lat_index_list)
			plt.pcolor(X,Y,np.ma.masked_equal(array.T,0),vmin=0,vmax=1)
			# plt.yticks(lat_index_list)
			# plt.xticks(lon_index_list)
			plt.title('Correlation at '+str(base_lat)+' Lat, '+str(base_lon)+' Lon')
			plt.colorbar()
			plt.savefig(file_name_[:-4]+'.png')
			plt.close()
	return np.stack(array_list,axis=0)

# dummy_array= correlation_calc(pco2,'pco2')
# np.save('pco2_corr',dummy_array)
dummy_array= correlation_calc(o2,'o2')
np.save('o2_corr',dummy_array)