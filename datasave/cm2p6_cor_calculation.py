import numpy as np
from netCDF4 import Dataset
import fnmatch
import sys,os
import pickle
import time


def find_nearest(items, pivot):
	return min(items, key=lambda x: abs(x - pivot))

nc_fid_token = Dataset('/data/SO12/CM2p6/ocean_scalar.static.nc')
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
np.save('lat_list.npy',lat_list)
np.save('lon_list.npy',lon_list)



try:
	o2_surf = np.load('subsampled_o2_surf.npy')
	pco2_surf = np.load('subsampled_pco2_surf.npy')
	dic_surf = np.load('subsampled_dic_surf.npy')

	o2_100m = np.load('subsampled_o2_100m.npy')
	dic_100m = np.load('subsampled_dic_100m.npy')

except IOError:
	data_directory_list = [('surf','/data/SO12/CM2p6/ocean_minibling_surf_flux/'),('100m','/data/SO12/CM2p6/ocean_minibling_100m/')]
	degree_cutoff = 12  #so that we dont calculate unphysical correlation scales
	for label,data_directory in data_directory_list:
		matches = []
		for root, dirnames, filenames in os.walk(data_directory):
			for filename in fnmatch.filter(filenames, '*.nc'):
				matches.append(os.path.join(root, filename))	
		pco2_list = []
		o2_list = []
		dic_list = []
		for n, match in enumerate(matches):
			print 'file is ',match,', there are ',len(matches[:])-n,'files left'
			nc_fid = Dataset(match, 'r')
			if label == 'surf':
				for k in range(nc_fid.variables['pco2'].shape[0]):
					print 'day ',k
					matrix_holder = nc_fid.variables['pco2'][k,:,:][mask]
					pco2_list.append(matrix_holder.flatten()[subsample_mask].data)
					matrix_holder = nc_fid.variables['o2_saturation'][k,:,:][mask]
					o2_list.append(matrix_holder.flatten()[subsample_mask].data)
					matrix_holder = nc_fid.variables['dic_stf'][k,:,:][mask]
					dic_list.append(matrix_holder.flatten()[subsample_mask].data)
			if label == '100m':
				for k in range(nc_fid.variables['o2'].shape[0]):
					print 'day ',k
					matrix_holder = nc_fid.variables['o2'][k,:,:][mask]
					o2_list.append(matrix_holder.flatten()[subsample_mask].data)
					matrix_holder = nc_fid.variables['dic'][k,:,:][mask]
					dic_list.append(matrix_holder.flatten()[subsample_mask].data)				
		try:
			o2 = np.vstack(o2_list)
			np.save('subsampled_o2_'+label,o2)
		except KeyError:
			pass
		try:
			pco2 = np.vstack(pco2_list)
			np.save('subsampled_pco2_'+label,pco2)
		except (KeyError,ValueError) as e:
			pass
		try:	
			dic = np.vstack(dic_list)
			np.save('subsampled_dic_'+label,dic)
		except KeyError:
			pass

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


def correlation_calc(matrix_,name_,plot=False,scale_factor=0.05):
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
			Y,X = np.meshgrid(lat_index_list,lon_index_list)
			m = Basemap(projection='cea',llcrnrlat=min(lat_index_list),urcrnrlat=max(lat_index_list),\
			llcrnrlon=min(lon_index_list),urcrnrlon=max(lon_index_list),resolution='c')

			x = np.multiply(array,X).flatten()
			y = np.multiply(array,Y).flatten()

			mask = (x!=0)|(y!=0) 
			x = x[mask]
			y = y[mask]
			w,v = np.linalg.eig(np.cov(x,y))

			angle = np.degrees(np.arctan(v[1,np.argmax(w)]/v[0,np.argmax(w)]))

			axis1 = 2*max(w)*np.sqrt(5.991)

			axis2 = 2*min(w)*np.sqrt(5.991)

			try:
				poly = m.ellipse(base_lon,base_lat, axis1*scale_factor,axis2*scale_factor, 80,phi=angle, facecolor='red', zorder=3,alpha=0.35)
			except ValueError:
				pass

			m.pcolor(X,Y,np.ma.masked_equal(array,0),vmin=0,vmax=1,latlon=True)
			plt.title('Correlation at '+str(base_lat)+' Lat, '+str(base_lon)+' Lon')
			plt.colorbar()
			plt.savefig(file_name_[:-4]+'.png')
			plt.close()
	return np.stack(array_list,axis=0)



dummy_array= correlation_calc(pco2_surf,'pco2_surf',plot=False)
np.save('pco2_surf_corr',dummy_array)
dummy_array= correlation_calc(o2_surf,'o2_surf',plot=False)
np.save('o2_surf_corr',dummy_array)
dummy_array= correlation_calc(dic_surf,'dic_surf',plot=False)
np.save('dic_surf_corr',dummy_array)
dummy_array= correlation_calc(o2_100m,'o2_100m',plot=False)
np.save('o2_100m_corr',dummy_array)
dummy_array= correlation_calc(dic_100m,'dic_100m',plot=False)
np.save('dic_100m_corr',dummy_array)