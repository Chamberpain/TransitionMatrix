from __future__ import print_function
import numpy as np
from netCDF4 import Dataset
import fnmatch
import sys,os
import time
import matplotlib
import matplotlib.pyplot as plt
from compute_utilities.list_utilities import find_nearest, flat_list
from transition_matrix.makeplots.plot_utils import basemap_setup,transition_vector_to_plottable
import scipy.sparse.linalg
from scipy.sparse import csc_matrix
from transition_matrix.definitions import ROOT_DIR


def save_lat_lon():
	nc_fid_token = Dataset('/data/SO12/CM2p6/ocean_scalar.static.nc')
	mask = nc_fid_token.variables['ht'][:]>2000
	lon_token = nc_fid_token.variables['xt_ocean'][:]
	lat_token = nc_fid_token.variables['yt_ocean'][:]
	X,Y = np.meshgrid(lon_token,lat_token)
	lon_list = X[mask]
	lat_list = Y[mask]
	rounded_lat_list = np.arange(lat_token.min(),lat_token.max()+.5,0.5)
	rounded_lon_list = np.arange(lon_token.min(),lat_token.max()+.5,0.5)
	target_lat_list = [find_nearest(lat_token,x) for x in rounded_lat_list]
	target_lon_list = [find_nearest(lon_token,x) for x in rounded_lon_list]
	subsample_mask = np.isin(lon_list,target_lon_list)&np.isin(lat_list,target_lat_list)
	lat_list = lat_list[subsample_mask]
	lon_list = lon_list[subsample_mask]
	lat_list = np.array([find_nearest(rounded_lat_list,x) for x in lat_list])
	lon_list = np.array([find_nearest(rounded_lon_list,x) for x in lon_list])
	np.save('lat_list.npy',lat_list)
	np.save('lon_list.npy',lon_list)
	np.save('mask.npy',mask.data)
	np.save('subsample_mask.npy',subsample_mask)

def recompile_subsampled():
	def variable_extract(nc_,variable_list):
		matrix_holder = nc_[mask]
		variable_list.append(matrix_holder.flatten()[subsample_mask].data)		
		return variable_list

	mask = np.load('mask.npy')
	subsample_mask = np.load('subsample_mask.npy')

	surf_dir = '/data/SO12/CM2p6/ocean_minibling_surf_flux/'
	hun_m_dir = '/data/SO12/CM2p6/ocean_minibling_100m/'
	data_directory = '/data/SO12/pchamber/cm2p6/'
	matches = []

	for root, dirnames, filenames in os.walk(data_directory):
		for filename in fnmatch.filter(filenames, '*ocean.nc'):
			matches.append(os.path.join(root, filename))	

	for n, match in enumerate(matches):
		o2_list = []
		dic_list = []
		temp_list_100m = []
		salt_list_100m = []
		temp_list_surf = []
		salt_list_surf = []


		print('file is ',match,', there are ',len(matches[:])-n,'files left')
		file_date = match.split('/')[-1].split('.')[0]
		try:
			hun_m_fid = Dataset(hun_m_dir+file_date+'.ocean_minibling_100m.nc')
		except FileNotFoundError:
			try:
				hun_m_fid = Dataset(data_directory+file_date+'.ocean_minibling_100.nc')
			except OSError:
				print('There was a problem with '+data_directory+file_date+'.ocean_minibling_100.nc')
				print('continuing')
				continue
		hun_m_time = hun_m_fid.variables['time'][:]
		nc_fid = Dataset(match, 'r')
		nc_fid_time = nc_fid.variables['time'][:]
		time_idx = [hun_m_time.tolist().index(find_nearest(hun_m_time,t)) for t in nc_fid_time]
		for k in time_idx:
			print('day ',k)
			o2_list = variable_extract(hun_m_fid.variables['o2'][k,:,:],o2_list)
			dic_list = variable_extract(hun_m_fid.variables['dic'][k,:,:],dic_list)
		for k in range(len(nc_fid.variables['time'][:])):
#9 corresponds to 100m
			salt_list_100m = variable_extract(nc_fid.variables['salt'][k,9,:,:],salt_list_100m)
			temp_list_100m = variable_extract(nc_fid.variables['temp'][k,9,:,:],temp_list_100m)
			salt_list_surf = variable_extract(nc_fid.variables['salt'][k,0,:,:],salt_list_surf)
			temp_list_surf = variable_extract(nc_fid.variables['temp'][k,0,:,:],temp_list_surf)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_time',nc_fid_time.data)
		o2 = np.vstack(o2_list)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_100m_subsampled_o2',o2)
		dic = np.vstack(dic_list)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_100m_subsampled_dic',dic)
		salt_100m = np.vstack(salt_list_100m)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_100m_subsampled_salt',salt_100m)
		temp_100m = np.vstack(temp_list_100m)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_100m_subsampled_temp',temp_100m)
		salt_surf = np.vstack(salt_list_surf)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_surf_subsampled_salt',salt_surf)
		temp_surf = np.vstack(temp_list_surf)
		np.save('/data/SO12/pchamber/'+str(file_date)+'_surf_subsampled_temp',temp_surf)



class matrix_element(object):
	def __init__(self,_array,lats,lons,lat_grid,lon_grid,file_name):
		self.array = _array
		self.lats = lats
		self.lons = lons
		self.lat_grid = lat_grid
		self.lon_grid = lon_grid
		self.file_name = file_name

	def return_eig_vals(self):
		eigs = np.linalg.eigvals(self.array)
		return eigs

	def eig_val_plot(self):
		base = ROOT_DIR+'/plots/cm2p6_covariance/'


		eigs = self.return_eig_vals()
		fig = plt.figure()
		ax = fig.add_subplot(2, 1, 1)
		ax.plot(np.sort(eigs))
		plt.xlim([0,len(self.array)])
		plt.ylabel('Eig Value')
		plt.title('Positive Eigenvalues')
		ax.set_yscale('log')

		ax = fig.add_subplot(2, 1, 2)
		ax.plot(-1*np.sort(eigs))
		plt.xlim([0,len(self.array)])
		plt.xlabel('Number')
		plt.ylabel('Neg Eig Value')
		plt.title('Negative Eigenvalues')
		ax.set_yscale('log')
		plt.gca().invert_yaxis()
		plt.savefig(base+self.file_name+'_eig_vals')
		plt.close()

class cov_element(matrix_element):
	def __init__(self,_array,lats,lons,lat_grid,lon_grid,row_var,col_var,file_name):
		super().__init__(_array,lats,lons,lat_grid,lon_grid,file_name)
		self.row_var = row_var
		self.col_var = col_var
		eigs, eig_vecs = np.linalg.eig(self.array)
		eigs_sum_forward = np.array([eigs[:x].sum() for x in range(len(eigs))])
		eigs_idx = (eigs_sum_forward>0.99*eigs.sum()).tolist().index(True)
		matrix_holder = np.zeros(eig_vecs.shape)
		for idx in np.arange(eigs_idx):
			temp = matrix_holder+eigs[idx]*np.outer(eig_vecs[:,idx],eig_vecs[:,idx])
			matrix_holder = temp
		self._array = matrix_holder

class cov_array(object):
	data_directory = '/Users/pchamberlain/Data/Processed/transition_matrix/cm2p6/'
	def __init__(self,lon_sep,lat_sep,l=300):
		self.lats,self.lons,self.truth_array,self.lat_grid,self.lon_grid = self.lats_lons(lat_sep,lon_sep)
		self.lats = np.round(self.lats)
		self.lons = np.round(self.lons)
		self.file_name = 'lat_sep_'+str(lat_sep)+'_lon_sep_'+str(lon_sep)
		self.l = l
		self.lat_sep = lat_sep
		self.lon_sep = lon_sep


		o2_matches = []
		salt_matches = []
		dic_matches = []
		temp_matches = []

		def combine_data(file_format,data_directory):
			matches = []
			for root, dirnames, filenames in os.walk(data_directory):
				for filename in fnmatch.filter(filenames, file_format):
					matches.append(os.path.join(root, filename))
			
			array_list = [np.load(match)[:,self.truth_array] for match in sorted(matches)]
			return np.vstack(array_list)

		salt = combine_data('*100m_subsampled_salt.npy',self.data_directory)
		dic = combine_data('*100m_subsampled_dic.npy',self.data_directory)
		o2 = combine_data('*100m_subsampled_o2.npy',self.data_directory)
		temp = combine_data('*100m_subsampled_temp.npy',self.data_directory)

		file_format = '*_time.npy'
		matches = []
		for root, dirnames, filenames in os.walk(self.data_directory):
			for filename in fnmatch.filter(filenames, file_format):
				matches.append(os.path.join(root, filename))		
		self.time = flat_list([np.load(match).tolist() for match in sorted(matches)])


		def normalize_data(data,label=None,percent=0.1,plot=True):
			mean_removed = data-data.mean(axis=0)
			data_scale = mean_removed.std(axis=0)

			if plot:
				histogram, bins = np.histogram(data_scale)
				bin_centers = 0.5*(bins[1:] + bins[:-1])
				plt.figure(figsize=(6, 4))
				plt.plot(bin_centers, histogram)
				plt.xlabel('STD')
				plt.title('Histogram of STD of Ensemble')
				plt.savefig(label+'_std')
				plt.close()

			dummy = 0
			greater_mask = data_scale>(data_scale.max()-dummy*0.001*data_scale.mean())
			while greater_mask.sum()<percent*len(data_scale):
				dummy +=1
				value = data_scale.max()-dummy*0.001*data_scale.mean()
				greater_mask = data_scale>value
			# data_scale[greater_mask]=value
			print('greater value is '+str(value))
			dummy = 0
			lesser_mask = data_scale<(data_scale.min()+dummy*0.001*data_scale.mean())
			while lesser_mask.sum()<percent*len(data_scale):
				dummy +=1
				value = data_scale.min()+dummy*0.001*data_scale.mean()
				lesser_mask = data_scale<value
			# data_scale[lesser_mask]=value
			print('lesser value is '+str(value))
			data_scale[~greater_mask&~lesser_mask] = data_scale[~greater_mask&~lesser_mask]*0.7
			return mean_removed/data_scale

		salt = normalize_data(salt,'salt')
		dic = normalize_data(dic,'dic')
		o2 = normalize_data(o2,'o2')
		temp = normalize_data(temp,'temp')
		self.data = np.hstack([o2,dic,temp,salt])

		del salt
		del dic
		del o2
		del temp

		self.variable_list = ['o2','dic','temp','salt']
		
		def subtract_e_vecs_return_space_space(data,e_vec_num=6):
			space_space = np.cov(data.T)
			time_time = np.cov(data)
			tt_e_vals,tt_e_vecs = np.linalg.eig(time_time)
			for k in range(len(tt_e_vecs))[:e_vec_num]:
				time_time_e_holder = tt_e_vecs[:,k]
				e_val = tt_e_vals[k]
				space_space_e_holder = time_time_e_holder.reshape(1,time_time_e_holder.shape[0]).dot(data)
				remove_matrix = e_val*np.outer(space_space_e_holder,space_space_e_holder)/(space_space_e_holder**2).sum()
				space_space -= remove_matrix
			return space_space
		self.cov = subtract_e_vecs_return_space_space(self.data)

		holder = self.calculate_scaling(lat_sep,lon_sep,l)
		cov_scale = 0.7 
		self.scaling = np.vstack([
			np.hstack([holder,cov_scale*holder,cov_scale*holder,cov_scale*holder]),
			np.hstack([cov_scale*holder,holder,cov_scale*holder,cov_scale*holder]),
			np.hstack([cov_scale*holder,cov_scale*holder,holder,cov_scale*holder]),
			np.hstack([cov_scale*holder,cov_scale*holder,cov_scale*holder,holder])
			])
		del holder
		self.cov = csc_matrix(self.cov*self.scaling)
		# self.scaling = matrix_element(scaling,self.lats,self.lons,self.lat_grid,self.lon_grid,self.file_name+'_scaling_l_'+str(l))


	def save(self):
		from transition_matrix.compute.trans_read import BaseMat
		mat_obj = InverseInstance(self.cov,self.cov.shape,list(zip(self.lats,self.lons)),self.lat_sep,self.lon_sep,'covariance',self.l)
		mat_obj.save()

	def diagnostic_data_plots(self,first_e_vals=10):
		base = ROOT_DIR+'/plots/cm2p6_covariance/'

		time_time = np.cov(self.data)
		tt = matrix_element(time_time,self.lats,self.lons,self.lat_grid,self.lon_grid,self.file_name+'_time_time_cov')
		tt.eig_val_plot()
		space_space = np.cov(self.data.T)
		tt_e_vals,tt_e_vecs = np.linalg.eig(time_time)
		percent_constrained = [str(round(_,2))+'%' for _ in tt_e_vals/tt_e_vals.sum()*100]

		lons = [find_nearest(self.lon_grid,x) for x in self.lons] 
		lats = [find_nearest(self.lat_grid,x) for x in self.lats]
		index_list = [list(x) for x in list(zip(lats,lons))]
		for k in range(first_e_vals):
			e_val = tt_e_vals[k]
			e_holder = tt_e_vecs[:,k]
			percent = percent_constrained[k]
			e_vecs = np.split(e_holder.reshape(1,e_holder.shape[0]).dot(self.data),4,axis=1)
			for var,e_vec in zip(self.variable_list,e_vecs):
				plot_vec = transition_vector_to_plottable(self.lat_grid.tolist(),self.lon_grid.tolist(),index_list,e_vec[0].tolist())
				plt.subplot(2,1,1)
				XX,YY,m = basemap_setup(self.lat_grid.tolist(),self.lon_grid.tolist(),'')
				m.pcolormesh(XX,YY,plot_vec)
				plt.title(var+' e_vec_'+str(k)+', percent constrained = '+percent)
				plt.subplot(2,1,2)
				plt.plot(self.time,e_holder)
				plt.savefig(base+var+'_e_vec_'+str(k))
				plt.close()

	def generate_cov(self):
		self.cov_dict = {}
		for col_var in self.variable_list:
			variable_dict = {}
			for row_var in self.variable_list:
				cov_holder = self.get_cov(row_var,col_var,cov,break_unit)
				variable_dict[row_var] = cov_element(cov_holder,self.lats,self.lons,self.lat_grid,self.lon_grid,row_var,col_var,self.file_name+'_'+str(row_var)+'_'+str(col_var))
			self.cov_dict[col_var] = variable_dict

	def get_cov(self,row_var,col_var):
		row_idx = self.variable_list.index(row_var)
		col_idx = self.variable_list.index(col_var)
		split_array = np.split(np.split(cov,4)[row_idx],4,axis=1)[col_idx]
		return split_array

	@staticmethod
	def calculate_scaling(lat_sep,lon_sep,l=300):
		def get_dist(lat_sep,lon_sep):
			def calculate_distance():
				import geopy.distance
				lats,lons,truth_list,lat_grid,lon_grid = cov_array.lats_lons(lat_sep,lon_sep)
				dist = np.zeros([len(lats),len(lons)])
				coords = list(zip(lats,lons))
				for ii,coord1 in enumerate(list(coords)):
					print(ii)
					for jj,coord2 in enumerate(coords):
						dist[ii,jj] = geopy.distance.great_circle(coord1,coord2).km 
				assert (dist>=0).all()&(dist<=40000).all()
				return dist

			filename = cov_array.data_directory+'distance_lat_'+str(lat_sep)+'_lon_'+str(lon_sep)
			try:
				dist = np.load(filename+'.npy')
			except IOError:
				dist = calculate_distance(lat_sep,lon_sep)
				np.save(filename,dist)
			return dist

		def check_symmetric(a, rtol=1e-05, atol=1e-08):
		    return np.allclose(a, a.T, rtol=rtol, atol=atol)
		dist = get_dist(lat_sep,lon_sep)

		c = np.sqrt(10/3.)*l
#For this scaling we use something derived by gassbury and coehn to not significantly change eigen spectrum of 
#local support scaling function
#last peice wise poly 		
		scaling = np.zeros(dist.shape)
		# dist[dist>2*c]=0
		second_poly_mask = (dist>c)&(dist<2*c)
		dist_holder = dist[second_poly_mask].flatten()
		assert (dist_holder.min()>c)&(dist_holder.max()<2*c)
		second_poly = 1/12.*(dist_holder/c)**5 \
		-1/2.*(dist_holder/c)**4 \
		+5/8.*(dist_holder/c)**3 \
		+5/3.*(dist_holder/c)**2 \
		-5.*(dist_holder/c) \
		+4 \
		- 2/3.*(c/dist_holder)
		
		scaling[second_poly_mask]=second_poly
		assert check_symmetric(scaling)

		first_poly_mask = (dist<c)
		dist_holder = dist[first_poly_mask].flatten()
		assert (dist_holder.min()>=0)&(dist_holder.max()<c)

		first_poly = -1/4.*(dist_holder/c)**5 \
		+1/2.*(dist_holder/c)**4 \
		+5/8.*(dist_holder/c)**3 \
		-5/3.*(dist_holder/c)**2 \
		+1
		scaling[first_poly_mask]=first_poly
		assert check_symmetric(scaling)
		return scaling

	@staticmethod
	def lats_lons(lat_sep,lon_sep):
		lat_grid = np.arange(-90,91,lat_sep)
		lats = np.load(cov_array.data_directory+'lat_list.npy')
		lat_translate = [find_nearest(lats,x) for x in lat_grid] 
		lat_truth_list = np.array([x in lat_translate for x in lats])

		lon_grid = np.arange(-180,180,lon_sep)
		lons = np.load(cov_array.data_directory+'lon_list.npy')
		lons[lons<-180]=lons[lons<-180]+360
		lon_translate = [find_nearest(lons,x) for x in lon_grid] 
		lon_truth_list = np.array([x in lon_translate for x in lons])

		truth_list = lat_truth_list&lon_truth_list
		lats = lats[truth_list]
		lons = lons[truth_list]
		return (lats,lons,truth_list,lat_grid,lon_grid)

