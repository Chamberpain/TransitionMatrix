from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list,full_argo_list
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from TransitionMatrix.Utilities.Compute.trans_read import TransMat,TransitionGeo
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from TransitionMatrix.Utilities.Plot.transition_matrix_plot import TransPlot
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
from TransitionMatrix.Utilities.Plot.argo_data import Core,BGC
from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo,InverseInstance,HInstance,InverseCCS,InverseGOM,InverseSOSE
import scipy
import numpy as np 

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')


def figure_6():
	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	high_res = TransPlot.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
	XX,YY,ax1 = high_res.trans_geo.plot_setup(ax=ax1)	
	standard_error = high_res.return_standard_error()
	standard_error_plot = high_res.trans_geo.transition_vector_to_plottable(standard_error)
	standard_error_plot = np.ma.masked_greater(standard_error_plot,100)
	number_matrix = high_res.new_sparse_matrix(high_res.number_data)
	k = number_matrix.sum(axis=0)
	k = k.T
	standard_error_plot = np.ma.array(standard_error_plot,mask=high_res.trans_geo.transition_vector_to_plottable(k)==0)
	ax1.pcolormesh(XX,YY,standard_error_plot*100,cmap=plt.cm.cividis,vmax=high_res.trans_geo.std_vmax)

	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
	high_res = TransPlot.load_from_type(GeoClass=TransitionGeo,lat_spacing = 4,lon_spacing = 4,time_step = 90)
	XX,YY,ax2 = high_res.trans_geo.plot_setup(ax=ax2)	
	standard_error = high_res.return_standard_error()
	standard_error_plot = high_res.trans_geo.transition_vector_to_plottable(standard_error)
	standard_error_plot = np.ma.masked_greater(standard_error_plot,100)
	number_matrix = high_res.new_sparse_matrix(high_res.number_data)
	k = number_matrix.sum(axis=0)
	k = k.T
	standard_error_plot = np.ma.array(standard_error_plot,mask=high_res.trans_geo.transition_vector_to_plottable(k)==0)
	ax2.pcolormesh(XX,YY,standard_error_plot*100,cmap=plt.cm.cividis,vmax=high_res.trans_geo.std_vmax)
	PCM = ax2.get_children()[0]
	fig.colorbar(PCM,ax=[ax1,ax2],fraction=0.10,label='Mean Standard Error (%)')
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(file_handler.out_file('standard_error_resolution_compare'))
	plt.close()



def figure_sample_by_sensor_in_year():
	full_argo_list()
	inverse_mat = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=2)
	trans_mat = TransMat.load_from_type(lat_spacing=3,lon_spacing=3,time_step=90)
	trans_mat.trans_geo.variable_list = inverse_mat.trans_geo.variable_list
	trans_mat.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	float_mat = BGC.recent_floats(trans_mat.trans_geo, BGCReader)
	total_obs_all = np.sum([trans_mat.multiply(x) for x in range(4)])
	total_obs_min = [trans_mat.multiply(x) for x in range(4)]
	for k,var in enumerate(['so','o2','chl','ph']):
		fig = plt.figure(figsize=(14,14))
		ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())

		obs_out = scipy.sparse.csc_matrix(total_obs_all).dot(float_mat.get_sensor(var))
		plottable = total_obs.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
		print(var)
		print(np.mean(plottable))
		plottable = np.ma.masked_less(plottable,1)
		XX,YY,ax = total_obs.trans_geo.plot_setup(ax=ax1)
		ax.pcolor(XX,YY,plottable,vmin=0,vmax=100)
		ax.annotate('a', xy = (0.17,0.8),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

		ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())

		obs_out = [scipy.sparse.csc_matrix(x).dot(float_mat.get_sensor(var)) for x in total_obs_min]
		first_min = np.minimum(obs_out[0].todense(),obs_out[1].todense())
		second_min = np.minimum(obs_out[2].todense(),obs_out[3].todense())
		minimum = np.minimum(first_min,second_min)		
		plottable = trans_mat.trans_geo.transition_vector_to_plottable(minimum)*100
		print(var)
		print(np.mean(plottable))
		plottable = np.ma.masked_less(plottable,1)
		XX,YY,ax = total_obs.trans_geo.plot_setup(ax=ax2)
		ax.pcolor(XX,YY,plottable,vmin=0,vmax=100)
		ax.annotate('b', xy = (0.17,0.8),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		PCM = ax.get_children()[0]
		plt.subplots_adjust(left=0.06)
		fig.colorbar(PCM,ax=[ax1,ax2],label='Observation in Next Year (%)',fraction=0.10,)
		plt.savefig(file_handler.out_file('chance_of_observation_'+var))
		plt.close()

def figure_sample_in_year():
	full_argo_list()
	inverse_mat = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=2)
	trans_mat = TransMat.load_from_type(lat_spacing=3,lon_spacing=3,time_step=90)
	trans_mat.trans_geo.variable_list = inverse_mat.trans_geo.variable_list
	trans_mat.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	lats = trans_mat.trans_geo.get_lat_bins()
	lons = trans_mat.trans_geo.get_lon_bins()
	float_mat = BGC.recent_floats(trans_mat.trans_geo, BGCReader)

	plot_dict = {'so':'a','ph':'d','chl':'c','o2':'b'}
	fig = plt.figure(figsize=(14,14))
	
	ax_list = [fig.add_subplot(4,1,(k+1), projection=ccrs.PlateCarree()) for k in range(4)]
	for k,var in enumerate(['so','o2','chl','ph']):
		obs_out = [scipy.sparse.csc_matrix(x).dot(float_mat.get_sensor(var)) for x in total_obs]
		first_min = np.minimum(obs_out[0].todense(),obs_out[1].todense())
		second_min = np.minimum(obs_out[2].todense(),obs_out[3].todense())
		minimum = np.minimum(first_min,second_min)		
		plottable = trans_mat.trans_geo.transition_vector_to_plottable(minimum)*100
		print(var)
		print(np.mean(plottable))
		plottable = np.ma.masked_less(plottable,1)
		ax = ax_list[k]
		XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)
		ax.pcolor(XX,YY,plottable,vmin=0,vmax=100)
		ax.annotate(plot_dict[var], xy = (0.17,0.8),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	PCM = ax.get_children()[0]
	plt.subplots_adjust(left=0.06)
	fig.colorbar(PCM,ax=ax_list,label='Year Round Observations (%)',fraction=0.10,)
	plt.savefig(file_handler.out_file('chance_of_yearround'))
	plt.close()

def figure_random_float_array():
	trans_mat = TransMat.load_from_type(lat_spacing=3,lon_spacing=3,time_step=90)
	lats = trans_mat.trans_geo.get_lat_bins()
	lons = trans_mat.trans_geo.get_lon_bins()
	total_obs = [trans_mat.multiply(x) for x in range(4)]
	out_list = []
	for size_of_array in np.arange(600,2401,100):
		plottable_list = []
		for dummy in range(500):
			print(dummy)
			row_idx = random.sample(range(trans_mat.shape[0]), size_of_array)
			data = [1]*len(row_idx)
			col_idx = [0]*len(row_idx)

			float_mat = scipy.sparse.csc_matrix((data,(row_idx,col_idx)), shape=(trans_mat.shape[0],1))		
			obs_out = [scipy.sparse.csc_matrix(x).dot(float_mat) for x in total_obs]
			first_min = np.minimum(obs_out[0].todense(),obs_out[1].todense())
			second_min = np.minimum(obs_out[2].todense(),obs_out[3].todense())
			minimum = np.minimum(first_min,second_min)		
			plottable_list.append(minimum)
		out_list.append((size_of_array,np.mean(plottable_list),np.std(plottable_list)))
	number_of_floats,mean,std = zip(*out_list)
	mean = np.array(mean)*100
	std = np.array(std)*100
	fig = plt.figure(figsize=(14,14))
	plt.plot(number_of_floats,mean)
	plt.fill_between(number_of_floats,mean-std,mean+std,alpha=0.1)
	plt.xlabel('Number of Floats in Array')
	plt.ylabel('Grid Seasonally Sampled (%)')
	plt.xlim([600,2400])
	plt.ylim([0,100])


def future_agg_plots():
	trans_mat = TransMat.load_from_type(lat_spacing=2,lon_spacing=2,time_step=30)
	full_argo_list()
	inverse_mat = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=2)
	trans_mat.trans_geo.variable_list = inverse_mat.trans_geo.variable_list
	trans_mat.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	lats = trans_mat.trans_geo.get_lat_bins()
	lons = trans_mat.trans_geo.get_lon_bins()
	float_mat = Core.recent_floats(trans_mat.trans_geo, ArgoReader)
	plot_results = [float_mat.todense(),trans_mat.todense().dot(float_mat.todense()),trans_mat.multiply(5).todense().dot(float_mat.todense())]
	for pc,area in zip([GlobalCartopy,NAtlanticCartopy,NPacificCartopy,SAtlanticCartopy,SPacificCartopy],['global','Natlantic','Npacific','Satlantic','Spacific']):
		trans_mat.trans_geo.plot_class = pc
		fig = plt.figure(figsize=(14,14))
		if 'pacific' in area:
			ax_list = [fig.add_subplot(3,1,(k+1), projection=ccrs.PlateCarree(central_longitude=180)) for k in range(3)]
		else:
			ax_list = [fig.add_subplot(3,1,(k+1), projection=ccrs.PlateCarree()) for k in range(3)]
		for k,(var,out_plot) in enumerate(zip(['a','b','c'],plot_results)):

			plottable = trans_mat.trans_geo.transition_vector_to_plottable(out_plot)
			plottable = np.ma.masked_equal(plottable,0)*100
			XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax_list[k])
			if 'pacific' in area:
				XX = XX+180
			ax.pcolormesh(XX,YY,plottable,vmin=0,vmax=100)
			ax.annotate(var, xy = (0.17,0.8),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		PCM = ax.get_children()[0]
		fig.colorbar(PCM,ax=ax_list,label='Chance of Float (%)',fraction=0.10,)
		plt.savefig(file_handler.out_file(area+'_future_array'))
		plt.close()




def five_year_density_plot():
	holder = argos_class = TransMat.load_from_type(GeoClass=GPSGeo,lat_spacing = 2,lon_spacing = 3,time_step = 90)
	t = 90
	for k in range(8):
		print(k)
		t = 2*t
		holder = holder.dot(holder)
	trans_mat = holder
	plottable = scipy.sparse.csc_matrix(trans_mat).dot(scipy.sparse.csc_matrix(np.ones([trans_mat.shape[0],1]))) 
	plottable = trans_mat.trans_geo.transition_vector_to_plottable(plottable.todense())
	plottable = np.ma.masked_array(plottable,mask=abs(plottable)<10**-6)

	XX,YY,ax = trans_mat.trans_geo.plot_setup()	
	ax.pcolormesh(XX,YY,plottable*100,vmax=175,vmin=0)
	PCM = ax.get_children()[0]
	plt.colorbar(PCM,ax=ax,label='Relative Chance of Aggregation (%)')
	plt.savefig(file_handler.out_file('future_aggreation_gps'))
	plt.close()

def eig_all_plots(trans_mat):
	def plot_the_data(plottable):
		XX,YY = np.meshgrid(bins_lon,bins_lat)
		XX,YY,m = basemap_setup(bins_lat,bins_lon,trans_mat.traj_file_type)
		plottable = np.ma.masked_array(plottable,mask=abs(plottable)<10**-(2.5))
		m.pcolormesh(XX,YY,plottable,vmin=-10**-(1.5),vmax=10**-(1.5))
		plt.colorbar()
		# if not real_mask.all():
		#     m,e_w,n_s = TransPlot(matrix,total_list=trans_mat.total_list,lat_spacing=trans_mat.degree_bins[0]
		#         ,lon_spacing=trans_mat.degree_bins[1],traj_file_type='Argo').quiver_plot(m=m,ellipses=False,mask=mask)


	trans_mat = TransPlot.load_from_type(1,1,30)
	# transmat = transmat.remove_small_values(0.04)
	# transmat.rescale()
	plot_name = str(int(trans_mat.degree_bins[0]))+'_'+str(int(trans_mat.degree_bins[1]))+'_'+str(trans_mat.traj_file_type)
	r_e_vals,r_e_vecs = scipy.sparse.linalg.eigs(trans_mat.T,k=60) 

	r_idx = [x for _,x in sorted(zip([abs(k) for k in r_e_vals],range(len(r_e_vals))))]
	r_idx = r_idx[::-1]
	# l_idx = [x for _,x in sorted(zip([abs(k) for k in l_e_vals],range(len(l_e_vals))))]
	# l_idx = l_idx[::-1]
	mag_mask = 10**-3
	vmin_mag = 3.
	plt.plot(r_e_vals[r_idx])
	plt.savefig(plot_name+'_e_val_spectrum_r')
	plt.close()

	bins_lat,bins_lon = trans_mat.bins_generator(trans_mat.degree_bins)
	# bins_lat,bins_lon = trans_mat.bins_generator(trans_mat.degree_bins)
	for n in range(len(r_idx)):

		r_e_vec = r_e_vecs[:,r_idx[n]]
		r_e_val = r_e_vals[r_idx[n]]

		# matrix_component = r_e_val*np.outer(r_e_vec,r_e_vec)/(r_e_vec**2).sum()
		# matrix_component = matrix_component/abs(matrix_component).max()
		
		# eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
		# eig_vec_token_holder = np.ma.masked_inside(eig_vec_token,-mag_mask,mag_mask)
		# mag = eig_vec_token_holder.data.max()
		# real_mask = eig_vec_token_holder.mask
		plt.subplot(1,2,1)
		eig_vec_token = transition_vector_to_plottable(bins_lat,bins_lon,trans_mat.total_list,r_e_vec.real)
		plot_the_data(eig_vec_token)
		plt.title('real right evec')
		plt.subplot(1,2,2)
		eig_vec_token = transition_vector_to_plottable(bins_lat,bins_lon,trans_mat.total_list,r_e_vec.imag)
		plot_the_data(eig_vec_token)
		plt.title('imag right evec')

		# eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
		# eig_vec_token_holder = np.ma.masked_inside(eig_vec_token,-mag_mask,mag_mask)
		# mag = eig_vec_token_holder.data.max()
		# imag_mask = eig_vec_token_holder.mask
		plt.suptitle(str(r_e_val))
		plt.savefig(plot_name+'transition_'+str(n))
		plt.close()

	def eig_total_plot(self,label=''):
		eig_vals,eig_vecs = scipy.sparse.linalg.eigs(self,k=30)


		overlap_colors = [(0, 1, 0),(1, 1, 0),(0, 1, 1),(1, 0, 0),(0, 0, 1),(1, 0, 1),(0.2, 1, 0.5),(0.5, 1, 0),(0, 1, 0.5),(0.5, 0.5, 0.1),
		(0, 1, 0),(1, 1, 0),(0, 1, 1),(1, 0, 0),(0, 0, 1),(1, 0, 1),(0.2, 1, 0.5),(0.5, 1, 0),(0, 1, 0.5),(0.5, 0.5, 0.1),
		(0, 1, 0),(1, 1, 0),(0, 1, 1),(1, 0, 0),(0, 0, 1),(1, 0, 1),(0.2, 1, 0.5),(0.5, 1, 0),(0, 1, 0.5),(0.5, 0.5, 0.1),
		]
		cmap_name = 'my_list'
		n_bin = 2
		cmaps=[colors.LinearSegmentedColormap.from_list(cmap_name, [_,_] , N=n_bin) for _ in overlap_colors]
		XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type) 
		for k,eig_val in enumerate(eig_vals):
			eig_vec = eig_vecs[:,k]
			eig_vec_token = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,eig_vec)
			eig_vec_token = abs(eig_vec_token)
			eig_vec_token = np.ma.masked_less(eig_vec_token,eig_vec_token.mean()+1.1*eig_vec_token.std()) 
			m.pcolormesh(XX,YY,eig_vec_token,cmap=cmaps[k],alpha=0.7)
		plt.show()


		for k,eig_val in enumerate(eig_vals):
			print('plotting eigenvector '+str(k)+' for '+str(self.degree_bins)+' eigenvalue is '+str(eig_val))
			eig_vec = eig_vecs[:,k]
			assert (self.todense().dot(eig_vec.T)-eig_val*eig_vec).max()<10**-1


			plt.figure(figsize=(10,10))
			plt.subplot(3,1,1)
			self.eig_vec_plot(eig_vec.real)
			plt.title('left eig vec (real)')
			plt.subplot(3,1,2)
			self.eig_vec_plot(np.absolute(eig_vec))
			plt.title('left eig vec (absolute)')
			plt.subplot(3,1,3)
			plt.plot(np.abs(eig_vals)[::-1])
			plt.ylabel('Eigen Value')
			plt.xlabel('Eign Number')
			plt.ylim([0,1])
			plt.xlim([0,len(eig_vals)])
			plt.title('Eigen Value Spectrum')
			plt.suptitle('Eigen Value '+'{0:.4f} + {1:.4f}i'.format(float(eig_val.real),float(eig_val.imag)))
			plt.savefig('../plots/eig_vals_'+str(k)+'_time_step_'+str(self.time_step)+'.png')
			plt.close()



def number_matrix_plot():
	import sys
	from PIL import Image
	for traj in ['argo','SOSE']:
		holder = TransPlot.load_from_type(2,2,60,traj_type=traj) 
		holder.number_plot()
		holder.standard_error_plot()
		images = [Image.open(x) for x in [holder.std_error_file(),holder.number_file()]]
		widths, heights = zip(*(i.size for i in images))
		total_width = sum(widths)
		max_height = max(heights)

		new_im = Image.new('RGB', (total_width, max_height))

		x_offset = 0
		for im in images:
		  new_im.paste(im, (x_offset,0))
		  x_offset += im.size[0]
		new_im.save('/Users/pchamberlain/Projects/transition_matrix_paper/plots/'+traj+'number_plot.jpg')


def eigen_spectrum_plot():
	for lat,lon in [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]:
		for time in [30,60,90,120,140,160,180]:
			holder = TransPlot.load_from_type(lon,lat,time)
			# trans_mat = holder.multiply(multiplyer,0.0000001)
			e_vals,e_vecs =scipy.sparse.linalg.eigs(holder,k=40)
			plt.plot(np.sort(e_vals)[::-1],label=str(time)+' days')
		plt.legend()
		plt.savefig(holder.plot_folder()+'eigen_time_plot')
		plt.close()

def standard_error_plot():
	for traj_type in ['argo','SOSE']:
		error_list = []
		for lat,lon in [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]:
			for time in [30,60,90,120,140,160,180]:
				print(time)
				holder_low_res = TransPlot.load_from_type(lon,lat,time,traj_type=traj_type)
				std_lr = holder_low_res.return_standard_error()
				error_list.append((lat,lon,time,np.nanmean(std_lr),np.nanstd(std_lr)))
		start_idx = (np.array(range(6)))*7
		end_idx = (np.array(range(6))+1)*7
		fig,ax = plt.subplots()
		for ii,kk in zip(start_idx,end_idx):
			holder = error_list[ii:kk]
			label = str((holder[0][0],holder[0][1]))
			dummy,dummy,time,mean,std = zip(*holder)
			mean = np.array(mean)
			print(mean)
			print(time)
			# std = np.array(std)
			# ax.fill_between(time,mean-std,mean+std,alpha=0.2)
			ax.plot(time,mean)
			ax.scatter(time,mean,label=label)
		plt.legend()
		plt.xlabel('Timestep (days)')
		plt.ylabel('Mean Error')
		plt.savefig('/Users/pchamberlain/Projects/transition_matrix_paper/plots/'+traj_type+'_error_compare.jpg')
		plt.close()

def resolution_bias_plot():
	from transition_matrix.compute.compute_utils import matrix_size_match    
	for traj_type in ['argo','SOSE']:
		error_list = []
		for lat,lon in [(2,2),(3,3),(4,4),(2,3),(3,6)]:
			for time in [30,60,90,120,140,160,180]:
				print(time)
				print(lat)
				print(lon)
				holder_low_res = TransPlot.load_from_type(lon,lat,time,traj_type=traj_type)
				holder_low_res = holder_low_res.multiply(2)        
				holder_high_res = TransPlot.load_from_type(1,1,time,traj_type=traj_type)
				holder_high_res = holder_high_res.multiply(2)                
				holder_high_res = holder_high_res.reduce_resolution((lat,lon))

				holder_low_res,holder_high_res = matrix_size_match(holder_low_res,holder_high_res)
				east_west_lr, north_south_lr = holder_low_res.return_mean()
				east_west_hr, north_south_hr = holder_high_res.return_mean()

				list_low_res = []
				list_high_res = []
				for i in range(holder_low_res.shape[0]):
					list_low_res.append((east_west_lr[:,i].data.mean(),north_south_lr[:,i].data.mean()))
					list_high_res.append((east_west_hr[:,i].data.mean(),north_south_hr[:,i].data.mean()))
				e_w_lr, n_s_lr = zip(*list_low_res)
				e_w_hr, n_s_hr = zip(*list_high_res)
				e_w_diff = np.array(e_w_lr)-np.array(e_w_hr)
				n_s_diff = np.array(n_s_lr)-np.array(n_s_hr)
				total_diff = np.sqrt(e_w_diff**2+n_s_diff**2)
				error_list.append((lat,lon,time,total_diff.mean(),total_diff.std()))
		start_idx = (np.array(range(5)))*7
		end_idx = (np.array(range(5))+1)*7
		fig,ax = plt.subplots()
		for ii,kk in zip(start_idx,end_idx):
			holder = error_list[ii:kk]
			label = str((holder[0][0],holder[0][1]))
			dummy,dummy,time,mean,std = zip(*holder)
			mean = np.array(mean)
			print(mean)
			print(time)
			# std = np.array(std)
			# ax.fill_between(time,mean-std,mean+std,alpha=0.2)
			ax.plot(time,mean)
			ax.scatter(time,mean,label=label)
		plt.legend()
		plt.xlabel('Timestep (days)')
		plt.ylabel('Mean Error')
		plt.savefig('/Users/pchamberlain/Projects/transition_matrix_paper/plots/'+traj_type+'_resolution_error_compare.jpg')
		plt.close()



def bgc_pdf():
	import matplotlib.pyplot as plt
	import numpy as np
	import matplotlib.colors	
	from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list,full_argo_list
	from TransitionMatrix.Utilities.Inversion.target_load import InverseInstance,InverseGeo
	from TransitionMatrix.Utilities.Plot.argo_data import BGC
	from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader

	
	norm = matplotlib.colors.Normalize(0,400)
	colors = [[norm(0), "yellow"],
	          [norm(10), "lightgoldenrodyellow"],
	          [norm(30), "lightyellow"],
	          [norm(50), "powderblue"],
			  [norm(75), "skyblue"],
	          [norm(100), "deepskyblue"],
	          [norm(400), "dodgerblue"],
	          ]
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
	full_argo_list()
	inverse_mat = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=2)
	trans_mat = TransMat.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)

	ew_data_list = []
	ns_data_list = []
	for k in range(10):
		holder = trans_mat.multiply(k)
		east_west, north_south = holder.return_mean()
		ew_data_list.append(east_west)
		ns_data_list.append(north_south)
	ew = np.vstack(ew_data_list)
	ns = np.vstack(ns_data_list)

	trans_mat.trans_geo.variable_list = inverse_mat.trans_geo.variable_list
	trans_mat.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	lats = trans_mat.trans_geo.get_lat_bins()
	lons = trans_mat.trans_geo.get_lon_bins()
	float_mat = BGC.recent_floats(trans_mat.trans_geo, BGCReader)
	total_obs = [trans_mat.multiply(x) for x in range(10)]

	plot_dict = {'so':'a','ph':'d','chl':'c','o2':'b'}

	for i,trans in enumerate(total_obs):
		for k,var in enumerate(['so','o2','chl','ph']):
			fig = plt.figure(figsize=(21,14))
			ax = fig.add_subplot(projection=ccrs.PlateCarree())
			obs_out = scipy.sparse.csc_matrix(trans).dot(float_mat.get_sensor(var))
			plottable = trans_mat.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
			plottable[plottable<0.1]=0.1
			plottable = np.ma.masked_less(plottable,1)
			ax.annotate(var, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
			XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)
			pcm = ax.pcolor(XX,YY,plottable,vmin=0.1,vmax=400,cmap='Greys',norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=400))
			for idx in scipy.sparse.find(float_mat.get_sensor(var))[0]:
				point = list(trans_mat.trans_geo.total_list)[idx]
				ew_holder = ew[:i,idx]
				ns_holder = ns[:i,idx]
				lons = [point.longitude + x for x in ew_holder]
				lats = [point.latitude + x for x in ns_holder]
				ax.plot(lons,lats,'r',linewidth=2)
			fig.colorbar(pcm,orientation="horizontal",shrink=0.9,label='Chance of Observation in Cell')
			plt.title(str(90*i)+' Day Timestep')
			plt.savefig(file_handler.tmp_file(var+'/timestep_'+str(i)))
			plt.close()
	for k,var in enumerate(['so','o2','chl','ph']):
		os.chdir(file_handler.tmp_file(var+'/'))
		os.system("ffmpeg -r 11/12 -i timestep_%01d.png -vcodec mpeg4 -y movie.mp4")







plot_style_dict = {'argo':'--','SOSE':':'}
plot_color_dict = {(1,1):'teal',(1,2):'brown',(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}

def temporal_bias_plot():
	from TransitionMatrix.Utilities.Compute.compute_utils import matrix_size_match    
	out = []
	for traj_type in [TransitionGeo,SOSEGeo]:
		for lat,lon in [(1,1),(1,2),(2,2),(3,3),(4,4),(2,3),(4,6)]:
			for time, multiplyer in [(30,11),(60,5),(90,3),(120,2)]:
				print('lat is ',lat,' lon is ',lon)
				print('time is ',time)
				holder_low_res = TransMat.load_from_type(GeoClass=traj_type,lat_spacing=lat,lon_spacing=lon,time_step=time)
				holder_low_res = holder_low_res.multiply(multiplyer)
				holder_high_res = TransMat.load_from_type(GeoClass=traj_type,lat_spacing=lat,lon_spacing=lon,time_step=180)
				holder_high_res = holder_high_res.multiply(1)
				out.append(matrix_compare(holder_low_res,holder_high_res,traj_type.file_type))
	with open(file_handler.tmp_file('resolution_difference_data'), 'wb') as fp:
		pickle.dump(out, fp)
	fp.close()

def resolution_bias_plot():
	data_list = []
	for time in [30,60,90,120,150,180]:	
		high_res = TransMat.load_from_type(lat_spacing=1,lon_spacing=1,time_step=time)
		hr_ew_scaled,hr_ns_scaled = high_res.return_mean()

		for lat,lon in [(1,2),(2,2),(3,3),(4,4),(2,3),(4,6)]:
			print('lat = ',lat)
			print('lon = ',lon)
			print('time = ',time)
			low_res = TransMat.load_from_type(lat_spacing=lat,lon_spacing=lon,time_step=time)
			lr_ew_scaled,lr_ns_scaled = low_res.return_mean()
			for lr_idx in range(low_res.shape[0]):
				print('idx = ',lr_idx)
				point_list = low_res.trans_geo.total_list.reduced_res(lr_idx,1,1)
				mean_list = []
				for x in point_list:
					try:
						hr_idx = high_res.trans_geo.total_list.index(geopy.Point(x))
						mean_list.append(geopy.Point(x[0]+hr_ns_scaled[hr_idx],x[1]+hr_ew_scaled[hr_idx]))
					except ValueError:
						continue
				if not mean_list:
					continue
				lat_list,lon_list = zip(*[(x.latitude,x.longitude) for x in mean_list])
				high_res_mean = geopy.Point(np.mean(lat_list),np.mean(lon_list))
				low_res_lat = low_res.trans_geo.total_list[lr_idx].latitude+lr_ns_scaled[lr_idx]
				low_res_lon = low_res.trans_geo.total_list[lr_idx].longitude+lr_ew_scaled[lr_idx]
				low_res_mean = geopy.Point(low_res_lat,low_res_lon)
				error = geopy.distance.great_circle(high_res_mean,low_res_mean).km
				data_list.append((error,lat,lon,time))
	with open(file_handler.tmp_file('resolution_bias_data'), 'wb') as fp:
		pickle.dump(data_list, fp)
	fp.close()

def matrix_resolution_decomp_1():
	loc = geopy.Point(-54,-54)
	lat = 2
	lon = 2
	short_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = 30)
	short_mat = short_mat.multiply(5)
	long_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = 180)
	fig = plt.figure(figsize=(12,12))
	ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
	short_mat.distribution_and_mean_of_column(loc,ax=ax1,pad=25)
	ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
	long_mat.distribution_and_mean_of_column(loc,ax=ax2,pad=25)
	ax1.annotate('a', xy = (0.9,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.9,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)

	from GeneralUtilities.Plot.Cartopy.eulerian_plot import PointCartopy
	loc = geopy.Point(-54,-40)
	small_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 180)
	big_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 4,lon_spacing = 4,time_step = 180)
	idx = big_mat.trans_geo.total_list.index(loc)
	reduced_loc_list = big_mat.trans_geo.total_list.reduced_res(idx,2,2)

	ax3 = fig.add_subplot(2,2,3, projection=ccrs.PlateCarree())
	XX,YY,ax3 = PointCartopy(loc,lat_grid = small_mat.trans_geo.get_lat_bins(),
		lon_grid = small_mat.trans_geo.get_lon_bins(),pad=25,ax=ax3).get_map()

	pdf_list = []
	mean_list = []
	for x in reduced_loc_list:
		small_mean = small_mat.mean_of_column(geopy.Point(x))
		mean_list.append(small_mean)
		pdf_list.append(np.array(small_mat[:,small_mat.trans_geo.total_list.index(geopy.Point(x))].todense()).flatten())
	pdf = small_mat.trans_geo.transition_vector_to_plottable(sum(pdf_list)/4)
	ax3.pcolormesh(XX,YY,pdf,cmap='Blues')
	x_mean,y_mean = zip(*[(x.longitude,x.latitude) for x in mean_list])
	ax3.scatter(x_mean,y_mean,c='black',linewidths=5,marker='x',s=80,zorder=10)
	ax3.scatter(np.mean(x_mean),np.mean(y_mean),c='pink',linewidths=8,marker='x',s=120,zorder=10)
	start_lat,start_lon = zip(*reduced_loc_list)
	ax3.scatter(start_lon,start_lat,c='red',linewidths=5,marker='x',s=80,zorder=10)


	ax4 = fig.add_subplot(2,2,4, projection=ccrs.PlateCarree())
	big_mat.distribution_and_mean_of_column(loc,ax=ax4,pad=25)
	ax3.annotate('c', xy = (0.9,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax4.annotate('d', xy = (0.9,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(plot_handler.out_file('resolution_bias_example'))
	plt.close()




def seasonal_spatial_plot():
	lat = 2
	lon = 3 
	date = 90
	summer_class = TransMat.load_from_type(GeoClass=SummerGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
	winter_class = TransMat.load_from_type(GeoClass=WinterGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
	(ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff) = matrix_compare(summer_class,winter_class)
	fig = plt.figure(figsize=(10,5))
	ax1 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
	XX,YY,ax1 = summer_class.trans_geo.plot_setup(ax = ax1)

	q = ax1.quiver(XX,YY,u=ew_mean_diff,v=ns_mean_diff,scale=100)
	ax1.quiverkey(q, X=0.3, Y=1.1, U=5,
             label='5 Degree', labelpos='E')
	ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
	XX,YY,ax2 = summer_class.trans_geo.plot_setup(ax = ax2)
	ax2.pcolormesh(XX,YY,ns_std_diff,vmin=-0.15,vmax=0.15,cmap='bwr')
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	PCM = ax2.get_children()[0]
	plt.colorbar(PCM,ax=ax2)
	plt.savefig(plot_handler.out_file('summer_winter_compare'))
	plt.close()

def resolution_bias_plot():
	plt.rcParams['font.size'] = '16'
	plot_color_dict = {(1,1):'teal',(1,2):'brown',(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}
	with open(data_handler.tmp_file('resolution_difference_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,lat,lon,time,pos_type = zip(*datalist)
	fig = plt.figure(figsize=(14,12))
	ax1 = fig.add_subplot(2,2,1)
	ax2 = fig.add_subplot(2,2,2)
	for system in ['argo']:
		for grid in list(plot_color_dict.keys()):

			lat_mask = np.array([grid[0]==x for x in lat])
			lon_mask = np.array([grid[1]==x for x in lon])
			system_mask = np.array([system==x for x in pos_type])
			mask = lat_mask&lon_mask&system_mask
			ew_mean_diff_holder = np.array(ew_mean_diff)[mask]
			ns_mean_diff_holder = np.array(ns_mean_diff)[mask]
			ew_std_diff_holder = np.array(ew_std_diff)[mask]
			ns_std_diff_holder = np.array(ns_std_diff)[mask]

			out = np.sqrt(ew_mean_diff_holder**2+ns_mean_diff_holder**2)
			print('I am plotting ',system)
			ax1.plot(np.unique(time),out,color=plot_color_dict[grid],label=str(grid))
			ax1.scatter(np.unique(time),out,color=plot_color_dict[grid])
			out = np.sqrt(ew_std_diff_holder**2+ns_std_diff_holder**2)
			ax2.plot(np.unique(time),out,color=plot_color_dict[grid],label=str(grid))
			ax2.scatter(np.unique(time),out,color=plot_color_dict[grid])
	ax1.set_xlim(28,122)
	ax2.set_xlim(28,122)
	ticks = [30,60,90,120]
	ax1.set_xticks(ticks)
	ax2.set_xticks(ticks)
	ax1.set_ylabel('Misfit')
	ax2.set_ylabel('Misfit')
	ax1.legend(loc='upper center', bbox_to_anchor=(1.2, 1.3),
          ncol=4, fancybox=True, shadow=True)
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)


	with open(data_handler.tmp_file('resolution_bias_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	error,lat,lon,time = zip(*datalist)
	ax3 = fig.add_subplot(2,2,3)
	time_axis = np.sort(np.unique(time))
	for lat_holder,lon_holder in plot_color_dict.keys():
		dim_mask = (np.array(lat)==lat_holder)&(np.array(lon)==lon_holder)
		if not dim_mask.tolist():
			continue
		plot_mean = []
		plot_std = []
		for time_holder in time_axis:
			time_mask = np.array(time)==time_holder
			mask = dim_mask&time_mask
			error_holder = np.array(error)[mask]
			plot_mean.append(error_holder.mean())
			plot_std.append(error_holder.std())
		ax3.plot(time_axis,plot_mean,label=str((lat_holder,lon_holder)),color=plot_color_dict[(lat_holder,lon_holder)])
		ax3.scatter(time_axis,plot_mean,c=plot_color_dict[(lat_holder,lon_holder)])
	ax4 = fig.add_subplot(2,2,4)
	with open(data_handler.tmp_file('resolution_standard_error'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	error,std,lat,lon,time = zip(*datalist)
	for lat_holder,lon_holder in plot_color_dict.keys():
		dim_mask = (np.array(lat)==lat_holder)&(np.array(lon)==lon_holder)
		if not dim_mask.tolist():
			continue
		plot_mean = []
		plot_std = []
		for time_holder in time_axis:
			time_mask = np.array(time)==time_holder
			mask = dim_mask&time_mask
			error_holder = np.array(error)[mask]
			std_holder = np.array(std)[mask]
			plot_mean.append(error_holder[0])
			plot_std.append(std_holder[0])
		ax4.plot(time_axis,plot_mean,label=str((lat_holder,lon_holder)),color=plot_color_dict[(lat_holder,lon_holder)])
		ax4.scatter(time_axis,plot_mean,c=plot_color_dict[(lat_holder,lon_holder)])
	ax4.set_xlabel('Timestep')
	ax3.set_xlabel('Timestep')

	ticks = [30,60,90,120,150,180]
	ax3.set_xticks(ticks)
	ax4.set_xticks(ticks)

	ax3.set_ylabel('Misfit (km)')
	ax4.set_ylabel('Error')
	ax3.annotate('c', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax4.annotate('d', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(plot_handler.out_file('resolution_bias_plot'))
	plt.close()


def argos_gps_stats_plot():
	with open(data_handler.tmp_file('argos_gps_withholding'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	plot_style_dict = {'gps':'--','argos':':'}
	ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,lat,lon,time,pos_type = zip(*datalist)
	fig = plt.figure(figsize=(9,8))
	plot_color_dict = {(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}
	ax1 = fig.add_subplot(2,1,1)
	ax2 = fig.add_subplot(2,1,2)
	for k,pos in enumerate(np.unique(pos_type)):
		for grid in list(plot_color_dict.keys()):
			lat_mask = np.array([grid[0]==x for x in lat])
			lon_mask = np.array([grid[1]==x for x in lon])
			pos_mask = np.array([pos==x for x in pos_type])
			mask = lat_mask&lon_mask&pos_mask
			ew_mean_diff_holder = np.array(ew_mean_diff)[mask]
			ns_mean_diff_holder = np.array(ns_mean_diff)[mask]
			ew_std_diff_holder = np.array(ew_std_diff)[mask]
			ns_std_diff_holder = np.array(ns_std_diff)[mask]

			out = np.sqrt(ew_mean_diff_holder**2+ns_mean_diff_holder**2)
			ax1.plot(np.unique(time),out,linestyle=plot_style_dict[pos],color=plot_color_dict[grid],label=str(grid))
			ax1.scatter(np.unique(time),out,color=plot_color_dict[grid])
			out = np.sqrt(ew_std_diff_holder**2+ns_std_diff_holder**2)
			ax2.plot(np.unique(time),out,linestyle=plot_style_dict[pos],color=plot_color_dict[grid],label=str(grid))
			ax2.scatter(np.unique(time),out,color=plot_color_dict[grid])
		if k ==0:
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
      		ncol=3, fancybox=True, shadow=True)

	ax1.set_xlim(28,182)
	ax2.set_xlim(28,182)
	ticks = [30,60,90,120,180]
	ax1.set_xticks(ticks)
	ax2.set_xticks(ticks)
	ax2.set_xlabel('Timestep')
	ax1.set_ylabel('Mean Difference')
	ax2.set_ylabel('Mean Difference')

	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(plot_handler.out_file('argos_stats_plot'))
	plt.close()


def seasonal_plot():
	with open(data_handler.tmp_file('seasonal_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	plot_style_dict = {'summer':'--','winter':':'}
	plot_color_dict = {(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}
	ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,lat,lon,time,season = zip(*datalist)
	fig = plt.figure(figsize=(9,8))
	ax1 = fig.add_subplot(2,1,1)
	ax2 = fig.add_subplot(2,1,2)
	for k,pos in enumerate(np.unique(season)):
		for grid in list(plot_color_dict.keys()):
			lat_mask = np.array([grid[0]==x for x in lat])
			lon_mask = np.array([grid[1]==x for x in lon])
			season_mask = np.array([pos==x for x in season])
			mask = lat_mask&lon_mask&season_mask
			ew_mean_diff_holder = np.array(ew_mean_diff)[mask]
			ns_mean_diff_holder = np.array(ns_mean_diff)[mask]
			ew_std_diff_holder = np.array(ew_std_diff)[mask]
			ns_std_diff_holder = np.array(ns_std_diff)[mask]

			out = np.sqrt(ew_mean_diff_holder**2+ns_mean_diff_holder**2)
			ax1.plot(np.unique(time),out,linestyle=plot_style_dict[pos],color=plot_color_dict[grid],label=str(grid))
			ax1.scatter(np.unique(time),out,color=plot_color_dict[grid])
			out = np.sqrt(ew_std_diff_holder**2+ns_std_diff_holder**2)
			ax2.plot(np.unique(time),out,linestyle=plot_style_dict[pos],color=plot_color_dict[grid],label=str(grid))
			ax2.scatter(np.unique(time),out,color=plot_color_dict[grid])
		if k ==0:
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
      		ncol=3, fancybox=True, shadow=True)

	ax1.set_xlim(28,182)
	ax2.set_xlim(28,182)
	ticks = [30,60,90,120,180]
	ax1.set_xticks(ticks)
	ax2.set_xticks(ticks)
	ax2.set_xlabel('Timestep')
	ax1.set_ylabel('Mean Difference')
	ax2.set_ylabel('Mean Difference')

	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(plot_handler.out_file('seasonal_stats_plot'))
	plt.close()


def data_withholding_plot():
	plot_color_dict = {0.95:'teal',0.9:'red',0.85:'blue',0.8:'yellow',0.75:'orange',0.7:'green'}
	plt.rcParams['font.size'] = '16'
	with open(data_handler.tmp_file('transition_matrix_withholding_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,lat,lon,time,descriptor = zip(*datalist)
	pos_type,percentage = zip(*descriptor)

	for system in ['argo','SOSE']:
		fig = plt.figure(figsize=(9,8))
		ax1 = fig.add_subplot(2,1,1)
		ax2 = fig.add_subplot(2,1,2)
		for grid in [(2,2)]:
			for percent in np.unique(percentage):
				time_list = []
				mean_mean_list = []
				mean_std_list = []
				std_mean_list = []
				std_std_list = []
				for t in np.sort(np.unique(time)):
					time_list.append(t)
					percent_mask = np.array([percent==x for x in percentage])
					lat_mask = np.array([grid[0]==x for x in lat])
					lon_mask = np.array([grid[1]==x for x in lon])
					system_mask = np.array([system==x for x in pos_type])
					time_mask = np.array([t==x for x in time])
					mask = lat_mask&lon_mask&system_mask&percent_mask&time_mask

					ew_mean_diff_holder = np.array(ew_mean_diff)[mask]
					ns_mean_diff_holder = np.array(ns_mean_diff)[mask]

					out = np.sqrt(ew_mean_diff_holder**2+ns_mean_diff_holder**2)
					mean_mean_list.append(out.mean())
					mean_std_list.append(out.std())

					ew_std_diff_holder = np.array(ew_std_diff)[mask]
					ns_std_diff_holder = np.array(ns_std_diff)[mask]
					out = np.sqrt(ew_std_diff_holder**2+ns_std_diff_holder**2)

					std_mean_list.append(out.mean())
					std_std_list.append(out.std())


				ax1.errorbar(time_list,mean_mean_list,yerr=mean_std_list,linestyle=plot_style_dict[system],color=plot_color_dict[percent],label=percent)
				ax1.scatter(time_list,mean_mean_list,color=plot_color_dict[percent])

				ax2.errorbar(time_list,std_mean_list,yerr=std_std_list,linestyle=plot_style_dict[system],color=plot_color_dict[percent],label=percent)
				ax2.scatter(time_list,std_mean_list,color=plot_color_dict[percent])
		ax1.set_xlim(28,92)
		ax2.set_xlim(28,92)
		ticks = [30,60,90]
		ax1.set_xticks(ticks)
		ax2.set_xticks(ticks)
		ax2.set_xlabel('Timestep')
		ax1.set_ylabel('Mean Difference')
		ax2.set_ylabel('Mean Difference')
		ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
	          ncol=3, fancybox=True, shadow=True)
		ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
		ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
		plt.savefig(plot_handler.out_file(system+'_data_withholding_plot'))
		plt.close()




