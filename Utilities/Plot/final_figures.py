from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list
from TransitionMatrix.Utilities.Compute.trans_read import TransMat,TransitionGeo
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from TransitionMatrix.Utilities.Plot.transition_matrix_plot import TransPlot
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')


def figure_3():
	trans_mat = TransMat.load_from_type(lat_spacing=2,lon_spacing=2,time_step=30)
	aggregate_argo_list()
	lat_list,lon_list = ArgoReader.get_full_lat_lon_list()
	pos_list = ArgoReader.get_pos_list()
	argos_start_lat_list = [x[0] for x,y in zip(lat_list,pos_list) if y=='ARGOS']
	argos_start_lon_list = [x[0] for x,y in zip(lon_list,pos_list) if y=='ARGOS']

	gps_start_lat_list = [x[0] for x,y in zip(lat_list,pos_list) if y=='GPS']
	gps_start_lon_list = [x[0] for x,y in zip(lon_list,pos_list) if y=='GPS']
	print('length of GPS list is ',len(gps_start_lon_list))
	print('length of ARGOS list is ',len(argos_start_lon_list))
	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	XX,YY,ax1 = trans_mat.trans_geo.plot_setup(ax=ax1)
	ax1.scatter(argos_start_lon_list,argos_start_lat_list,s=0.5,c='r',label='ARGOS',zorder=11)
	ax1.scatter(gps_start_lon_list,gps_start_lat_list,s=0.5,c='b',label='GPS',zorder=11)
	ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2),
      		ncol=3, fancybox=True, shadow=True, markerscale=20)

	tp = TransPlot.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)
	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
	XX,YY,ax2 = tp.trans_geo.plot_setup(ax=ax2)
	number_matrix = tp.new_sparse_matrix(tp.number_data)
	k = number_matrix.sum(axis=0)
	k = k.T
	print(k)
	number_matrix_plot = tp.trans_geo.transition_vector_to_plottable(k)
	XX,YY,ax2 = tp.trans_geo.plot_setup(ax=ax2)  
	number_matrix_plot = np.ma.masked_equal(number_matrix_plot,0)   #this needs to be fixed in the plotting routine, because this is just showing the number of particles remaining
	ax2.pcolormesh(XX,YY,number_matrix_plot,cmap=plt.cm.magma,vmin=tp.trans_geo.number_vmin,vmax=tp.trans_geo.number_vmax)
	# plt.title('Transition Density',size=30)
	PCM = ax2.get_children()[0]
	fig.colorbar(PCM,pad=.15,label='Transition Number',orientation='horizontal',fraction=0.10)
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)

	plt.savefig(file_handler.out_file('initial_deployments'))
	plt.close()

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

def figure_7():
	trans_geo = TransPlot.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
	trans_geo.quiver_plot()

def figure_sample_by_sensor_in_year():
	full_argo_list()
	inverse_mat = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=2)
	trans_mat = TransMat.load_from_type(lat_spacing=3,lon_spacing=3,time_step=90)
	trans_mat.trans_geo.variable_list = inverse_mat.trans_geo.variable_list
	trans_mat.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	float_mat = BGC.recent_floats(trans_mat.trans_geo, BGCReader)
	total_obs = np.sum([trans_mat.multiply(x) for x in range(4)])
	plot_dict = {'so':'a','ph':'d','chl':'c','o2':'b'}
	fig = plt.figure(figsize=(14,14))
	
	ax_list = [fig.add_subplot(4,1,(k+1), projection=ccrs.PlateCarree()) for k in range(4)]
	for k,var in enumerate(['so','o2','chl','ph']):
		obs_out = scipy.sparse.csc_matrix(total_obs).dot(float_mat.get_sensor(var))
		plottable = total_obs.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
		print(var)
		print(np.mean(plottable))
		plottable = np.ma.masked_less(plottable,1)
		ax = ax_list[k]
		XX,YY,ax = total_obs.trans_geo.plot_setup(ax=ax)
		ax.pcolor(XX,YY,plottable,vmin=0,vmax=100)
		ax.annotate(plot_dict[var], xy = (0.17,0.8),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	PCM = ax.get_children()[0]
	plt.subplots_adjust(left=0.06)
	fig.colorbar(PCM,ax=ax_list,label='Observation in Next Year (%)',fraction=0.10,)
	plt.savefig(file_handler.out_file('chance_of_observation'))
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
	total_obs = [trans_mat.multiply(x) for x in range(4)]

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
