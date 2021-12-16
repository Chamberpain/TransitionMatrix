
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