def SOCCOM_death_plot():
	from TransitionMatrix.Utilities.Plot.argo_data import BGC as SOCCOM
	from TransitionMatrix.Utilities.Plot.argo_data import Core as Argo
	from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
	from GeneralUtilities.Plot.Cartopy.regional_plot import SOSECartopy
	from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo
	from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
	from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader

	from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list
	from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo,InverseInstance
	full_argo_list()

	def recent_bins_by_sensor(variable,lat_bins,lon_bins,float_type):
		date_list = BGCReader.get_recent_date_list()
		bin_list = BGCReader.get_recent_bins(lat_bins,lon_bins)
		sensor_list = BGCReader.get_sensors()
		sensor_mask = [variable in x for x in sensor_list]
		date_mask =[max(date_list)-datetime.timedelta(days=180)<x for x in date_list]
		if float_type == 'BGC':
			soccom_mask = ['SOCCOM' in x.meta.project_name for dummy,x in BGCReader.all_dict.items()]
			mask = np.array(sensor_mask)&np.array(date_mask)&np.array(soccom_mask)
		else:
			mask = np.array(sensor_mask)&np.array(date_mask)

		age_list = [(x.prof.date._list[-1]-x.prof.date._list[0]).days/365. for x in BGCReader.all_dict.values()]
		return (np.array(bin_list)[mask],1/(np.ceil(age_list)+1)[mask])

	def recent_floats(cls,GeoClass, FloatClass):
		out_list = []
		for variable in GeoClass.variable_list:
			float_var = GeoClass.variable_translation_dict[variable]
			var_grid,age_list = recent_bins_by_sensor(float_var,GeoClass.get_lat_bins(),GeoClass.get_lon_bins(),cls.traj_file_type)
			idx_list = [GeoClass.total_list.index(x) for x in var_grid if x in GeoClass.total_list]
			holder_array = np.zeros([len(GeoClass.total_list),1])
			for k,idx in enumerate(idx_list):
				holder_array[idx]+=age_list[k]
			out_list.append(holder_array)
		out = np.vstack(out_list)
		return cls(out,trans_geo=GeoClass)


	inverse_mat = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=2)
	traj_class_1 = TransPlot.load_from_type(lat_spacing=4,lon_spacing=4,time_step=180)
	traj_class_1.trans_geo.plot_class = SOSECartopy
	traj_class_1.trans_geo.variable_list = ['so']
	traj_class_1.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	float_vector_1 = recent_floats(SOCCOM,traj_class_1.trans_geo, BGCReader)

	traj_class_2 = TransPlot.load_from_type(lat_spacing=2,lon_spacing=2,time_step=180)
	traj_class_2.trans_geo.plot_class = GlobalCartopy
	traj_class_2.trans_geo.variable_list = ['so']
	traj_class_2.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	float_vector_2 = recent_floats(Argo,traj_class_2.trans_geo, ArgoReader)

	bins_lat = traj_class.trans_geo.get_lat_bins()
	bins_lon = traj_class.trans_geo.get_lon_bins()
	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	XX,YY,ax1 = traj_class_1.trans_geo.plot_setup(ax = ax1)
	plottable = traj_class_1.trans_geo.transition_vector_to_plottable(traj_class.todense().dot(float_vector_1.todense()))
	traj_class_1.traj_file_type = 'SOSE'
	ax1.pcolormesh(XX,YY,np.ma.masked_less(plottable,5*10**-3),cmap=plt.cm.YlGn,vmax=0.3)
	PCM = ax1.get_children()[0]
	row_idx,column_idx,data = scipy.sparse.find(float_vector_1)

	lats = [list(traj_class_1.trans_geo.total_list)[x].latitude for x in row_idx]
	lons = [list(traj_class_1.trans_geo.total_list)[x].longitude for x in row_idx]

	ax1.scatter(lons,lats,c='m',marker='*')

	plt.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
	traj_class_2.traj_file_type = 'Argo'
	XX,YY,ax2 = traj_class_2.trans_geo.plot_setup(ax = ax2)
	plottable = traj_class_2.trans_geo.transition_vector_to_plottable(traj_class_2.todense().dot(float_vector_2.todense()))
	ax2.pcolormesh(XX,YY,np.ma.masked_less(plottable,5*10**-3),cmap=plt.cm.YlGn,vmax = 0.3)
	PCM = ax2.get_children()[0]
	fig.colorbar(PCM,ax=[ax1,ax2],fraction=0.10,label='Probability Density/Age')
	row_idx,column_idx,data = scipy.sparse.find(float_vector_2)

	lats = [list(traj_class_2.trans_geo.total_list)[x].latitude for x in row_idx]
	lons = [list(traj_class_2.trans_geo.total_list)[x].longitude for x in row_idx]

	ax2.scatter(lons,lats,c='r',s=4)
	plt.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(file_handler.out_file('death_plot'))
	plt.close()