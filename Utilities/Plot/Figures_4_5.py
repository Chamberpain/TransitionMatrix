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
