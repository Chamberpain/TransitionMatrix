import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Compute.__init__ import ROOT_DIR as data_root
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.Compute.trans_read import TransMat,TransitionGeo,SOSEGeo,SummerGeo,WinterGeo,SummerSOSEGeo,WinterSOSEGeo,ARGOSGeo,GPSGeo,WithholdingGeo,SOSEWithholdingGeo
import cartopy.crs as ccrs
import pickle
import numpy as np
import geopy

data_handler = FilePathHandler(data_root,'transmat_withholding')
plot_handler = FilePathHandler(ROOT_DIR,'transition_matrix_withholding_plot')

plot_style_dict = {'argo':'--','SOSE':':'}
plot_color_dict = {(1,1):'teal',(1,2):'brown',(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}


def matrix_compare(matrix_1,matrix_2):
	east_west_lr, north_south_lr = matrix_1.return_mean()
	east_west_lr = matrix_1.trans_geo.transition_vector_to_plottable(east_west_lr)
	north_south_lr = matrix_1.trans_geo.transition_vector_to_plottable(north_south_lr)

	east_west_hr, north_south_hr = matrix_2.return_mean()
	east_west_hr = matrix_2.trans_geo.transition_vector_to_plottable(east_west_hr)
	north_south_hr = matrix_2.trans_geo.transition_vector_to_plottable(north_south_hr)

	ew_std_lr, ns_std_lr = matrix_1.return_std()
	ew_std_lr = matrix_1.trans_geo.transition_vector_to_plottable(ew_std_lr)
	ns_std_lr = matrix_1.trans_geo.transition_vector_to_plottable(ns_std_lr)

	ew_std_hr, ns_std_hr = matrix_2.return_std()
	ew_std_hr = matrix_2.trans_geo.transition_vector_to_plottable(ew_std_hr)
	ns_std_hr = matrix_2.trans_geo.transition_vector_to_plottable(ns_std_hr)

	ew_mean_diff = (east_west_lr-east_west_hr)
	ns_mean_diff = (north_south_lr-north_south_hr)
	ew_std_diff = (ew_std_lr-ew_std_hr)
	ns_std_diff = (ns_std_lr-ns_std_hr)
	return (ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff)

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

def sose_compare():
	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	lat = 1 
	lon =2 
	date = 180
	base_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
	sose_mat = TransMat.load_from_type(GeoClass=SOSEGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
	(ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff) = matrix_compare(sose_mat,base_mat)
	XX,YY,ax1 = sose_mat.trans_geo.plot_setup(ax = ax1)
	ax1.pcolormesh(XX,YY,ns_std_diff+ew_std_diff,vmin=-1,vmax=1,cmap='bwr')

	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
	lat = 4 
	lon =6 
	date = 180
	base_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
	sose_mat = TransMat.load_from_type(GeoClass=SOSEGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
	(ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff) = matrix_compare(sose_mat,base_mat)
	XX,YY,ax2 = sose_mat.trans_geo.plot_setup(ax = ax2)
	ax2.pcolormesh(XX,YY,ns_std_diff+ew_std_diff,vmin=-1,vmax=1,cmap='bwr')	
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	PCM = ax2.get_children()[0]
	plt.colorbar(PCM,ax=[ax1,ax2],fraction=0.10,label='Transition Diffusion')	
	plt.savefig(plot_handler.out_file('sose_total_std_diff'))
	plt.close()

def argos_gps_spatial_plot():
	lat = 2
	lon = 3 
	date = 180
	argos_class = TransMat.load_from_type(GeoClass=ARGOSGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
	gps_class = TransMat.load_from_type(GeoClass=GPSGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)

	for k in range(4):
		print(k)
		argos_class = argos_class.dot(argos_class)

	for k in range(4):
		print(k)
		gps_class = gps_class.dot(gps_class)


	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	XX,YY,ax1 = argos_class.trans_geo.plot_setup(ax=ax1)
	plottable = np.array(argos_class.sum(axis=1)).flatten()
	ax1.pcolor(XX,YY,argos_class.trans_geo.transition_vector_to_plottable(plottable)*100,vmin=40,vmax=160)

	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
	XX,YY,ax2 = gps_class.trans_geo.plot_setup(ax = ax2)
	plottable = np.array(gps_class.sum(axis=1)).flatten()
	ax2.pcolor(XX,YY,gps_class.trans_geo.transition_vector_to_plottable(plottable)*100,vmin=40,vmax=160)


	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	PCM = ax2.get_children()[0]
	fig.colorbar(PCM,ax=[ax1,ax2],label='Argo Density (%)',fraction=0.10)
	plt.savefig(plot_handler.out_file('argos_gps_comparison'))
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


