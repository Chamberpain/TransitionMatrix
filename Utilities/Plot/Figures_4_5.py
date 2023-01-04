import geopy
from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.TransPlot import TransPlot
from GeneralUtilities.Compute.constants import degree_dist
import matplotlib.pyplot as plt
from GeneralUtilities.Plot.Cartopy.eulerian_plot import PointCartopy
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from GeneralUtilities.Compute.list import GeoList
from TransitionMatrix.Utilities.TransGeo import TransitionGeo,SOSEGeo
import pickle
import numpy as np
import cartopy.crs as ccrs

plt.rcParams['font.size'] = '16'
plot_color_dict = {(1,1):'teal',(1,2):'brown',(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}
file_handler = FilePathHandler(ROOT_DIR,'final_figures')
scale = 1.6
size_scale = 1.6

def distribution_and_mean_of_column(self,geo_point,pad = 6,ax=False):
	from GeneralUtilities.Plot.Cartopy.eulerian_plot import PointCartopy
	col_idx = self.trans_geo.total_list.index(geo_point)
	mean = self.mean_of_column(geo_point)
	XX,YY,ax = PointCartopy(self.trans_geo.total_list[col_idx],lat_grid = self.trans_geo.get_lat_bins(),lon_grid = self.trans_geo.get_lon_bins(),pad=pad,ax=ax).get_map()
	ax.pcolormesh(XX,YY,self.trans_geo.transition_vector_to_plottable(np.array(self[:,col_idx].todense()).flatten()),cmap='Blues')
	ax.scatter(mean.longitude,mean.latitude,c='orange',linewidths=10/scale,marker='x',s=160*size_scale,zorder=10)
	ax.scatter(geo_point.longitude,geo_point.latitude,c='red',linewidths=10/scale,marker='x',s=160*size_scale,zorder=10)

TransMat.distribution_and_mean_of_column = distribution_and_mean_of_column

def resolution_standard_error():
	data_list = []
	for time in [30,60,90,120,150,180]:	
		for lat,lon in [(1,1),(1,2),(2,2),(3,3),(4,4),(2,3),(4,6)]:
			print('lat = ',lat)
			print('lon = ',lon)
			print('time = ',time)
			trans_mat = TransPlot.load_from_type(lat_spacing=lat,lon_spacing=lon,time_step=time)
			standard_error_holder = trans_mat.return_standard_error()
			data_list.append((standard_error_holder.mean(),standard_error_holder.std(),lat,lon,time))

	with open(file_handler.tmp_file('resolution_standard_error'), 'wb') as fp:
		pickle.dump(data_list, fp)
	fp.close()

def temporal_bias_calc():
	from TransitionMatrix.Utilities.Utilities import matrix_compare    
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

def resolution_bias_calc():
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



def figure_4():
	loc = geopy.Point(-54,-54)
	lat = 2
	lon = 2
	short_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = 30)
	short_mat = short_mat.multiply(5)
	long_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = 180)
	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
	long_mat.distribution_and_mean_of_column(loc,ax=ax1,pad=25)
	ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
	short_mat.distribution_and_mean_of_column(loc,ax=ax2,pad=25)
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)

	with open(file_handler.tmp_file('resolution_difference_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,lat,lon,time,pos_type = zip(*datalist)
	ax3 = fig.add_subplot(2,2,3)
	ax4 = fig.add_subplot(2,2,4)
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

			out = degree_dist*np.sqrt(ew_mean_diff_holder**2+ns_mean_diff_holder**2)
			print('I am plotting ',system)
			ax3.plot(np.unique(time),out,color=plot_color_dict[grid],label=r'$%s^\circ\times%s^\circ$'%(grid[0],grid[1]))
			ax3.scatter(np.unique(time),out,color=plot_color_dict[grid])
			out = degree_dist*np.sqrt(ew_std_diff_holder**2+ns_std_diff_holder**2)
			ax4.plot(np.unique(time),out,color=plot_color_dict[grid],label=r'$%s^\circ\times%s^\circ$'%(grid[0],grid[1]))
			ax4.scatter(np.unique(time),out,color=plot_color_dict[grid])
	ax3.set_xlim(28,122)
	ax4.set_xlim(28,122)
	ticks = [30,60,90,120]
	ax3.set_xticks(ticks)
	ax4.set_xticks(ticks)
	ax3.set_ylabel('Misfit (km)')
	ax4.set_ylabel('Misfit (km)')
	ax4.set_xlabel('Timestep (days)')
	ax3.set_xlabel('Timestep (days)')
	ax3.legend(loc='upper center', bbox_to_anchor=(1.1, 1.25),
          ncol=4, fancybox=True, shadow=True)
	ax3.annotate('c', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax4.annotate('d', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.subplots_adjust(hspace=0.35)
	plt.savefig(file_handler.out_file('figure_4'))
	plt.close()

def resolution_bias_plot():
	loc = geopy.Point(-54,-40)
	small_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 180)
	big_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 4,lon_spacing = 4,time_step = 180)
	idx = big_mat.trans_geo.total_list.index(loc)
	reduced_loc_list = big_mat.trans_geo.total_list.reduced_res(idx,2,2)

	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,2,1, projection=ccrs.PlateCarree())
	XX,YY,ax1 = PointCartopy(loc,lat_grid = small_mat.trans_geo.get_lat_bins(),
		lon_grid = small_mat.trans_geo.get_lon_bins(),pad=25,ax=ax1).get_map()

	pdf_list = []
	mean_list = []
	for x in reduced_loc_list:
		small_mean = small_mat.mean_of_column(geopy.Point(x))
		mean_list.append(small_mean)
		pdf_list.append(np.array(small_mat[:,small_mat.trans_geo.total_list.index(geopy.Point(x))].todense()).flatten())
	pdf = small_mat.trans_geo.transition_vector_to_plottable(sum(pdf_list)/4)
	ax1.pcolormesh(XX,YY,pdf,cmap='Blues')
	x_mean,y_mean = zip(*[(x.longitude,x.latitude) for x in mean_list])
	ax1.scatter(x_mean,y_mean,c='lime',linewidths=5/scale,marker='x',s=80*size_scale,zorder=10)
	ax1.scatter(np.mean(x_mean),np.mean(y_mean),c='orange',linewidths=8/scale,marker='x',s=120*size_scale,zorder=10)
	start_lat,start_lon = zip(*reduced_loc_list)
	ax1.scatter(start_lon,start_lat,c='red',linewidths=5/scale,marker='x',s=80*size_scale,zorder=10)
	ax2 = fig.add_subplot(2,2,2, projection=ccrs.PlateCarree())
	big_mat.distribution_and_mean_of_column(loc,ax=ax2,pad=25)
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)


	with open(file_handler.tmp_file('resolution_bias_data'), 'rb') as fp:
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
		ax3.plot(time_axis,plot_mean,label=r'$%s^\circ\times%s^\circ$'%(lat_holder,lon_holder),color=plot_color_dict[(lat_holder,lon_holder)])
		ax3.scatter(time_axis,plot_mean,c=plot_color_dict[(lat_holder,lon_holder)])
	ax4 = fig.add_subplot(2,2,4)
	with open(file_handler.tmp_file('resolution_standard_error'), 'rb') as fp:
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
		ax4.plot(time_axis,plot_mean,label=r'$%s^\circ\times%s^\circ$'%(lat_holder,lon_holder),color=plot_color_dict[(lat_holder,lon_holder)])
		ax4.scatter(time_axis,plot_mean,c=plot_color_dict[(lat_holder,lon_holder)])
	ax4.set_xlabel('Timestep (days)')
	ax3.set_xlabel('Timestep (days)')

	ticks = [30,60,90,120,150,180]
	ax3.set_xticks(ticks)
	ax4.set_xticks(ticks)

	ax3.set_ylabel('Misfit (km)')
	ax4.set_ylabel('Error')
	ax3.annotate('c', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax4.annotate('d', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax3.legend(loc='upper center', bbox_to_anchor=(1.1, 1.25),
          ncol=4, fancybox=True, shadow=True)
	plt.subplots_adjust(hspace=0.35)

	plt.savefig(file_handler.out_file('figure_5'))
	plt.close()
