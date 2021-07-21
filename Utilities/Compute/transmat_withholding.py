from TransitionMatrix.Utilities.Compute.trans_read import TransMat,TransitionGeo,SOSEGeo,SummerGeo,WinterGeo,SummerSOSEGeo,WinterSOSEGeo,ARGOSGeo,GPSGeo,WithholdingGeo,SOSEWithholdingGeo
import numpy as np
import copy
import random
from TransitionMatrix.Utilities.Compute.compute_utils import matrix_size_match,matrix_difference_compare
from GeneralUtilities.Compute.list import flat_list,find_nearest
from TransitionMatrix.Utilities.Compute.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import pickle

file_handler = FilePathHandler(ROOT_DIR,'transmat_withholding')


def matrix_compare(matrix_1,matrix_2,descripton):
	east_west_lr, north_south_lr = matrix_1.return_mean()
	east_west_lr = matrix_1.trans_geo.transition_vector_to_plottable(east_west_lr)
	north_south_lr = matrix_1.trans_geo.transition_vector_to_plottable(north_south_lr)

	east_west_hr, north_south_hr = matrix_2.return_mean()
	east_west_hr = matrix_2.trans_geo.transition_vector_to_plottable(east_west_hr)
	north_south_hr = matrix_2.trans_geo.transition_vector_to_plottable(north_south_hr)

	mask = (~north_south_lr.mask)&(~north_south_hr.mask)

	east_west_lr = east_west_lr.data[mask]
	north_south_lr = north_south_lr.data[mask]
	east_west_hr = east_west_hr.data[mask]
	north_south_hr = north_south_hr.data[mask]

	ew_std_lr, ns_std_lr = matrix_1.return_std()
	ew_std_lr = matrix_1.trans_geo.transition_vector_to_plottable(ew_std_lr)
	ns_std_lr = matrix_1.trans_geo.transition_vector_to_plottable(ns_std_lr)

	ew_std_hr, ns_std_hr = matrix_2.return_std()
	ew_std_hr = matrix_2.trans_geo.transition_vector_to_plottable(ew_std_hr)
	ns_std_hr = matrix_2.trans_geo.transition_vector_to_plottable(ns_std_hr)
					
	ew_std_lr = ew_std_lr.data[mask]
	ns_std_lr = ns_std_lr.data[mask]
	ew_std_hr = ew_std_hr.data[mask]
	ns_std_hr = ns_std_hr.data[mask]

	ew_mean_diff = abs(east_west_lr-east_west_hr).mean()
	ns_mean_diff = abs(north_south_lr-north_south_hr).mean()
	ew_std_diff = abs(ew_std_lr-ew_std_hr).mean()
	ns_std_diff = abs(ns_std_lr-ns_std_hr).mean()
	return (ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,matrix_1.trans_geo.lat_sep,matrix_1.trans_geo.lon_sep,matrix_1.trans_geo.time_step,descripton)

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

def data_withholding_calc():
	datalist = []
	coord_list = [(2,2),(2,3),(3,3),(4,4),(4,6)]
	for base_traj_class,traj_class in ((TransitionGeo,WithholdingGeo),(SOSEGeo,SOSEWithholdingGeo)):
		for percentage in [0.95,0.9,0.85,0.8,0.75,0.7]:
			for k in range(5):
				for coord in coord_list:
					for time_step in [30,60,90]:
						lat,lon = coord 
						print('time step is ',time_step)
						print('lat is ',lat)
						print('lon is ',lon)
						print('percentage is ',percentage)
						base_mat = TransMat.load_from_type(GeoClass=base_traj_class,lat_spacing = lat,lon_spacing = lon,time_step = time_step)
						traj_geo = WithholdingGeo(percentage,k,lat_sep=lat,lon_sep=lon,time_step=time_step)
						withholding_mat = TransMat.load(traj_geo.make_filename())
						datalist.append(matrix_compare(base_mat,withholding_mat,(base_traj_class.file_type,percentage)))
	with open(file_handler.tmp_file('transition_matrix_withholding_data'), 'wb') as fp:
		pickle.dump(datalist, fp)
	fp.close()

def matrix_seasonal_intercomparison():
	datalist = []
	datelist = [30,60,90,120,150,180]
	coord_list = [(2,2),(2,3),(3,3),(4,4),(4,6)]
	for lat,lon in coord_list:
		for date in datelist:
			print('time step is ',date)
			print('lat is ',lat)
			print('lon is ',lon)
			base_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
			summer_class = TransMat.load_from_type(GeoClass=SummerGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
			winter_class = TransMat.load_from_type(GeoClass=WinterGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
			datalist.append(matrix_compare(base_mat,summer_class,'summer'))
			datalist.append(matrix_compare(base_mat,winter_class,'winter'))
	with open(file_handler.tmp_file('seasonal_data'), 'wb') as fp:
		pickle.dump(datalist, fp)
	fp.close()

def matrix_ARGOS_intercomparison():
	datalist = []
	datelist = [30,60,90,120,150,180]
	coord_list = [(2,2),(2,3),(3,3),(4,4),(4,6)]
	for lat,lon in coord_list:
		for date in datelist:
			print('time step is ',date)
			print('lat is ',lat)
			print('lon is ',lon)
			base_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
			argos_class = TransMat.load_from_type(GeoClass=ARGOSGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
			gps_class = TransMat.load_from_type(GeoClass=GPSGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
			datalist.append(matrix_compare(base_mat,argos_class,'argos'))
			datalist.append(matrix_compare(base_mat,gps_class,'gps'))
	with open(file_handler.tmp_file('argos_gps_withholding'), 'wb') as fp:
		pickle.dump(datalist, fp)
	fp.close()	