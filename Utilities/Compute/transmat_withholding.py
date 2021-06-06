from TransitionMatrix.Utilities.Compute.trans_read import TransMat,TransitionGeo,SOSEGeo,SummerGeo,WinterGeo,SummerSOSEGeo,WinterSOSEGeo,ARGOSGeo,GPSGeo,WithholdingGeo,SOSEWithholdingGeo
import numpy as np
import copy
import random
from TransitionMatrix.Utilities.Compute.compute_utils import matrix_size_match,matrix_difference_compare
from GeneralUtilities.Compute.list import flat_list,find_nearest
from TransitionMatrix.Utilities.Compute.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler

file_handler = FilePathHandler(ROOT_DIR,'transmat_withholding')

def matrix_resolution_intercomparison():
	datalist = []
	coord_list = [(1,1),(1,2),(2,2),(2,3),(3,3),(4,4),(4,6)]
	for outer in coord_list:
		for inner in coord_list:
			for traj_class in [TransitionGeo,SOSEGeo]:
				for time_step in [30,60,90,120,150,180]:
					if outer==inner:
						continue
					lat_outer,lon_outer = outer
					lat_inner,lon_inner = inner

					if (lat_outer%lat_inner==0)&(lon_outer%lon_inner==0): #outer lat will always be greater than inner lat
						max_len = lat_outer/lat_inner*lon_outer/lon_inner
						max_lat = abs(lat_outer-lat_inner)
						max_lon = abs(lon_outer-lon_inner)              
						print('they are divisable')

						outer_class = TransMat.load_from_type(GeoClass=traj_class,lat_spacing = lat_outer,lon_spacing = lon_outer,time_step = time_step)
						outer_class = outer_class.dot(outer_class)
						inner_class = TransMat.load_from_type(GeoClass=traj_class,lat_spacing = lat_inner,lon_spacing = lon_inner,time_step = time_step)
						inner_class = inner_class.dot(inner_class)
						high_res_outer_class = inner_class.reduce_resolution(lat_outer,lon_outer)
						matrix_1, matrix_2 = matrix_size_match(outer_class,high_res_outer_class)
						datalist.append(matrix_difference_compare(matrix_1,matrix_2))
	with open(file_handler.tmp_file('resolution_difference_data'), 'wb') as fp:
		pickle.dump(datalist, fp)
	fp.close()

def data_withholding_calc():
	coord_list = [(2,2),(2,3),(3,3),(4,4),(4,6)]
	for base_traj_class,traj_class in ((TransitionGeo,WithholdingGeo),(SOSEGeo,SOSEWithholdingGeo)):
		for percentage in [0.95,0.9,0.85,0.8,0.75,0.7]:
			for k in range(5):
				for coord in coord_list:
					for time_step in [30,60,90]:
						lat,lon = coord 
						base_mat = TransMat.load_from_type(GeoClass=base_traj_class,lat_spacing = lat,lon_spacing = lon,time_step = time_step)
						traj_geo = WithholdingGeo(percentage,k,lat_sep=lat,lon_sep=lon,time_step=time_step)
						withholding_mat = TransMat.load(traj_geo.make_filename())
						matrix_1,matrix_2 = matrix_size_match(base_mat,withholding_mat)
						datalist.append(matrix_difference_compare(matrix_1,matrix_2))
	with open(file_handler.tmp_file('transition_matrix_withholding_data'), 'wb') as fp:
		pickle.dump(datalist, fp)
	fp.close()



def matrix_datespace_intercomparison():
	datalist = []
	datelist = [60,90,120,150,180]
	coord_list = [(1,1),(1,2),(2,2),(2,3),(3,3),(4,4),(4,6)]
	for lat,lon in coord_list:
		for date2 in datelist:
			for traj_class in [TransitionGeo,SOSEGeo]:
				outer_class = TransMat.load_from_type(GeoClass=traj_class,lat_spacing = lat,lon_spacing = lon,time_step = date2)
				inner_class = TransMat.load_from_type(GeoClass=traj_class,lat_spacing = lat,lon_spacing = lon,time_step = 30)

				outer_class,inner_class = matrix_size_match(outer_class,inner_class)
				matrix_token = copy.deepcopy(inner_class)
				for dummy in range(int(date2/30)-1):
					inner_class = inner_class.dot(matrix_token)
				matrix_1 , matrix_2 = matrix_size_match(outer_class,inner_class)
				datalist.append(matrix_difference_compare(matrix_1,matrix_2))
	with open(file_handler.tmp_file('datespace_data'), 'wb') as fp:
		pickle.dump(datalist, fp)
	fp.close()


def matrix_seasonal_intercomparison():
	datalist = []
	datelist = [60,90,120,150,180]
	coord_list = [(2,2),(2,3),(3,3),(4,4),(4,6)]
	for lat,lon in coord_list:
		for date in datelist:
			for summer_class,winter_class in [(SummerGeo,WinterGeo)]:
				outer_class = TransMat.load_from_type(GeoClass=summer_class,lat_spacing = lat,lon_spacing = lon,time_step = date)
				inner_class = TransMat.load_from_type(GeoClass=winter_class,lat_spacing = lat,lon_spacing = lon,time_step = date)
				outer_class,inner_class = matrix_size_match(outer_class,inner_class)
				matrix_1 , matrix_2 = matrix_size_match(outer_class,inner_class)
				datalist.append(matrix_difference_compare(matrix_1,matrix_2))
	with open(file_handler.tmp_file('seasonal_data'), 'wb') as fp:
		pickle.dump(datalist, fp)
	fp.close()


def matrix_ARGOS_intercomparison():
	datalist = []
	datelist = [60,90,120,150,180]
	coord_list = [(2,2),(2,3),(3,3),(4,4),(4,6)]
	argos_class = ARGOSGeo
	gps_class = GPSGeo
	for lat,lon in coord_list:
		for date in datelist:
			outer_class = TransMat.load_from_type(GeoClass=argos_class,lat_spacing = lat,lon_spacing = lon,time_step = date)
			inner_class = TransMat.load_from_type(GeoClass=gps_class,lat_spacing = lat,lon_spacing = lon,time_step = date)
			outer_class,inner_class = matrix_size_match(outer_class,inner_class)
			matrix_1 , matrix_2 = matrix_size_match(outer_class,inner_class)
			datalist.append(matrix_difference_compare(matrix_1,matrix_2))
	with open(file_handler.tmp_file('argos_gps_data'), 'wb') as fp:
		pickle.dump(datalist, fp)
	fp.close()