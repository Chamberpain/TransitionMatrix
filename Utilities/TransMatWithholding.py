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