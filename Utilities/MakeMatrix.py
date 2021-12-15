from GeneralUtilities.Compute.list import find_nearest,flat_list
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list
from GeneralUtilities.Data.lagrangian.sose.SOSE_read import SOSEReader,aggregate_sose_list
from GeneralUtilities.Data.lagrangian.drifter_base_class import BaseRead
from TransitionMatrix.Utilities.TransGeo import TransitionGeo,WinterGeo,SummerGeo,GPSGeo,ARGOSGeo,WinterSOSEGeo,SummerSOSEGeo,SOSEGeo, SOSEWithholdingGeo, WithholdingGeo
from TransitionMatrix.Utilities.TransMat import TransMat

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import scipy.spatial as spatial
import geopy
import scipy.sparse.linalg
import copy
import os


class SetToDealWithGeo(set):

	def geo_list_from_set(self):
		return [geopy.Point(x) for x in self]

	@staticmethod
	def set_from_geo_list(geo_list):
		geo_list = [tuple(x) for x in geo_list]		
		return SetToDealWithGeo(geo_list)

class ProfileDict(dict):
	"""class that takes the argo read instance as input. This is a cludgy way of doing things because
	it should probably just be inherited, but helped the debug process considerably. 

	Performs parsing logic to determine the proper transitions 
	"""

	bin_dict = {}

	def space_and_time_bins(self,trans_geo):

		def get_time_start_and_end_indexes(item,trans_geo):
			""" finds the start and end indexes for the transition matrix positions
				Performs logic to eliminate values that are out of diff_check tollerances

				Parameters
				----------
				None

				Returns
				-------
				List of start indexes and list of end indexes
				 """

			def get_index_of_decorrelated(item,time_delta=10):
				""" Argo trajectories are correlated on time scales of about 30 days (Gille et al. 2003). This subsampling removes the possibiity
					of dependent data 
					
					Parameters
					----------
					time delta: float that describes the time difference you prescribe as decorrelation (in days)

					Returns
					-------
					list of the indexes of the decorrelated start positions
					 """

				idx_list = []
				seconds_to_days = 1/(3600*24.)
				diff_list = [(_-item.prof.date._list[0]).total_seconds()*seconds_to_days for _ in item.prof.date._list]
				diff_array = np.array(diff_list)
				time_list = np.arange(0,max(diff_list),time_delta)
				for time in time_list:
					idx_list.append(diff_list.index(diff_array[diff_array>=time][0]))
				return idx_list

			def find_next_time_index(item,start_index,time_delta,diff_check=None):
				""" finds the index cooresponding to the nearest time delta from a start index

					Parameters
					----------
					start_index: the index at which you are starting your calculation from
					time delta: float that describes the time difference you want in your list (in days)
					diff_check (optional) : float that describes the acceptable difference away from your time delta

					Returns
					-------
					Index of the next time
					 """
				if not diff_check:
					diff_check = time_delta/3.
				seconds_to_days = 1/(3600*24.)
				diff_list = [(_-item.prof.date._list[start_index]).total_seconds()*seconds_to_days for _ in item.prof.date._list[(start_index+1):]]
				if not diff_list:
					return None
				closest_diff = find_nearest(diff_list,time_delta,test=False)
				if abs(closest_diff-time_delta)>diff_check:
					return None
				else:
					return start_index+diff_list.index(closest_diff)+1

			start_indexes =get_index_of_decorrelated(item,time_delta = max([30,trans_geo.time_step/3]))
			end_indexes = [find_next_time_index(item,x,time_delta = trans_geo.time_step) for x in start_indexes]
			mask=[i for i,v in enumerate(end_indexes) if v != None]
			start_indexes = np.array(start_indexes)[mask].tolist()
			end_indexes = np.array(end_indexes)[mask].tolist()
			return (start_indexes,end_indexes)



		print('Getting the space and time bins')
		start_bin_list = []
		end_bin_list = []
		lat_bins = trans_geo.get_lat_bins()
		lon_bins = trans_geo.get_lon_bins()
		for wmoid,item in self.items():
			if not item.prof.pos._list:
				continue
			#get the time indexes
			start_pos_indexes,end_pos_indexes = get_time_start_and_end_indexes(item,trans_geo)
			#get the bin indexes
			start_bin_holder= list(item.prof.pos.return_pos_bins(lat_bins,lon_bins,
			index_values=start_pos_indexes))
			start_bin_list+=start_bin_holder
			end_bin_holder = list(item.prof.pos.return_pos_bins(lat_bins,lon_bins,
			index_values=end_pos_indexes))
			end_bin_list+= end_bin_holder
			for idx in zip(start_bin_holder,end_bin_holder):
				try:
					self.bin_dict[(tuple(idx[0]),tuple(idx[1]))]+=1
				except KeyError:
					self.bin_dict[(tuple(idx[0]),tuple(idx[1]))]=1
		return (start_bin_list,end_bin_list)

class ARGOSDict(ProfileDict):
	def __init__(self,all_dict,*args,**kwargs):
		token = copy.deepcopy(all_dict)
		key_list = [(key,item) for key,item in token.items() if item.meta.positioning_system=='ARGOS' ]
		token = dict(key_list)
		super().__init__(token,*args,**kwargs)

class GPSDict(ProfileDict):
	def __init__(self,all_dict,*args,**kwargs):
		token = copy.deepcopy(all_dict)
		key_list = [(key,item) for key,item in token.items() if item.meta.positioning_system=='GPS' ]
		token = dict(key_list)
		super().__init__(token,*args,**kwargs)

class SeasonalDict(ProfileDict):
	def __init__(self,all_dict,*args,**kwargs):
		token = copy.deepcopy(all_dict)
		for key,item in token.items():
			month_list = [x.month for x in item.prof.date._list]
			pos_list = item.prof.pos._list[:]
			item.prof.pos._list = [pos for pos,month in zip(pos_list,month_list) if month in self.months]
			item.prof.date._list = [datetime for datetime,month in zip(item.prof.date._list,month_list) if month in self.months]
			assert len(item.prof.date._list)==len(item.prof.pos._list)
		super().__init__(token,*args,**kwargs)

class SummerDict(SeasonalDict):
	months = [4,5,6,7,8,9]
	description = 'Summer'

class WinterDict(SeasonalDict):
	months = [1,2,3,10,11,12]
	description = 'Winter'


def mask_compute(start_bins,end_bins,total_list):

	def not_enough_numbers(matrix_holder,num=4):
		print('Checking if there are not enough numbers')
		return np.where((matrix_holder).sum(axis=0)<=num)[1].tolist()

	def not_enough_rows(matrix_holder,num=2):
		print('Checking if there are not enough numbers')
		return np.where((matrix_holder!=0).sum(axis=0)<=num)[1].tolist()

	def lone_eigen_vectors(matrix_holder,num=3):
		print('Checking for lone eigen vectors')
		eig_val,eig_vecs = scipy.sparse.linalg.eigs(matrix_holder.asfptype(),k=30)
		print('calculated the eigen vectors')				
		checksum = [abs(eig_vecs[:,k]).max()*0.2 for k in range(eig_vecs.shape[1])]
		_idx_list = [k for k in range(eig_vecs.shape[1]) if (abs((eig_vecs[:,k]))>checksum[k]).sum()<=num]
		mask = [abs(eig_vecs[:,idx]).tolist().index(abs(eig_vecs[:,idx]).max()) for idx in _idx_list]
		return mask

	translation_dict = dict(zip(total_list,list(range(len(total_list)))))
	col_idx = []
	row_idx = []
	for k in range(len(start_bins)):
		if k%1000==0:
			print(k)
		col_idx.append(translation_dict[start_bins[k]])
		row_idx.append(translation_dict[end_bins[k]])
	data = list(all_dict.bin_dict.values())
	base_mat = scipy.sparse.csc_matrix((data,(row_idx,col_idx)),shape=(len(total_list),len(total_list)))
	mask = []
	tmp_mask = []
	compute = True
	idx_list = list(range(len(total_list)))
	total_list_holder = copy.deepcopy(total_list)
	while compute:
		print('mask length is ',len(mask))
		for idx in tmp_mask:
			idx_list.remove(idx)
			total_list_holder.remove(total_list[idx])
		trans_mat_holder = base_mat[idx_list,:]
		trans_mat_holder = trans_mat_holder[:,idx_list]
		number_mask = not_enough_numbers(trans_mat_holder)
		if number_mask:
			print('there was a number mask')
			tmp_mask = []
			for idx in number_mask:
				mask.append(translation_dict[total_list_holder[idx]])
				tmp_mask.append(translation_dict[total_list_holder[idx]])
			continue
		row_mask = not_enough_rows(trans_mat_holder)
		if row_mask:
			print('there was a row mask')
			tmp_mask = []
			for idx in row_mask:
				mask.append(translation_dict[total_list_holder[idx]])
				tmp_mask.append(translation_dict[total_list_holder[idx]])
			continue
		eigen_mask = lone_eigen_vectors(trans_mat_holder)
		if eigen_mask:
			print('there was an eigen mask')
			tmp_mask = []
			for idx in eigen_mask:
				try:
					new_idx = translation_dict[total_list_holder[idx]]
					assert new_idx not in mask
					mask.append(new_idx)
					tmp_mask.append(new_idx)
				except AssertionError:
					continue
			continue
		compute = False
	row_idx,column_idx,data = scipy.sparse.find(trans_mat_holder)
	return (row_idx,column_idx,data,total_list_holder)

def withholding_calc():
	for agg_function,ReadToken,GeoToken in [(aggregate_argo_list,ArgoReader,WithholdingGeo),(aggregate_sose_list,SOSEReader,SOSEWithholdingGeo)]:
		BaseRead.all_dict = []
		agg_function()
		for degree_bins in [(1,2),(2,2),(2,3),(3,3),(4,4),(4,6)]:
			lat_sep,lon_sep = degree_bins
			for percentage in [0.95,0.9,0.85,0.8,0.75,0.7]:
				for time_step in [30,60,90,120]:
					for k in range(10):
						trans_geo = GeoToken(percentage,k,lat_sep=lat_sep,lon_sep=lon_sep,time_step=time_step)
						if os.path.isfile(trans_geo.make_filename()):
							print('file ',trans_geo.make_filename(),' already there, continuing')
							continue
						all_dict = ProfileDict(ReadToken.get_subsampled_float_dict(percentage))
						all_dict.bin_dict = {}
						assert len(all_dict)-percentage*len(ReadToken.all_dict)<10
						start_bin_list, end_bin_list = all_dict.space_and_time_bins(trans_geo)
						start_bins,end_bins = zip(*all_dict.bin_dict.keys())
						total_list = list(SetToDealWithGeo.set_from_geo_list(start_bin_list+end_bin_list))

						(row_idx,column_idx,data,total_list) = mask_compute(start_bins,end_bins,total_list)
						# col_idx,row_idx,data = pos_obj.get_trans_idx_and_numbers(trans_geo,mask)
						trans_geo.set_total_list([geopy.Point(x) for x in total_list])
						transition_matrix = TransMat((data,(row_idx,column_idx)),trans_geo=trans_geo,number_data = data,rescale=True)
						transition_matrix.asfptype()
						transition_matrix.save()

def base_calc():
	for agg_function,ReadToken,GeoToken in [(aggregate_argo_list,ArgoReader,TransitionGeo),(aggregate_sose_list,SOSEReader,SOSEGeo)]:
		BaseRead.all_dict = {}
		agg_function()
		for degree_bins in [(1,1),(1,2),(2,2),(2,3),(3,3),(4,4),(4,6)]:
			lat_sep,lon_sep = degree_bins
			for time_step in [30,60,90,120,150,180]:
				trans_geo = GeoToken(lat_sep=lat_sep,lon_sep=lon_sep,time_step=time_step)
				if os.path.isfile(trans_geo.make_filename()):
					print('file ',trans_geo.make_filename(),' already there, continuing')
					continue
				all_dict = ProfileDict(ReadToken.get_subsampled_float_dict(1))
				all_dict.bin_dict = {}
				assert all_dict==ReadToken.all_dict
				start_bin_list, end_bin_list = all_dict.space_and_time_bins(trans_geo)
				start_bins,end_bins = zip(*all_dict.bin_dict.keys())
				total_list = list(SetToDealWithGeo.set_from_geo_list(start_bin_list+end_bin_list))

				(row_idx,column_idx,data,total_list) = mask_compute(start_bins,end_bins,total_list)
				# col_idx,row_idx,data = pos_obj.get_trans_idx_and_numbers(trans_geo,mask)
				trans_geo.set_total_list([geopy.Point(x) for x in total_list])
				transition_matrix = TransMat((data,(row_idx,column_idx)),trans_geo=trans_geo,number_data = data,rescale=True)
				transition_matrix.asfptype()
				transition_matrix.save()

def seasonal_calc():
	for agg_function,ReadToken,token_list in [(aggregate_argo_list,ArgoReader,((SummerGeo,SummerDict),(WinterGeo,WinterDict))),
	(aggregate_sose_list,SOSEReader,((SummerSOSEGeo,SummerDict),(WinterSOSEGeo,WinterDict)))]:
		BaseRead.all_dict = []
		agg_function()
		for degree_bins in [(2,2),(2,3),(3,3),(4,4),(4,6)]:
			lat_sep,lon_sep = degree_bins
			for time_step in [30,60,90,120,150,180]:
				for GeoToken,ProfileToken in token_list:
					trans_geo = GeoToken(lat_sep=lat_sep,lon_sep=lon_sep,time_step=time_step)
					if os.path.isfile(trans_geo.make_filename()):
						print('file ',trans_geo.make_filename(),' already there, continuing')
						continue
					all_dict = ProfileToken(ReadToken.get_subsampled_float_dict(1))
					assert len(all_dict)==len(ReadToken.all_dict)
					all_dict.bin_dict = {}
					start_bin_list, end_bin_list = all_dict.space_and_time_bins(trans_geo)
					start_bins,end_bins = zip(*all_dict.bin_dict.keys())
					total_list = list(SetToDealWithGeo.set_from_geo_list(start_bin_list+end_bin_list))

					(row_idx,column_idx,data,total_list) = mask_compute(start_bins,end_bins,total_list)
					# col_idx,row_idx,data = pos_obj.get_trans_idx_and_numbers(trans_geo,mask)
					trans_geo.set_total_list([geopy.Point(x) for x in total_list])
					transition_matrix = TransMat((data,(row_idx,column_idx)),trans_geo=trans_geo,number_data = data,rescale=True)
					transition_matrix.asfptype()
					transition_matrix.save()

def argos_gps_calc():
	token_list = [(ARGOSGeo,ARGOSDict),(GPSGeo,GPSDict)]
	BaseRead.all_dict = []
	aggregate_argo_list()
	for degree_bins in [(2,2),(2,3),(3,3),(4,4),(4,6)]:
		lat_sep,lon_sep = degree_bins
		for time_step in [30,60,90,120,150,180]:
			for GeoToken,DictToken in token_list:
				trans_geo = GeoToken(lat_sep=lat_sep,lon_sep=lon_sep,time_step=time_step)
				if os.path.isfile(trans_geo.make_filename()):
					print('file ',trans_geo.make_filename(),' already there, continuing')
					continue
				all_dict = DictToken(ArgoReader.all_dict)
				assert len(all_dict)==len(ReadToken.all_dict)
				all_dict.bin_dict = {}
				start_bin_list, end_bin_list = all_dict.space_and_time_bins(trans_geo)
				start_bins,end_bins = zip(*all_dict.bin_dict.keys())
				total_list = list(SetToDealWithGeo.set_from_geo_list(start_bin_list+end_bin_list))

				(row_idx,column_idx,data,total_list) = mask_compute(start_bins,end_bins,total_list)
				# col_idx,row_idx,data = pos_obj.get_trans_idx_and_numbers(trans_geo,mask)
				trans_geo.set_total_list([geopy.Point(x) for x in total_list])
				transition_matrix = TransMat((data,(row_idx,column_idx)),trans_geo=trans_geo,number_data = data,rescale=True)
				transition_matrix.asfptype()
				transition_matrix.save()