from GeneralUtilities.Compute.list import find_nearest,flat_list
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list
from GeneralUtilities.Data.lagrangian.sose.SOSE_read import SOSEReader,aggregate_sose_list
from GeneralUtilities.Data.lagrangian.drifter_base_class import BaseRead
from trans_read import TransitionGeo,TransMat,WinterGeo,SummerGeo,GPSGeo,ARGOSGeo,WinterSOSEGeo,SummerSOSEGeo,SOSEGeo

import numpy as np
import matplotlib.pyplot as plt
import random
import pandas as pd
import scipy.spatial as spatial
import geopy
import scipy.sparse.linalg
import copy

class SetToDealWithGeo(set):
	def __init__(self,item):
		super().__init__(item)

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
	def __init__(self,*args,**kwargs):
		super().__init__(*args,**kwargs)


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

			start_indexes =get_index_of_decorrelated(item,time_delta = trans_geo.time_step)
			end_indexes = [find_next_time_index(item,x,time_delta = trans_geo.time_step) for x in start_indexes]
			mask=[i for i,v in enumerate(end_indexes) if v != None]
			start_indexes = np.array(start_indexes)[mask].tolist()
			end_indexes = np.array(end_indexes)[mask].tolist()
			return (start_indexes,end_indexes)



		print('Getting the space and time bins')
		start_bin_list = []
		end_bin_list = []
		for wmoid,item in self.items():
			if not item.prof.pos._list:
				continue
			#get the time indexes
			start_pos_indexes,end_pos_indexes = get_time_start_and_end_indexes(item,trans_geo)
			#get the bin indexes
			start_bin_list+=list(item.prof.pos.return_pos_bins(trans_geo.get_lat_bins(),trans_geo.get_lon_bins(),
			index_values=start_pos_indexes))
			end_bin_list+=list(item.prof.pos.return_pos_bins(trans_geo.get_lat_bins(),trans_geo.get_lon_bins(),
			index_values=end_pos_indexes))
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

class PosBinList(object):
	""" class to inheret bin lists but with masking functionality 
	Input: list of geopy Points
	output: list of tuples plus dictionary of indexes and functionality"""
	def __init__(self,start_list,end_list):
		self.start_list = [tuple(x) for x in start_list]
		self.end_list = [tuple(x) for x in end_list]
		self.start_dict = self.make_dict(self.start_list)
		self.end_dict = self.make_dict(self.end_list)

	def make_dict(self,list_token):
		d = {}
		for idx,value in enumerate(list_token):

			try:
				d[tuple(value)].append(idx)
			except KeyError:
				d[tuple(value)] = [idx]
		return d

	def list_from_mask(self,masked_values):
		new_start_list = self.start_list[:]
		new_end_list = self.end_list[:]
		remove_idxs = []
		for value in masked_values:
			try:
				remove_idxs += self.start_dict[value]
			except KeyError:
				continue
		for value in masked_values:
			try:
				remove_idxs += self.end_dict[value]
			except KeyError:
				continue
		for idx in np.sort(np.unique(remove_idxs))[::-1]:
			del new_start_list[idx]
			del new_end_list[idx]
		return (new_start_list,new_end_list)

	def get_trans_idx_and_numbers(self,trans_geo,mask):
		masked_start,masked_end = self.list_from_mask(mask)
		transition_dict = self.make_dict(list(zip(masked_start,masked_end)))
		total_list = [tuple(x) for x in trans_geo.total_list]
		col_list = []
		row_list = []
		data_list = []
		for (start_idx,end_idx),idx_list in transition_dict.items():
			start_idx
			col_list.append(total_list.index(start_idx))
			row_list.append(total_list.index(end_idx))
			data_list.append(len(idx_list))
		return (col_list,row_list,data_list)


def create_total_list(pos_obj,mask):
	print('Creating the total list')
	masked_start_list,masked_end_list = pos_obj.list_from_mask(mask)
	masked_start_set = set(masked_start_list)
	masked_end_set = set(masked_end_list)
	while (len(masked_start_set.difference(masked_end_list))>0)|(len(masked_end_set.difference(masked_start_list))>0):
		mask += list(masked_start_set.symmetric_difference(masked_end_list))
		masked_start_list,masked_end_list = pos_obj.list_from_mask(mask)
		masked_start_set = set(masked_start_list)
		masked_end_set = set(masked_end_list)
	return ([geopy.Point(x) for x in list(masked_start_set)],mask)

def not_enough_numbers(pos_obj,mask,num=3):
	print('Checking if there are not enough numbers')
	mask_list = []
	masked_start,masked_end = pos_obj.list_from_mask(mask)
	for start_idx,idx_list in pos_obj.make_dict(masked_start).items():		
		if len(idx_list)<num:
			mask_list.append(start_idx)
	return mask_list

def isolated_points(trans_geo,mask):
	print('Checking for isolated points')
	tree = spatial.KDTree(trans_geo.tuple_total_list())
	closest_bins_spacing,_ = tree.query(trans_geo.tuple_total_list(),k=2)
	idx_list = np.where(closest_bins_spacing[:,1]>np.sqrt(trans_geo.lat_sep**2+trans_geo.lon_sep**2))[0].tolist()
	return [trans_geo.tuple_total_list()[x] for x in idx_list]

def lone_eigen_vectors(pos_obj,trans_geo,mask):
	XX,YY = trans_geo.get_coords()

	print('Checking for lone eigen vectors')
	col_idx,row_idx,data = pos_obj.get_trans_idx_and_numbers(trans_geo,mask)
	transition_matrix = TransMat((data,(row_idx,col_idx)),trans_geo=trans_geo,number_data = data,rescale=True)
	transition_matrix.asfptype()
	print('begin calculating eigen vectors')
	eig_val,eig_vecs = scipy.sparse.linalg.eigs(transition_matrix,k=50)
	print('calculated the eigen vectors')				
	checksum = [abs(eig_vecs[:,k]).max()*0.05 for k in range(eig_vecs.shape[1])]
	idx_list = [k for k in range(eig_vecs.shape[1]) if (abs((eig_vecs[:,k]))>checksum[k]).sum()<=3]
	pos_idx_list = [abs(eig_vecs[:,idx]).tolist().index(abs(eig_vecs[:,idx]).max()) for idx in idx_list]
	tuple_list = trans_geo.tuple_total_list()
	mask = [tuple_list[x] for x in pos_idx_list]
	return mask

def mask_compute(pos_obj,trans_geo):
	mask = []
	compute = True
	while compute:
		print('mask length is ',len(mask))
		total_list,mask = create_total_list(pos_obj,mask)
		trans_geo.set_total_list(total_list)
		number_mask = not_enough_numbers(pos_obj,mask)
		if number_mask:
			mask += number_mask
			continue
		isolated_mask = isolated_points(trans_geo,mask)
		if isolated_mask:
			mask += isolated_mask
			continue
		eigen_mask = lone_eigen_vectors(pos_obj,trans_geo,mask)
		if eigen_mask:
			mask += eigen_mask
			continue
		compute = False
	return mask

def withholding_calc():
	for agg_function,ReadToken,GeoToken in [(aggregate_argo_list,ArgoReader,WithholdingGeo),(aggregate_sose_list,SOSEReader,SOSEWithholdingGeo)]:
		ReadToken.all_dict = []
		agg_function()
		for degree_bins in [(1,1),(1,2),(2,2),(2,3),(3,3),(4,4),(4,6)]:
			lat_sep,lon_sep = degree_bins
			for percentage in [0.95,0.9,0.85,0.8,0.75,0.7]:
				for time_step in [30,60,90,120,150,180]:
					for k in range(10):
						all_dict = ProfileDict(ReadToken.get_subsampled_float_dict(percentage))
						trans_geo = GeoToken(percentage,k,lat_sep=lat_sep,lon_sep=lon_sep,time_step=time_step)
						start_bin_list, end_bin_list = all_dict.space_and_time_bins(trans_geo)
						pos_obj = PosBinList(start_bin_list,end_bin_list)
						mask = mask_compute(pos_obj,trans_geo)
						col_idx,row_idx,data = pos_obj.get_trans_idx_and_numbers(trans_geo,mask)
						transition_matrix = TransMat((data,(row_idx,col_idx)),trans_geo=trans_geo,number_data = data,rescale=True)
						transition_matrix.asfptype()
						transition_matrix.save()

def base_calc():
	for agg_function,ReadToken,GeoToken in [(aggregate_argo_list,ArgoReader,TransitionGeo),(aggregate_sose_list,SOSEReader,SOSEWithholdingGeo)]:
		ReadToken.all_dict = []
		agg_function()
		for degree_bins in [(1,1),(1,2),(2,2),(2,3),(3,3),(4,4),(4,6)]:
			lat_sep,lon_sep = degree_bins
			for time_step in [30,60,90,120,150,180]:
				all_dict = ProfileDict(ReadToken.get_subsampled_float_dict(1))
				trans_geo = GeoToken(lat_sep=lat_sep,lon_sep=lon_sep,time_step=time_step)
				start_bin_list, end_bin_list = all_dict.space_and_time_bins(trans_geo)
				pos_obj = PosBinList(start_bin_list,end_bin_list)
				mask = mask_compute(pos_obj,trans_geo)
				col_idx,row_idx,data = pos_obj.get_trans_idx_and_numbers(trans_geo,mask)
				transition_matrix = TransMat((data,(row_idx,col_idx)),trans_geo=trans_geo,number_data = data,rescale=True)
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
					all_dict = ProfileToken(ReadToken.get_subsampled_float_dict(1))
					trans_geo = GeoToken(lat_sep=lat_sep,lon_sep=lon_sep,time_step=time_step)
					start_bin_list, end_bin_list = all_dict.space_and_time_bins(trans_geo)
					pos_obj = PosBinList(start_bin_list,end_bin_list)
					mask = mask_compute(pos_obj,trans_geo)
					col_idx,row_idx,data = pos_obj.get_trans_idx_and_numbers(trans_geo,mask)
					transition_matrix = TransMat((data,(row_idx,col_idx)),trans_geo=trans_geo,number_data = data,rescale=True)
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
				all_dict = DictToken(ArgoReader.all_dict)
				trans_geo = GeoToken(lat_sep=lat_sep,lon_sep=lon_sep,time_step=time_step)
				start_bin_list, end_bin_list = all_dict.space_and_time_bins(trans_geo)
				pos_obj = PosBinList(start_bin_list,end_bin_list)
				mask = mask_compute(pos_obj,trans_geo)
				col_idx,row_idx,data = pos_obj.get_trans_idx_and_numbers(trans_geo,mask)
				transition_matrix = TransMat((data,(row_idx,col_idx)),trans_geo=trans_geo,number_data = data,rescale=True)
				transition_matrix.asfptype()
				transition_matrix.save()