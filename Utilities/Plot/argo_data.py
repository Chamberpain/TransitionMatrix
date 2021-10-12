from __future__ import print_function
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import os, sys
import datetime
from scipy.sparse.base import isspmatrix
import scipy.sparse
import matplotlib.pyplot as plt
import geopy
import copy


class Float(scipy.sparse.csc_matrix):
	traj_file_type = 'float'
	marker_color = 'm'
	marker_size = 15

	def __init__(self,*args,trans_geo=None,**kwargs):
		self.trans_geo = trans_geo
		super().__init__(*args,**kwargs)


	def __setitem__(self, index, x):
		# Process arrays from IndexMixin
		i, j = self._unpack_index(index)
		i, j = self._index_to_arrays(i, j)

		if isspmatrix(x):
			broadcast_row = x.shape[0] == 1 and i.shape[0] != 1
			broadcast_col = x.shape[1] == 1 and i.shape[1] != 1
			if not ((broadcast_row or x.shape[0] == i.shape[0]) and
					(broadcast_col or x.shape[1] == i.shape[1])):
				raise ValueError("shape mismatch in assignment")

			# clear entries that will be overwritten
			ci, cj = self._swap((i.ravel(), j.ravel()))
			self._zero_many(ci, cj)

			x = x.tocoo(copy=True)
			x.sum_duplicates()
			r, c = x.row, x.col
			x = np.asarray(x.data, dtype=self.dtype)
			if broadcast_row:
				r = np.repeat(np.arange(i.shape[0]), len(r))
				c = np.tile(c, i.shape[0])
				x = np.tile(x, i.shape[0])
			if broadcast_col:
				r = np.repeat(r, i.shape[1])
				c = np.tile(np.arange(i.shape[1]), len(c))
				x = np.repeat(x, i.shape[1])
			# only assign entries in the new sparsity structure
			i = i[r, c]
			j = j[r, c]
		else:
			# Make x and i into the same shape
			x = np.asarray(x, dtype=self.dtype)
			x, _ = np.broadcast_arrays(x, i)

			if x.shape != i.shape:
				raise ValueError("shape mismatch in assignment")

		if np.size(x) == 0:
			return
		i, j = self._swap((i.ravel(), j.ravel()))
		self._set_many(i, j, x.ravel())
		lat,lon = self.total_list[index]
		for _ in range(x):
			self.df = pd.concat([self.df,pd.DataFrame({'latitude':[lat],'longitude':[lon]})])

class Core(Float):
	traj_file_type = 'Core'
	marker_color = 'r'
	marker_size = 5

	@classmethod
	def recent_floats(cls,GeoClass, FloatClass):
		var_grid = FloatClass.recent_bins(GeoClass.get_lat_bins(),GeoClass.get_lon_bins())
		idx_list = [GeoClass.total_list.index(x) for x in var_grid if x in GeoClass.total_list]
		holder_array = np.zeros([len(GeoClass.total_list),1])
		for idx in idx_list:
			holder_array[idx]+=1
		return cls(holder_array,trans_geo=GeoClass)

class BGC(Float):
	traj_file_type = 'BGC'	
	marker_color = 'm'
	marker_size = 20

	@classmethod
	def recent_floats(cls,GeoClass, FloatClass):
		out_list = []
		for variable in GeoClass.variable_list:
			float_var = GeoClass.variable_translation_dict[variable]
			var_grid = FloatClass.recent_bins_by_sensor(float_var,GeoClass.get_lat_bins(),GeoClass.get_lon_bins())
			idx_list = [GeoClass.total_list.index(x) for x in var_grid if x in GeoClass.total_list]
			holder_array = np.zeros([len(GeoClass.total_list),1])
			for idx in idx_list:
				holder_array[idx]+=1
			out_list.append(holder_array)
		out = np.vstack(out_list)
		return cls(out,trans_geo=GeoClass)

	def get_sensor(self,row_var):
		row_idx = self.trans_geo.variable_list.index(row_var)
		split_array = np.split(self.todense(),len(self.trans_geo.variable_list))[row_idx]
		trans_geo = copy.deepcopy(self.trans_geo)
		trans_geo.variable_list = [row_var]
		return BGC(split_array,split_array.shape,trans_geo=trans_geo)