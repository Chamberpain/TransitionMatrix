from __future__ import print_function
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from transition_matrix.makeplots.plot_utils import basemap_setup,transition_vector_to_plottable,plottable_to_transition_vector
import os, sys
import datetime
from scipy.sparse.base import isspmatrix
import scipy.sparse
from transition_matrix.compute.trans_read import BaseMat


class Float(scipy.sparse.csc_matrix):
	traj_file_type = 'float'
	marker_color = 'm'
	marker_size = 15

	def __init__(self,arg,shape=None,df=pd.DataFrame({'latitude':[],'longitude':[]}),degree_bins=None,total_list=None):
		super(Float,self).__init__(arg, shape=shape)
		self.degree_bins = degree_bins
		self.total_list = total_list
		self.df = df

	@ staticmethod
	def return_float_vector(degree_bins,df,total_list):
		bins_lat,bins_lon = BaseMat.bins_generator(degree_bins)
		df['bins_lat'] = pd.cut(df.latitude,bins = bins_lat,labels=bins_lat[:-1])
		df['bins_lon'] = pd.cut(df.longitude,bins = bins_lon,labels=bins_lon[:-1])
		df['bin_index'] = zip(df['bins_lat'].values,df['bins_lon'].values)
		float_vector = np.zeros((len(total_list),1))
		row_idx = []
		data = []
		for x in df.groupby(['bin_index']).count().Cruise.iteritems():
			try:
				row_idx.append(total_list.tolist().index(list(x[0])))
# this is terrible notation, but it finds the index in the index list of the touple bins_lat, bins_lon

				data.append(x[1])
			except ValueError:
				print(str(x)+' is not found')
		col_idx = [0]*len(row_idx)
		arg1 = (data,(row_idx,col_idx))
		shape = (len(total_list),1)
		return (arg1,shape)

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

	def scatter_plot(self,m=False):
		if not m:
			(bins_lat,bins_lon)=BaseMat.bins_generator(self.degree_bins)
			XX,YY,m = basemap_setup(bins_lat,bins_lon,self.traj_file_type)  
		y = self.df.latitude.tolist()
		x = self.df.longitude.tolist()
		m.scatter(x,y,marker='*',color=self.marker_color,s=self.marker_size,latlon=True)
		return m

	def grid_plot(self,m=False):
		(bins_lat,bins_lon)=BaseMat.bins_generator(self.degree_bins)
		if not m:	
			XX,YY,m = basemap_setup(bins_lat,bins_lon,self.traj_file_type)
		plottable = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,np.array(self.todense()).ravel())
		plottable = np.ma.masked_equal(plottable,0)
		m.pcolormesh(XX,YY,plottable,vmin=0,vmax=min(100,self.data.max()))


	def get_age(self):
		age = np.ceil((df.date.max()-df.date).dt.days/360.).array
		self.df.loc[df_.Age<1,'Age'] = 1

	@staticmethod
	def get_recent_df():
		from argo_traj_box.argo_traj_box_utils import load_df
		import datetime
		df = load_df()
		recent_df = df[df.date>(df.date.max()-datetime.timedelta(days=60))]
		recent_df = recent_df.drop_duplicates(subset=['Cruise'],keep='last')
		return recent_df


class Argo(Float):
	traj_file_type = 'argo'
	marker_color = 'r'
	marker_size = 5
	variables = ['temp','salt']
	def __init__(self,arg,**kwds):		
		super(Argo,self).__init__(arg,**kwds)


	@staticmethod
	def recent_floats(degree_bins,total_list):	
		df = Float.get_recent_df()
		arg1,shape = Float.return_float_vector(degree_bins,df,total_list)
		return Argo(arg1,shape=shape,df = df,degree_bins = degree_bins,total_list = total_list)


class SOCCOM(Float):
	traj_file_type = 'argo'	
	marker_color = 'm'
	marker_size = 20
	variables = ['temp','salt','dic','o2']
	def __init__(self,arg,**kwds):	
		super(SOCCOM,self).__init__(arg,**kwds)


	@staticmethod
	def recent_floats(degree_bins,total_list):
		df = Float.get_recent_df()
		df = df[df['SOCCOM']]
		arg1,shape = Float.return_float_vector(degree_bins,df,total_list)
		return SOCCOM(arg1,shape=shape,df = df,degree_bins = degree_bins,total_list = total_list)

