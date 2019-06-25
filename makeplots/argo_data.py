import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from plot_utils import basemap_setup,transition_vector_to_plottable
import matplotlib.pyplot as plt
import os, sys
import datetime
# get an absolute path to the directory that contains mypackage
try:
    make_plot_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    make_plot_dir = os.path.dirname(os.path.join(os.getcwd(),'dummy'))
sys.path.append(os.path.normpath(os.path.join(make_plot_dir, '../compute/')))
from transition_matrix_compute import Transition


class Float(object):
	def __init__(self,transition_plot,**kwds):
		self.bins_lat = transition_plot.bins_lat
		self.bins_lon = transition_plot.bins_lon
		self.list = transition_plot.list
		self.base_file = transition_plot.base_file

	def reshape_float_vector(self,age_return):
		self.df['bins_lat'] = pd.cut(self.df.Lat,bins = self.bins_lat,labels=self.bins_lat[:-1])
		self.df['bins_lon'] = pd.cut(self.df.Lon,bins = self.bins_lon,labels=self.bins_lon[:-1])
		self.df['bin_index'] = zip(self.df['bins_lat'].values,self.df['bins_lon'].values)
		float_vector = np.zeros(len(self.list))
		for x,age in self.df[['bin_index','Age']].values:
			try:
				idx = self.list.index(list(x))
# this is terrible notation, but it finds the index in the index list of the touple bins_lat, bins_lon
				if age_return:
					percent = 1/np.ceil(age)
					if percent < 1/6.:
						percent = 0
					float_vector[idx] = percent
				else:
					float_vector[idx] = 1
			except ValueError:
				print str(x)+' is not found'
		assert (float_vector<=1).all()
		self.vector = float_vector	

	def look_at_data(self):
		XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type)    
		plottable = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.index_list,self.vector)
		m.pcolormesh(XX,YY,plottable)
		plt.show()

class Argo(Float):
	def __init__(self,age_return=False,**kwds):
		super(Argo,self).__init__(**kwds)
		file_ = '../../argo_traj_box/ar_index_global_prof.txt'
		df_ = pd.read_csv(file_,skiprows=8)
		cruise_list = []
		for item in df_['file'].iteritems():
			try:
				cruise_list.append(item[1].split('/')[1])
			except IndexError:
				cruise_list.append(np.nan)
		df_['Cruise']=cruise_list
		df_ = df_[~df_[['date','Cruise']].isna().any(axis=1)]
		df_['Date'] = pd.to_datetime([int(_) for _ in df_.date.values.tolist()],format='%Y%m%d%H%M%S')
		df_['Lat'] = df_['latitude']
		df_['Lon'] = df_['longitude']
		df_ = df_[['Lat','Lon','Date','Cruise']]
		active_floats = df_[df_.Date>(df_.Date.max()-relativedelta(months=6))].Cruise.unique()
		df_ = df_[df_.Cruise.isin(active_floats)]
		df_ = df_.drop_duplicates(subset='Cruise',keep='first')
		df_['Age'] = np.ceil((df_.Date.max()-df_.Date).dt.days/360.)
		df_.loc[df_.Age<1,'Age'] = 1
		self.df = df_
		self.reshape_float_vector(age_return)

class SOCCOM(Float):
	def __init__(self,age_return=False,**kwds):
		super(SOCCOM,self).__init__(**kwds)
		path = self.base_file+'../SOCCOM_trajectory/'
		files = []
		# r=root, d=directories, f = files
		for r, d, f in os.walk(path):
			for file in f:
				if '.txt' in file:
					files.append(os.path.join(r, file))
		df_list = []
		for file in files:
			pd.read_csv(file)
			df_holder = pd.read_csv(file,skiprows=[1,2,3],delim_whitespace=True,usecols=['Float_ID','Cycle','Date','Time','Lat','Lon','POS_QC','#'])
			df_holder.columns = ['Float_ID','Cycle','Date','Time','Lat','Lon','POS_QC','Cruise'] 
			df_holder['Date'] = pd.to_datetime(df_holder['Date'],format='%Y%m%d')
			df_holder = df_holder[df_holder.POS_QC.isin([0,1])]
			if (datetime.datetime.today()-df_holder.Date.tail(1)).dt.days.values[0]>270:
				print 'Float is dead, rejecting'
				continue
			df_token = df_holder[['Lat','Lon']].tail(1)
			df_token['Age'] = ((df_holder.Date.tail(1)-df_holder.Date.head(1).values).dt.days/365).values[0]
			df_list.append(df_token)
		self.df = pd.concat(df_list)
		self.reshape_float_vector(age_return)
