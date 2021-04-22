import numpy as np
from netCDF4 import Dataset
import fnmatch
import sys,os
try:
	data_save_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
	data_save_dir = os.path.dirname(os.path.join(os.getcwd(),'dummy'))
import pandas as pd
sys.path.append(os.path.normpath(os.path.join(data_save_dir, '../compute/')))
sys.path.append(os.path.normpath(os.path.join(data_save_dir, '../makeplots/')))
import datetime
from compute_utils import find_nearest
from plot_utils import basemap_setup
import matplotlib.pyplot as plt
import pickle
import matplotlib.cm as cm
from sets import Set
import matplotlib.dates as mdates

class EulerianDataBase(object):
	def __init__(self,**kwds):
		self.base_lat = np.arange(-89,90,1)
		self.base_lon = np.arange(-180,180,1)

	def find_files(self,dir_,ext_):
		matches = []
		for root, dirnames, filenames in os.walk(dir_):
		    for filename in fnmatch.filter(filenames,ext_):
		        matches.append(os.path.join(root, filename))
		return matches

	def nearest_routine(self,dummy,list):
		nearest = find_nearest(list,dummy)
		if abs(nearest-dummy)>2:
			return False
		else:
			return nearest

	def setup_base_index(self,lat,lon):
		lat_idx = []
		for dummy in self.base_lat:
			nearest = self.nearest_routine(dummy,lat)
			if nearest:
				lat_idx.append(lat.tolist().index(nearest))
		lon_idx = []
		for dummy in self.base_lon:
			nearest = self.nearest_routine(dummy,lon)
			if nearest:
				lon_idx.append(lon.tolist().index(nearest))
		return (lat_idx,lon_idx)

	def plot_array(self,X,Y):
		XX,YY,m = basemap_setup(self.base_lat,self.base_lon,'Argo')
		m.pcolormesh(XX,YY,self.array.std(axis=0),cmap=self.cmap,vmin=self.vmin,vmax=self.vmax)
		plt.colorbar(label=self.unit)
		x,y = m(X,Y)
		plt.plot(x,y,'y*',markersize=10)

	def line_plot(self,X,Y,min_date):
		x_idx = self.base_lon.tolist().index(find_nearest(self.base_lat,X))
		y_idx = self.base_lat.tolist().index(find_nearest(self.base_lon,Y))

		time_series = self.array[:,y_idx,x_idx]
		time_series = time_series-time_series.mean()
		time_series = time_series/abs(time_series).max()

		dates = np.array(self.date_list)
		time_series = time_series[dates>min_date]

		std = np.std(time_series)
		plt.plot(dates[dates>min_date],time_series,label=self.label)
		plt.fill_between(dates[dates>min_date],time_series-std,time_series+std,alpha=0.3)

	def annual_plot(self,X,Y,ax):
		x_idx = self.base_lon.tolist().index(find_nearest(self.base_lat,X))
		y_idx = self.base_lat.tolist().index(find_nearest(self.base_lon,Y))

		time_series = self.array[:,y_idx,x_idx]


		dates = [_.timetuple().tm_yday for _ in self.date_list]  

		df = pd.DataFrame({'juld':dates,'data':time_series})   
		df['bins'] = pd.cut(dates,np.arange(0,390,15))
		group_df = df.groupby('bins')['data']
		data = group_df.mean().dropna().array.to_numpy()
		data_date = group_df.mean().dropna().index.array

		std = df.groupby('bins')['data'].std().dropna().array.to_numpy()
		std_date = df.groupby('bins')['data'].std().dropna().index.array

		intersection_set = Set(std_date).intersection(Set(data_date))
		data = data[data_date.isin(list(intersection_set))]
		std = std[std_date.isin(list(intersection_set))]
		date = data_date[data_date.isin(list(intersection_set))]


		std = std/abs(data).max()


		data = data-data.mean()
		data = data/abs(data).max()


		date = [_.left  for _ in date]
		date = [datetime.datetime(2000,1,1)+datetime.timedelta(days=_) for _ in date]

		plt.plot(np.array(date),data,label=self.label)
		plt.fill_between(np.array(date),data-std,data+std,alpha=0.3)
		ax.xaxis.set_major_formatter(mdates.DateFormatter('%m'))
		plt.ylim([-1.1,1.1])

class LandschutzerData(EulerianDataBase):
	def __init__(self,**kwds):
		super(LandschutzerData,self).__init__(**kwds)
		try:
			f = ('../data/landschutzer_array_processed.pkl',"rb")
			data_dict = pickle.load(f)
			self.array = data_dict['array']
			self.date_list = data_dict['date_list']
			self.lon_idx = data_dict['lon_idx']
			self.lat_idx = data_dict['lat_idx']
		except AttributeError:
			self.load_first_time_data()
		self.label = 'Landschutzer'
		self.cmap = self.cm = cm.RdBu
		self.unit = '$CO_2$ Flux $(gm\ C\ m^{-2}\ yr^{-1})^2$'
		self.vmax = 1
		self.vmin = 0.2

	def load_first_time_data(self):
		file_ = '../data/spco2_MPI_SOM-FFN_v2018.nc'
		nc_fid = Dataset(file_)
		self.lat_idx,self.lon_idx = self.setup_base_index(nc_fid['lat'][:],nc_fid['lon'][:])
		data = np.ma.masked_greater(nc_fid['fgco2_smoothed'][:],10**19) 
		self.date_list = self.time_parser(nc_fid['time'][:])
		dummy_time,dummy_lat,dummy_lon = np.meshgrid(range(len(self.date_list)),self.lat_idx,self.lon_idx,indexing='ij')
		self.array=data[dummy_time.flatten(),dummy_lat.flatten(),dummy_lon.flatten()].reshape(dummy_time.shape)
		save_dict = {'array':self.array,'lat_idx':self.lat_idx,'lon_idx':self.lon_idx,'date_list':self.date_list}
		f = open('../data/landschutzer_array_processed.pkl',"wb")
		pickle.dump(save_dict,f)
		f.close()

	def time_parser(self,time_list):
		#timelist is expressed in seconds before jan 1 2000
		base_time = datetime.datetime(2000,1,1)
		format_time_list = [base_time+datetime.timedelta(seconds=int(_)) for _ in time_list]
		return format_time_list

class MODISData(EulerianDataBase):
	def __init__(self,**kwds):
		super(MODISData,self).__init__(**kwds)
		try:
			f = open('../data/MODIS/array_processed.pkl',"rb")
			data_dict = pickle.load(f)
			self.array = data_dict['array']
			self.date_list = data_dict['date_list']
			self.lon_idx = data_dict['lon_idx']
			self.lat_idx = data_dict['lat_idx']
		except IOError:
			self.load_first_time_data()
		self.label='MODIS'
		self.unit = '$Chlor_a\ (mg\ m^-3)^2$'
		self.cmap = cm.PiYG
		self.vmax=0.7
		self.vmin=0.05
	def load_first_time_data(self):
		files = self.find_files('../data/MODIS/','*.nc')
		array_list = []
		date_list = []
		for n,file in enumerate(files):
			print n
			print len(files)
			nc_fid = Dataset(file)
			if n==0:
				self.lat_idx,self.lon_idx = self.setup_base_index(nc_fid.variables['lat'][:],nc_fid.variables['lon'][:])
			unicode_time = nc_fid.time_coverage_end
			date = self.time_parser(unicode_time)
			date_list.append(date)
			array = nc_fid.variables['chlor_a'][self.lat_idx,self.lon_idx]
			array_list.append(array)
		array_list = [x for _,x in sorted(zip(date_list,array_list))]
		self.date_list = sorted(date_list)
		self.array = np.ma.stack(array_list)		
		save_dict = {'array':self.array,'lat_idx':self.lat_idx,'lon_idx':self.lon_idx,'date_list':self.date_list}
		f = open('../data/MODIS/array_processed.pkl',"wb")
		pickle.dump(save_dict,f)
		f.close()

	def time_parser(self,unicode_time):
		return datetime.datetime.strptime(unicode_time, '%Y-%m-%dT%H:%M:%S.%fZ')

class CM2p6(EulerianDataBase):
	def __init__(self,**kwds):
		super(CM2p6,self).__init__(**kwds)
		pass

	def load_first_time_data(self,data_file,save_file):
		data_list = np.load(data_file)
		data_list = data_list[::10,:]
		lat_list = np.load('../data/lat_list.npy')
		lon_list = np.load('../data/lon_list.npy')
		lon_list[lon_list<-180] = lon_list[lon_list<-180]+360

		XX,YY = np.meshgrid(self.base_lon,self.base_lat)
		total_list = zip(XX.flatten(),YY.flatten())
		translation_list =[]
		for x in total_list:
			mask = (lon_list==x[0])&(lat_list==x[1])
			t = np.where(mask)
			if t[0].size>0:
				translation_list.append(t[0][0])
			else:
				translation_list.append(len(lon_list))
		nan_list = np.array([np.nan]*data_list.shape[0]).reshape([data_list.shape[0],1])
		dat = np.hstack([data_list,np.array(nan_list)])
		dat = dat[:,translation_list]
		self.array = dat.reshape(dat.shape[0],XX.shape[0],XX.shape[1])
		self.lat_idx = translation_list
		self.lon_idx = translation_list
		date_start = datetime.datetime(1998,1,1)
		self.date_list = [date_start+datetime.timedelta(days=(10*_)) for _ in range(self.array.shape[0])]

		save_dict = {'array':self.array,'lat_idx':self.lat_idx,'lon_idx':self.lon_idx,'date_list':self.date_list}
		f = open(save_file,"wb")
		pickle.dump(save_dict,f)
		f.close()

class CM2p6_O2(CM2p6):
	def __init__(self,**kwds):
		super(CM2p6,self).__init__(**kwds)
		try:
			f = open('../data/cm2p6_o2_array_processed.pkl',"rb")
			data_dict = pickle.load(f)
			self.array = data_dict['array']
			self.date_list = data_dict['date_list']
			self.lon_idx = data_dict['lon_idx']
			self.lat_idx = data_dict['lat_idx']
		except (ValueError,IOError),e:
			self.load_first_time_data('../data/subsampled_o2_100m.npy','../data/cm2p6_o2_array_processed.pkl')
		self.label='CM2p6 $O_2$'
		self.unit = '$O_2\ (mol\ m^{-2})^2$'
		self.cmap = cm.BrBG
		self.vmax=1.5
		self.vmin=0.2
class CM2p6_DIC(CM2p6):
	def __init__(self,**kwds):
		super(CM2p6,self).__init__(**kwds)
		try:
			f = open('../data/cm2p6_DIC_array_processed.pkl',"rb")
			data_dict = pickle.load(f)
			self.array = data_dict['array']
			self.date_list = data_dict['date_list']
			self.lon_idx = data_dict['lon_idx']
			self.lat_idx = data_dict['lat_idx']
		except (ValueError,IOError),e:
			self.load_first_time_data('../data/subsampled_DIC_100m.npy','../data/cm2p6_DIC_array_processed.pkl')
		self.label='CM2p6 DIC'
		self.unit = '$DIC\ (mol\ m^{-2})^2$'
		self.cmap = cm.PRGn
		self.vmax=3
		self.vmin=0.5

def run_plot():
	bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.9)
	q = MODISData()
	q.array = np.log(q.array)
	w = LandschutzerData()
	e = CM2p6_O2()
	f = CM2p6_DIC()
	fig = plt.figure(figsize=(10,10))
	ax = fig.add_subplot(3,2,1)
	ax.text(0.1, 0.85, "a", ha="center", va="center",transform=ax.transAxes, size=20,
        bbox=bbox_props)
	plt.title('MODIS')
	q.plot_array(1,-51)
	ax = fig.add_subplot(3,2,2)
	ax.text(0.1, 0.85, "b", ha="center", va="center",transform=ax.transAxes, size=20,
        bbox=bbox_props)
	w.plot_array(1,-51)
	plt.title('Landschutzer')
	ax = fig.add_subplot(3,2,3)
	ax.text(0.1, 0.85, "c", ha="center", va="center",transform=ax.transAxes, size=20,
        bbox=bbox_props)
	e.plot_array(1,-51)
	plt.title('CM2.6 $O_2$')
	ax = fig.add_subplot(3,2,4)
	ax.text(0.1, 0.85, "d", ha="center", va="center",transform=ax.transAxes, size=20,
    bbox=bbox_props)
	f.plot_array(1,-51)
	plt.title('CM2.6 DIC')	
	ax = fig.add_subplot(313)
	ax.text(0.1, 0.85, "e", ha="center", va="center",transform=ax.transAxes, size=20,
    bbox=bbox_props)
	q.annual_plot(1,-51,ax)
	w.annual_plot(1,-51,ax)
	e.annual_plot(1,-51,ax)
	f.annual_plot(1,-51,ax)
	plt.title('Seasonal Cycle')
	plt.xlabel('Month')
	plt.ylabel('Scaled Observation')
	plt.legend()
	plt.savefig('observation_var')