from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from GeneralUtilities.Plot.Cartopy.regional_plot import SOSECartopy
from GeneralUtilities.Filepath.instance import FilePathHandler
from GeneralUtilities.Compute.list import find_nearest,flat_list,LonList,LatList,GeoList
import numpy as np
import geopy

from TransitionMatrix.Utilities.__init__ import ROOT_DIR

file_handler = FilePathHandler(ROOT_DIR,'TransMat')

class GeoBase(object):
	""" geo information and tools for transition matrices """
	plot_class = GlobalCartopy
	file_type = 'argo'
	number_vmin=0
	number_vmax=250
	std_vmax=50
	def __init__(self,lat_sep=2,lon_sep=2):
		assert isinstance(lat_sep,int)
		assert isinstance(lon_sep,int)

		self.lat_sep = lat_sep
		self.lon_sep = lon_sep

	def plot_setup(self,ax=False,**kwargs):
		XX,YY,ax = self.plot_class(self.get_lat_bins(),self.get_lon_bins(),ax=ax,**kwargs).get_map()
		return (XX,YY,ax)

	def set_total_list(self,total_list):
		lats,lons,dummy = zip(*[tuple(x) for x in total_list])
		lons = [x if x<180 else x-360 for x in lons]
		total_list = GeoList([geopy.Point(x) for x in zip(lats,lons)],lat_sep=self.lat_sep, lon_sep=self.lon_sep)
		lats,lons = total_list.lats_lons()
		assert isinstance(total_list,GeoList) 
		#total list must be a geolist
		assert (set(lats).issubset(set(self.get_lat_bins())))&(set(lons).issubset(set(self.get_lon_bins())))
		# total list must be a subset of the coordinate lists
		total_list = GeoList(total_list,lat_sep=self.lat_sep,lon_sep=self.lon_sep)
		self.total_list = total_list #make sure they are unique

	def get_lat_bins(self):
		lat_grid,lon_grid = GeoList([],lat_sep=self.lat_sep,lon_sep=self.lon_sep).return_dimensions()
		return lat_grid

	def get_lon_bins(self):
		lat_grid,lon_grid = GeoList([],lat_sep=self.lat_sep,lon_sep=self.lon_sep).return_dimensions()
		return lon_grid

	def get_coords(self):
		XX,YY = np.meshgrid(self.get_lon_bins(),self.get_lat_bins())
		return (XX,YY)

	def transition_vector_to_plottable(self,vector):
		lon_grid = self.get_lon_bins()
		lat_grid = self.get_lat_bins()
		plottable = np.zeros([len(lon_grid),len(lat_grid)])
		plottable = np.ma.masked_equal(plottable,0)
		for n,pos in enumerate(self.total_list):
			ii_index = lon_grid.index(pos.longitude)
			qq_index = lat_grid.index(pos.latitude)
			plottable[ii_index,qq_index] = vector[n]
		return plottable.T

	def plottable_to_transition_vector(self,plottable):
		vector = np.zeros([len(index_list)])
		lon_grid = self.get_lon_bins().tolist()
		lat_grid = self.get_lat_bins().tolist()
		for n,pos in enumerate(self.total_list):
			lon_index = lon_grid.index(find_nearest(lon_grid,pos.longitude))
			assert abs(lon_grid[lon_index]-lon)<2
			lat_index = lat_grid.index(find_nearest(lat_grid,pos.longitude))
			assert abs(lat_grid[lat_index]-lat)<2
			vector[n] = plottable[lat_index,lon_index]
		return vector

	def get_direction_matrix(self):
		"""
		notes: this could be made faster by looping through unique values of lat and lon and assigning intelligently
		"""
		lat_list,lon_list = zip(*self.total_list.tuple_total_list())
		lat_list = np.array(lat_list)
		lon_list = np.array(lon_list)
		pos_max = 180/self.lon_sep #this is the maximum number of bins possible
		output_ns_list = []
		output_ew_list = []
		for token_lat,token_lon in self.total_list.tuple_total_list():
			token_ns = (token_lat-np.array(lat_list))/self.lat_sep
			token_ew = (token_lon-np.array(lon_list))/self.lon_sep
			token_ew[token_ew>pos_max]=token_ew[token_ew>pos_max]-2*pos_max #the equivalent of saying -360 degrees
			token_ew[token_ew<-pos_max]=token_ew[token_ew<-pos_max]+2*pos_max #the equivalent of saying +360 degrees
			output_ns_list.append(token_ns)
			output_ew_list.append(token_ew)
		self.east_west = np.array(output_ew_list)
		self.north_south = np.array(output_ns_list)
		assert (self.east_west<=180/self.lon_sep).all()
		assert (self.east_west>=-180/self.lon_sep).all()
		assert (self.north_south>=-180/self.lat_sep).all()
		assert (self.north_south<=180/self.lat_sep).all()



class TransitionGeo(GeoBase):
	def __init__(self,*args,time_step=60,**kwargs):
		super().__init__(*args,**kwargs)
		assert isinstance(time_step,int)
		self.time_step = time_step

	def make_filename(self):
		return file_handler.tmp_file(self.file_type+'-'+str(self.time_step)+'-'+str(self.lat_sep)+'-'+str(self.lon_sep))

	@classmethod
	def new_from_old(cls,trans_geo):
		new_trans_geo = cls(lat_sep=trans_geo.lat_sep,lon_sep=trans_geo.lon_sep,time_step=trans_geo.time_step)
		new_trans_geo.set_total_list(trans_geo.total_list)
		assert isinstance(new_trans_geo.total_list,GeoList) 
		return new_trans_geo

class SOSEGeo(TransitionGeo):
	plot_class = SOSECartopy
	file_type = 'SOSE'
	number_vmin=0
	number_vmax=900
	std_vmax=15

class SOSECaseGeo(SOSEGeo):
	def make_filename(self):
		return file_handler.tmp_file(self.description+'_'+self.file_type+'-'+str(self.time_step)+'-'+str(self.lat_sep)+'-'+str(self.lon_sep))

class SummerSOSEGeo(SOSECaseGeo):
	description = 'Summer'

class WinterSOSEGeo(SOSECaseGeo):
	description = 'Winter'

class CaseGeo(TransitionGeo):
	def make_filename(self):
		return file_handler.tmp_file(self.description+'_'+self.file_type+'-'+str(self.time_step)+'-'+str(self.lat_sep)+'-'+str(self.lon_sep))

class ARGOSGeo(CaseGeo):
	description = 'ARGOS_Positioning'

class GPSGeo(CaseGeo):
	description = 'GPS_Positioning'

class SummerGeo(CaseGeo):
	description = 'Summer'

class WinterGeo(CaseGeo):
	description = 'Winter'

class WithholdingGeo(CaseGeo):
	def __init__(self,percentage,idx_number,*args,**kwargs):
		self.description = str(percentage)+'_'+str(idx_number)
		super().__init__(*args,**kwargs)

	@classmethod
	def new_from_old(cls,trans_geo):
		return trans_geo

class SOSEWithholdingGeo(SOSECaseGeo):
	def __init__(self,percentage,idx_number,*args,**kwargs):
		self.description = str(percentage)+'_'+str(idx_number)
		super().__init__(*args,**kwargs)

	@classmethod
	def new_from_old(cls,trans_geo):
		return trans_geo
