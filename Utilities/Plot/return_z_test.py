from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.TransGeo import TransitionGeo
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
import scipy
import numpy as np 
import matplotlib.colors as colors
from TransitionMatrix.Utilities.Plot.Figure_6 import return_standard_error
from TransitionMatrix.Utilities.TransGeo import ARGOSGeo,GPSGeo
from TransitionMatrix.Utilities.Utilities import matrix_size_match


def return_standard_deviation(self):
	number_matrix = self.new_sparse_matrix(self.number_data.tolist())
	self.trans_geo.get_direction_matrix()
	row_list, column_list, data_array = scipy.sparse.find(self)
	n_s_distance_weighted = self.trans_geo.north_south[row_list,column_list]*data_array
	e_w_distance_weighted = self.trans_geo.east_west[row_list,column_list]*data_array
	# this is like calculating x*f(x)
	n_s_mat = self.new_sparse_matrix(n_s_distance_weighted)
	E_y = np.array(n_s_mat.sum(axis=0)).flatten()
	e_w_mat = self.new_sparse_matrix(e_w_distance_weighted)
	E_x = np.array(e_w_mat.sum(axis=0)).flatten()
	#this is like calculating E(x) = sum(xf(x)) = mean
	ns_x_minus_mu = (self.trans_geo.north_south[row_list,column_list]-E_y[column_list])**2
	ew_x_minus_mu = (self.trans_geo.east_west[row_list,column_list]-E_x[column_list])**2
	std_data = (ns_x_minus_mu+ew_x_minus_mu)*data_array
	std_mat = self.new_sparse_matrix(std_data)
	sigma = np.array(np.sqrt(std_mat.sum(axis=0))).flatten()
	return np.array(sigma).flatten()


TransMat.return_standard_deviation = return_standard_deviation
TransMat.return_standard_error = return_standard_error

lat = 2
lon = 3 
date = 180
argos_class = TransMat.load_from_type(GeoClass=ARGOSGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
gps_class = TransMat.load_from_type(GeoClass=GPSGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
argos_class,gps_class = matrix_size_match(argos_class,gps_class)

east_west_null, north_south_null = argos_class.return_mean()
east_west_sample, north_south_sample = gps_class.return_mean()
std = argos_class.return_standard_deviation()

z_test = np.sqrt((east_west_sample - east_west_null)**2 + (north_south_sample - north_south_null)**2)/std
z_test = np.ma.masked_array(z_test,mask=((z_test>5)|(z_test<-5)))
plottable = argos_class.trans_geo.transition_vector_to_plottable(z_test)
plt.pcolor(plottable)
plt.colorbar()
plt.show()