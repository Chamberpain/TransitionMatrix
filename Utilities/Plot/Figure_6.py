from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.TransGeo import TransitionGeo
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
import scipy
import numpy as np 


plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')


def return_standard_error(self):
	number_matrix = self.new_sparse_matrix(self.number_data)
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
	std_error = sigma/np.sqrt(number_matrix.sum(axis=0))
	return np.array(std_error).flatten()

TransMat.return_standard_error = return_standard_error

def figure_6():
	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	high_res = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
	XX,YY,ax1 = high_res.trans_geo.plot_setup(ax=ax1)	
	standard_error = high_res.return_standard_error()
	standard_error_plot = high_res.trans_geo.transition_vector_to_plottable(standard_error)
	standard_error_plot = np.ma.masked_greater(standard_error_plot,100)
	number_matrix = high_res.new_sparse_matrix(high_res.number_data)
	k = number_matrix.sum(axis=0)
	k = k.T
	standard_error_plot = np.ma.array(standard_error_plot,mask=high_res.trans_geo.transition_vector_to_plottable(k)==0)
	ax1.pcolormesh(XX,YY,standard_error_plot*100,cmap=plt.cm.cividis,vmax=high_res.trans_geo.std_vmax)

	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
	high_res = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 4,lon_spacing = 4,time_step = 90)
	XX,YY,ax2 = high_res.trans_geo.plot_setup(ax=ax2)	
	standard_error = high_res.return_standard_error()
	standard_error_plot = high_res.trans_geo.transition_vector_to_plottable(standard_error)
	standard_error_plot = np.ma.masked_greater(standard_error_plot,100)
	number_matrix = high_res.new_sparse_matrix(high_res.number_data)
	k = number_matrix.sum(axis=0)
	k = k.T
	standard_error_plot = np.ma.array(standard_error_plot,mask=high_res.trans_geo.transition_vector_to_plottable(k)==0)
	ax2.pcolormesh(XX,YY,standard_error_plot*100,cmap=plt.cm.cividis,vmax=high_res.trans_geo.std_vmax)
	PCM = ax2.get_children()[0]
	fig.colorbar(PCM,ax=[ax1,ax2],fraction=0.10,label='Mean Standard Error (%)')
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(file_handler.out_file('figure_6'))
	plt.close()