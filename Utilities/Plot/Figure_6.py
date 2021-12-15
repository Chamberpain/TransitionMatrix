from TransitionMatrix.Utilities.TransMat import TransMat,TransitionGeo
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from TransitionMatrix.Utilities.TransPlot import TransPlot
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
from TransitionMatrix.Utilities.Plot.argo_data import Core,BGC
from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo,InverseInstance,HInstance,InverseCCS,InverseGOM,InverseSOSE
import scipy
import numpy as np 


plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')


def figure_6():
	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	high_res = TransPlot.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
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
	high_res = TransPlot.load_from_type(GeoClass=TransitionGeo,lat_spacing = 4,lon_spacing = 4,time_step = 90)
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
	plt.savefig(file_handler.out_file('standard_error_resolution_compare'))
	plt.close()