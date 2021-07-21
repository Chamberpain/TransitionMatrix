from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list
from TransitionMatrix.Utilities.Compute.trans_read import TransMat
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from TransitionMatrix.Utilities.Plot.transition_matrix_plot import TransPlot
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')


def figure_3():
	trans_mat = TransMat.load_from_type(lat_spacing=2,lon_spacing=2,time_step=30)
	aggregate_argo_list()
	lat_list,lon_list = ArgoReader.get_full_lat_lon_list()
	pos_list = ArgoReader.get_pos_list()
	argos_start_lat_list = [x[0] for x,y in zip(lat_list,pos_list) if y=='ARGOS']
	argos_start_lon_list = [x[0] for x,y in zip(lon_list,pos_list) if y=='ARGOS']

	gps_start_lat_list = [x[0] for x,y in zip(lat_list,pos_list) if y=='GPS']
	gps_start_lon_list = [x[0] for x,y in zip(lon_list,pos_list) if y=='GPS']
	print('length of GPS list is ',len(gps_start_lon_list))
	print('length of ARGOS list is ',len(argos_start_lon_list))
	fig = plt.figure(figsize=(12,7))
	ax1 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
	XX,YY,ax1 = trans_mat.trans_geo.plot_setup(ax=ax1)
	ax1.scatter(argos_start_lon_list,argos_start_lat_list,s=0.5,c='r',label='ARGOS')
	ax1.scatter(gps_start_lon_list,gps_start_lat_list,s=0.5,c='b',label='GPS')
	ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
      		ncol=3, fancybox=True, shadow=True)

	tp = TransPlot.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)
	ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
	XX,YY,ax2 = tp.trans_geo.plot_setup(ax=ax2)
	ax2 = tp.number_plot(ax=ax2)

	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)

	plt.savefig(file_handler.out_file('initial_deployments'))
	plt.close()

def figure_6():
	fig = plt.figure(figsize=(16,7))
	ax1 = fig.add_subplot(1,2,1, projection=ccrs.PlateCarree())
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

	ax2 = fig.add_subplot(1,2,2, projection=ccrs.PlateCarree())
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
	plt.colorbar(PCM,ax=ax2,label='Mean Standard Error (%)')	
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(file_handler.out_file('standard_error_resolution_compare'))
	plt.close()
