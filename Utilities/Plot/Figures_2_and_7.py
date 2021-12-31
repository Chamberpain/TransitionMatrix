from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list
from TransitionMatrix.Utilities.TransMat import TransMat
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
import numpy as np 
from matplotlib.axes import Axes
from cartopy.mpl.geoaxes import GeoAxes
GeoAxes._pcolormesh_patched = Axes.pcolormesh


plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')


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
fig = plt.figure(figsize=(40,14))
ax1 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
XX,YY,ax1 = trans_mat.trans_geo.plot_setup(ax=ax1)
ax1.scatter(argos_start_lon_list,argos_start_lat_list,s=1,c='r',label='ARGOS',zorder=11)
ax1.scatter(gps_start_lon_list,gps_start_lat_list,s=1,c='b',label='GPS',zorder=11)
ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
  		ncol=3, fancybox=True, shadow=True, markerscale=20)
plt.savefig(file_handler.out_file('Figure_2'))
plt.close()



tp = TransMat.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)
fig = plt.figure(figsize=(36,14))
ax2 = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
XX,YY,ax2 = tp.trans_geo.plot_setup(ax=ax2)
number_matrix = tp.new_sparse_matrix(tp.number_data)
k = number_matrix.sum(axis=0)
k = k.T
print(k)
number_matrix_plot = tp.trans_geo.transition_vector_to_plottable(k)
XX,YY,ax2 = tp.trans_geo.plot_setup(ax=ax2)  
number_matrix_plot = np.ma.masked_equal(number_matrix_plot,0)   #this needs to be fixed in the plotting routine, because this is just showing the number of particles remaining
ax2.pcolor(XX,YY,number_matrix_plot,cmap=plt.cm.magma,vmin=tp.trans_geo.number_vmin,vmax=tp.trans_geo.number_vmax)
# plt.title('Transition Density',size=30)
PCM = ax2.get_children()[6]
fig.colorbar(PCM,pad=.15,label='Transition Number',orientation='horizontal',fraction=0.10)
plt.savefig(file_handler.out_file('Figure_7'))
plt.close()