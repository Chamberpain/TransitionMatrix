import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.TransGeo import ARGOSGeo,GPSGeo
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
import cartopy.crs as ccrs
import numpy as np

file_handler = FilePathHandler(ROOT_DIR,'final_figures')

lat = 2
lon = 3 
date = 180
argos_class = TransMat.load_from_type(GeoClass=ARGOSGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
gps_class = TransMat.load_from_type(GeoClass=GPSGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)

for k in range(4):
	print(k)
	argos_class = argos_class.dot(argos_class)

for k in range(4):
	print(k)
	gps_class = gps_class.dot(gps_class)


fig = plt.figure(figsize=(20,14))
ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
XX,YY,ax1 = argos_class.trans_geo.plot_setup(ax=ax1)
plottable = np.array(argos_class.sum(axis=1)).flatten()
ax1.pcolor(XX,YY,argos_class.trans_geo.transition_vector_to_plottable(plottable)*100,vmin=40,vmax=160)

ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
XX,YY,ax2 = gps_class.trans_geo.plot_setup(ax = ax2)
plottable = np.array(gps_class.sum(axis=1)).flatten()
ax2.pcolor(XX,YY,gps_class.trans_geo.transition_vector_to_plottable(plottable)*100,vmin=40,vmax=160)


ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
PCM = ax2.get_children()[3]
fig.colorbar(PCM,ax=[ax1,ax2],label='Argo Density (%)',fraction=0.10)
plt.savefig(file_handler.out_file('figure_20'))
plt.close()
