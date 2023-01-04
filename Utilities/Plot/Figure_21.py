import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.TransGeo import TransitionGeo,SOSEGeo
from TransitionMatrix.Utilities.Utilities import figure_20_matrix_compare
import cartopy.crs as ccrs
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR

file_handler = FilePathHandler(ROOT_DIR,'final_figures')
plt.rcParams['font.size'] = '16'

fig = plt.figure(figsize=(36,14))

ax1 = fig.add_subplot(1,2,1, projection=ccrs.SouthPolarStereo())
lat = 1 
lon =2 
date = 180
base_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
sose_mat = TransMat.load_from_type(GeoClass=SOSEGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
(ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff) = figure_20_matrix_compare(sose_mat,base_mat)
XX,YY,ax1 = sose_mat.trans_geo.plot_setup(ax = ax1)
ax1.pcolormesh(XX,YY,ns_std_diff+ew_std_diff,vmin=-1,vmax=1,cmap='bwr',transform=ccrs.PlateCarree())

ax2 = fig.add_subplot(1,2,2, projection=ccrs.SouthPolarStereo())
lat = 4 
lon =6 
date = 180
base_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
sose_mat = TransMat.load_from_type(GeoClass=SOSEGeo,lat_spacing = lat,lon_spacing = lon,time_step = date)
(ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff) = figure_20_matrix_compare(sose_mat,base_mat)
XX,YY,ax2 = sose_mat.trans_geo.plot_setup(ax = ax2)
ax2.pcolormesh(XX,YY,ns_std_diff+ew_std_diff,vmin=-1,vmax=1,cmap='bwr',transform=ccrs.PlateCarree())
ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
PCM = ax2.get_children()[4]
plt.colorbar(PCM,ax=[ax1,ax2],fraction=0.10,label='Transition Diffusion')	
plt.savefig(file_handler.out_file('figure_21'))
plt.close()
