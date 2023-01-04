from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.TransGeo import TransitionGeo
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import scipy
import numpy as np 
import cartopy.crs as ccrs
from TransitionMatrix.Utilities.Utilities import colorline
import cartopy.crs as ccrs

plt.rcParams['font.size'] = '26'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

lon = 2
lat = 2
time_step = 90
trans_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
ew_data_list = []
ns_data_list = []
for k in range(10):
	holder = trans_mat.multiply(k)
	east_west, north_south = holder.return_mean()
	ew_data_list.append(east_west)
	ns_data_list.append(north_south)
ew = np.vstack(ew_data_list)
ns = np.vstack(ns_data_list)

fig = plt.figure(figsize=(40,14))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)
for k,point in enumerate(trans_mat.trans_geo.total_list):
	ew_holder = ew[:,k]
	ns_holder = ns[:,k]
	lons = [point.longitude + x for x in ew_holder]
	lats = [point.latitude + x for x in ns_holder]
	lc = colorline(lons,lats,ax)
plt.colorbar(lc,label='Days Since Deployment')
plt.tight_layout()
plt.savefig(file_handler.out_file('figure_9'))
plt.close()