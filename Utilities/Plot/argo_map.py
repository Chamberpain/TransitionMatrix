import numpy as np
import geopy
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list,full_argo_list
from TransitionMatrix.Utilities.ArgoData import Core
from TransitionMatrix.Utilities.TransGeo import TransitionGeo
from TransitionMatrix.Utilities.TransMat import TransMat
from GeneralUtilities.Compute.list import VariableList
import cartopy.crs as ccrs
from TransitionMatrix.Utilities.TransMat import TransMat
import matplotlib.pyplot as plt
import matplotlib
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
from GeneralUtilities.Compute.constants import degree_dist


file_handler = FilePathHandler(ROOT_DIR,'final_figures')

def get_cmap():
	norm = matplotlib.colors.Normalize(0,3*2*degree_dist/256.)
	colors = [[norm(0./256.), "dodgerblue"],
			  [norm(1.5*degree_dist/256.), "deepskyblue"],
			  [norm(3/np.sqrt(2)*degree_dist/256.), "skyblue"],
			  [norm(3*degree_dist/256.), "powderblue"],
			  [norm(3*np.sqrt(2)*degree_dist/256.), "lightyellow"],
			  [norm(3*np.sqrt(3)*degree_dist/256.), "lightgoldenrodyellow"],
			  [norm(3*2*degree_dist/256.), "yellow"]
			  ]
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
	return cmap


full_argo_list()

trans_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)
cmap = get_cmap()


FloatClass = ArgoReader

float_pos_list = Core.recent_pos_list(ArgoReader)

lon_list = np.arange(-180,180.1,0.25)
lat_list = np.arange(-90,90.1,0.25)

XX,YY = np.meshgrid(lon_list,lat_list)
coord_list = [geopy.Point(x) for x in zip(YY.flatten(),XX.flatten())]
dist_list = []
for n,coord in enumerate(coord_list):
	print(n)
	dummy_dist_list = [geopy.distance.GreatCircleDistance(coord,x).km for x in float_pos_list]
	dist_list.append(min(dummy_dist_list))
dist_array = np.array(dist_list).reshape(XX.shape)
dist_array[dist_array>3*2*degree_dist] = 3*2*degree_dist

trans_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
fig = plt.figure(figsize=(15,10))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
dummy,dummy,ax = trans_mat.trans_geo.plot_setup(ax=ax)
ax.pcolor(XX,YY,dist_array,cmap=cmap)
PCM = ax.get_children()[3]
plt.colorbar(PCM,location = 'bottom',label = 'Distance to Nearest Float (km)')
plt.savefig(file_handler.out_file('argo_map'))
plt.close()