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
import scipy
from TransitionMatrix.Utilities.Utilities import colorline,get_cmap,shiftgrid


plt.rcParams['font.size'] = '16'
full_argo_list()
trans_mat = TransMat.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)
file_handler = FilePathHandler(ROOT_DIR,'ArgoMap')
plot_handler = FilePathHandler(ROOT_DIR,'final_figures')

ew_data_list = []
ns_data_list = []
for k in range(10):
	holder = trans_mat.multiply(k)
	east_west, north_south = holder.return_mean()
	ew_data_list.append(east_west)
	ns_data_list.append(north_south)
ew = np.vstack(ew_data_list)
ns = np.vstack(ns_data_list)

trans_mat.trans_geo.variable_list = VariableList(['thetao','so','ph','chl','o2'])
trans_mat.trans_geo.variable_translation_dict = {'thetao':'TEMP','so':'PSAL','ph':'PH_IN_SITU_TOTAL','chl':'CHLA','o2':'DOXY'}

def get_cmap():
	norm = matplotlib.colors.Normalize(-2/256.,2/256.)
	colors = [[norm(-2/256.), "yellow"],
			  [norm(-1/256.), "lightgoldenrodyellow"],
			  [norm(0/256.), "lightyellow"],
			  [norm(1./256.), "deepskyblue"],
			  [norm(2./256.), "dodgerblue"]
			  ]
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
	return cmap


def get_dimensions():
	lon_list = np.arange(-180,180.1,0.5)
	lat_list = np.arange(-90,90.1,0.5)
	XX,YY = np.meshgrid(lon_list,lat_list)
	coord_list = [geopy.Point(x) for x in zip(YY.flatten(),XX.flatten())]	
	return (XX,YY,coord_list)

def get_current_state_dist():
	try:
		dist_array = np.load(file_handler.out_file('current_state.npy'))
	except FileNotFoundError:	
		float_pos_list = Core.recent_pos_list(ArgoReader)
		XX,YY,coord_list = get_dimensions()
		dist_list = []
		for n,coord in enumerate(coord_list):
			print(n)
			dummy_dist_list = [geopy.distance.GreatCircleDistance(coord,x).km for x in float_pos_list]
			dist_list.append(min(dummy_dist_list))
		dist_array = np.array(dist_list).reshape(XX.shape)
		dist_array[dist_array>3*4*degree_dist] = 3*4*degree_dist
		dist_array = np.log2(dist_array/(3*degree_dist))
		np.save(file_handler.out_file('current_state'),dist_array)
	return dist_array

def get_future_state_dist(trans_mat,time_step):
	try:
		dist_array = np.load(file_handler.out_file('future_state_'+str(time_step)+'.npy'))
	except FileNotFoundError:
		trans_holder = trans_mat.multiply(time_step,value=0.00001)
		float_mat = Core.recent_floats(trans_mat.trans_geo, ArgoReader,days_delta=(90*time_step))
		XX,YY,coord_list = get_dimensions()
		obs_out = scipy.sparse.csc_matrix(trans_holder).dot(float_mat.get_sensor('so'))

		float_pos_list = []
		for idx in scipy.sparse.find(float_mat.get_sensor('so'))[0]:
			point = list(trans_mat.trans_geo.total_list)[idx]
			ew_holder = ew[time_step,idx]
			ns_holder = ns[time_step,idx]
			lon = point.longitude + ew_holder
			lat = point.latitude + ns_holder
			float_pos_list.append(geopy.Point(lat,lon))
		dist_list = []
		for n,coord in enumerate(coord_list):
			print(n)
			dummy_dist_list = [geopy.distance.GreatCircleDistance(coord,x).km for x in float_pos_list]
			dist_list.append(min(dummy_dist_list))
		dist_array = np.array(dist_list).reshape(XX.shape)
		dist_array[dist_array>3*4*degree_dist] = 3*4*degree_dist
		dist_array = np.log2(dist_array/(3*degree_dist))
		np.save(file_handler.out_file('future_state_'+str(time_step)+'.npy'),dist_array)
	return dist_array


current_dist_array = get_current_state_dist()
one_year_dist_array = get_future_state_dist(trans_mat,4)
two_year_dist_array = get_future_state_dist(trans_mat,8)

cmap = get_cmap()
XX,YY,coord_list = get_dimensions()

fig = plt.figure(figsize=(18,14))
ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
dummy,dummy,ax1 = trans_mat.trans_geo.plot_setup(ax=ax1)
pcm = ax1.pcolormesh(XX,YY,-current_dist_array,cmap=cmap,vmin=-2,vmax=2)
ax1.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
dummy,dummy,ax2 = trans_mat.trans_geo.plot_setup(ax=ax2)
ax2.pcolormesh(XX,YY,-one_year_dist_array,cmap=cmap,vmin=-2,vmax=2)
ax2.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

cbar = fig.colorbar(pcm,ax=[ax1,ax2],pad=.05,label='Nominal Coverage',location='right')
cbar.set_ticks([-2,-1,0,1,2])
cbar.set_ticklabels(['0.25X','0.5X','1X','2X','4X'])

plt.savefig(plot_handler.out_file('argo_map'))
plt.close()

PCM = ax.get_children()[3]
plt.colorbar(PCM,location = 'bottom',label = 'Distance to Nearest Float (km)')
plt.savefig(file_handler.out_file('argo_map'))
plt.close()