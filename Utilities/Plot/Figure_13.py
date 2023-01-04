from TransitionMatrix.Utilities.ArgoData import BGC as SOCCOM
from TransitionMatrix.Utilities.ArgoData import Core as Argo
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from GeneralUtilities.Plot.Cartopy.regional_plot import SOSECartopy
from GeneralUtilities.Data.Lagrangian.Argo.argo_read import ArgoReader
from GeneralUtilities.Data.Lagrangian.Argo.array_class import ArgoArray
import datetime
import numpy as np 
import scipy
from TransitionMatrix.Utilities.TransMat import TransMat
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR

file_handler = FilePathHandler(ROOT_DIR,'final_figures')
argo_array = ArgoArray.compile()

def recent_floats(cls,GeoClass, FloatClass):
	out_list = []
	for variable in GeoClass.variable_list:
		float_var = GeoClass.variable_translation_dict[variable]

		variable,lat_bins,lon_bins,float_type = (float_var,GeoClass.get_lat_bins(),GeoClass.get_lon_bins(),cls.traj_file_type)
		date_list = argo_array.get_recent_date_list()
		bin_list = argo_array.get_recent_bins(lat_bins,lon_bins)
		sensor_list = argo_array.get_sensors()
		sensor_mask = [variable in x for x in sensor_list]
		date_mask =[max(date_list)-datetime.timedelta(days=180)<x for x in date_list]
		if float_type == 'BGC':
			soccom_mask = ['SOCCOM' in x.meta.project_name for dummy,x in argo_array.items()]
			mask = np.array(sensor_mask)&np.array(date_mask)&np.array(soccom_mask)
		else:
			mask = np.array(sensor_mask)&np.array(date_mask)

		age_list = [(x.prof.date[-1]-x.prof.date[0]).days/365. for x in argo_array.values()]

		var_grid,age_list = (np.array(bin_list)[mask],1/(np.ceil(age_list)+1)[mask])
		idx_list = [GeoClass.total_list.index(x) for x in var_grid if x in GeoClass.total_list]
		holder_array = np.zeros([len(GeoClass.total_list),1])
		for k,idx in enumerate(idx_list):
			holder_array[idx]+=age_list[k]
		out_list.append(holder_array)
	out = np.hstack([out_list[0]]).max(axis=1)
	out = out.reshape(out.shape[0],1)
	return cls(out,trans_geo=GeoClass)


traj_class_1 = TransMat.load_from_type(lat_spacing=4,lon_spacing=4,time_step=180)
traj_class_1.trans_geo.plot_class = SOSECartopy
traj_class_1.trans_geo.variable_list = ['ph','chl','o2']
traj_class_1.trans_geo.variable_translation_dict = {'thetao':'TEMP','so':'PSAL','ph':'PH_IN_SITU_TOTAL','chl':'CHLA','o2':'DOXY'}
float_vector_1 = recent_floats(SOCCOM,traj_class_1.trans_geo, argo_array)

traj_class_2 = TransMat.load_from_type(lat_spacing=2,lon_spacing=2,time_step=180)
traj_class_2.trans_geo.plot_class = GlobalCartopy
traj_class_2.trans_geo.variable_list = ['so']
traj_class_2.trans_geo.variable_translation_dict = {'thetao':'TEMP','so':'PSAL','ph':'PH_IN_SITU_TOTAL','chl':'CHLA','o2':'DOXY'}
float_vector_2 = recent_floats(Argo,traj_class_2.trans_geo, argo_array)

fig = plt.figure(figsize=(14,14))
ax1 = fig.add_subplot(2,1,1, projection=ccrs.SouthPolarStereo())
XX,YY,ax1 = traj_class_1.trans_geo.plot_setup(ax = ax1,adjustable=True)
plottable = traj_class_1.trans_geo.transition_vector_to_plottable(traj_class_1.todense().dot(float_vector_1.todense()))
traj_class_1.traj_file_type = 'SOSE'
ax1.pcolormesh(XX,YY,np.ma.masked_less(plottable,5*10**-3),cmap=plt.cm.YlGn,vmax=0.3,transform=ccrs.PlateCarree())
row_idx,column_idx,data = scipy.sparse.find(float_vector_1)

lats = [list(traj_class_1.trans_geo.total_list)[x].latitude for x in row_idx]
lons = [list(traj_class_1.trans_geo.total_list)[x].longitude for x in row_idx]

ax1.scatter(lons,lats,c='m',marker='*',transform=ccrs.PlateCarree())

plt.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
traj_class_2.traj_file_type = 'Argo'
XX,YY,ax2 = traj_class_2.trans_geo.plot_setup(ax = ax2)
plottable = traj_class_2.trans_geo.transition_vector_to_plottable(traj_class_2.todense().dot(float_vector_2.todense()))
ax2.pcolormesh(XX,YY,np.ma.masked_less(plottable,5*10**-3),cmap=plt.cm.YlGn,vmax = 0.3)
PCM = ax2.get_children()[3]
fig.colorbar(PCM,ax=ax1,location = 'left',fraction=0.10,label='Probability Density/Age')
row_idx,column_idx,data = scipy.sparse.find(float_vector_2)

lats = [list(traj_class_2.trans_geo.total_list)[x].latitude for x in row_idx]
lons = [list(traj_class_2.trans_geo.total_list)[x].longitude for x in row_idx]

ax2.scatter(lons,lats,c='r',s=4)
plt.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
plt.savefig(file_handler.out_file('figure_13'))
plt.close()