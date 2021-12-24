import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
from GeneralUtilities.Compute.list import VariableList
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list,full_argo_list
from OptimalArray.Utilities.target_load import InverseInstance,InverseGeo
from TransitionMatrix.Utilities.ArgoData import Core,BGC
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from TransitionMatrix.Utilities.TransMat import TransMat
import cartopy.crs as ccrs
import scipy
from TransitionMatrix.Utilities.Utilities import colorline,get_cmap,shiftgrid
from GeneralUtilities.Plot.Cartopy.regional_plot import SouthernOceanCartopy,PacificCartopy,AtlanticCartopy

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

cmap = get_cmap()	
full_argo_list()
trans_mat = TransMat.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)

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

total_obs = [trans_mat.multiply(x,value=0.00001) for x in [4,8]]
float_list = [Core.recent_floats(trans_mat.trans_geo, ArgoReader,days_delta=(90*x)) for x in [4,8]]

var = 'so'
i = 1
ax_list = []
fig = plt.figure(figsize=(22.5,14))
trans_mat.trans_geo.plot_class = PacificCartopy
for trans,float_mat,label,k in zip(total_obs,float_list,['a','b'],[4,8]):

	ax = fig.add_subplot(2,1,i, projection=ccrs.PlateCarree(central_longitude=180))
	ax_list.append(ax)
	obs_out = scipy.sparse.csc_matrix(trans).dot(float_mat.get_sensor(var))
	plottable = trans_mat.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
	ax.annotate(label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)
	lats = YY[:,0]
	lons = XX[0,:]
	plottable, lons = shiftgrid(0, plottable, lons, start=True)
	# XX,YY = np.meshgrid(lons,lats)

	pcm = ax.pcolor(XX,YY,plottable,cmap=cmap)
	for idx in scipy.sparse.find(float_mat.get_sensor(var))[0]:
		point = list(trans_mat.trans_geo.total_list)[idx]
		ew_holder = ew[:k,idx]
		ns_holder = ns[:k,idx]
		lons = [point.longitude + x for x in ew_holder]
		lats = [point.latitude + x for x in ns_holder]
		lc = colorline(lons,lats,ax,z = [90*x for x in range(k)],norm=plt.Normalize(0.0, 720.0))
	i += 1
fig.colorbar(lc,ax=ax_list,pad=.05,label='Days Since Deployment',location='left')
fig.colorbar(pcm,ax=ax_list,pad=.05,label='Chance of Float (%)',location='right')
plt.savefig(file_handler.out_file('figure_10'))
plt.close()



i = 1
ax_list = []
fig = plt.figure(figsize=(22.5,14))
trans_mat.trans_geo.plot_class = AtlanticCartopy
for trans,float_mat,label,k in zip(total_obs,float_list,['a','b'],[4,8]):

	ax = fig.add_subplot(1,2,i, projection=ccrs.PlateCarree())
	ax_list.append(ax)
	obs_out = scipy.sparse.csc_matrix(trans).dot(float_mat.get_sensor(var))
	plottable = trans_mat.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
	ax.annotate(label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)

	pcm = ax.pcolor(XX,YY,plottable,cmap=cmap)
	for idx in scipy.sparse.find(float_mat.get_sensor(var))[0]:
		point = list(trans_mat.trans_geo.total_list)[idx]
		ew_holder = ew[:k,idx]
		ns_holder = ns[:k,idx]
		lons = [point.longitude + x for x in ew_holder]
		lats = [point.latitude + x for x in ns_holder]
		lc = colorline(lons,lats,ax,z = [90*x for x in range(k)],norm=plt.Normalize(0.0, 720.0))
	i += 1
fig.colorbar(lc,ax=ax_list,pad=.05,label='Days Since Deployment',location='top')
fig.colorbar(pcm,ax=ax_list,pad=.05,label='Chance of Float (%)',location='bottom')
plt.savefig(file_handler.out_file('figure_11'))
plt.close()


i = 1
ax_list = []
fig = plt.figure(figsize=(44,14))
trans_mat.trans_geo.plot_class = SouthernOceanCartopy
for trans,float_mat,label,k in zip(total_obs,float_list,['a','b'],[4,8]):

	ax = fig.add_subplot(1,2,i, projection=ccrs.SouthPolarStereo())
	i += 1
	ax_list.append(ax)
	obs_out = scipy.sparse.csc_matrix(trans).dot(float_mat.get_sensor(var))
	plottable = trans_mat.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
	plottable = np.hstack([plottable,plottable[:,-1].reshape(91,1)])
	plottable = np.ma.masked_equal(plottable,0)
	ax.annotate(label, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)
	lats = YY[:,0]
	lons = XX[0,:]
	lons = lons.tolist()+[180]
	XX,YY = np.meshgrid(lons,lats)
	pcm = ax.pcolor(XX,YY,plottable,cmap=cmap,transform=ccrs.PlateCarree())
	for idx in scipy.sparse.find(float_mat.get_sensor(var))[0]:
		point = list(trans_mat.trans_geo.total_list)[idx]
		ew_holder = ew[:k,idx]
		ns_holder = ns[:k,idx]
		lons = [point.longitude + x for x in ew_holder]
		lats = [point.latitude + x for x in ns_holder]
		lc = colorline(lons,lats,ax,z = [90*x for x in range(k)],norm=plt.Normalize(0.0, 720.0))
fig.colorbar(lc,ax=ax_list,pad=.07,label='Days Since Deployment',location='left')
fig.colorbar(pcm,ax=ax_list,pad=.05,label='Chance of Float (%)',location='right')
plt.savefig(file_handler.out_file('figure_12'))
plt.close()