import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors
from GeneralUtilities.Compute.list import VariableList
from GeneralUtilities.Data.Lagrangian.Argo.array_class import ArgoArray
from TransitionMatrix.Utilities.ArgoData import Core,BGC
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from TransitionMatrix.Utilities.TransMat import TransMat
import cartopy.crs as ccrs
import scipy
from TransitionMatrix.Utilities.Utilities import colorline,get_cmap


plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

color_map = plt. cm. get_cmap('viridis')
argo_array = ArgoArray.compile()
trans_mat = TransMat.load_from_type(lat_spacing=3,lon_spacing=3,time_step=90)
trans_mat.trans_geo.variable_list = VariableList(['thetao','so','ph','chl','o2'])
trans_mat.trans_geo.variable_translation_dict = {'thetao':'TEMP','so':'PSAL','ph':'PH_IN_SITU_TOTAL','chl':'CHLA','o2':'DOXY'}
float_mat = BGC.recent_floats(trans_mat.trans_geo, argo_array)
total_obs_all = np.sum([trans_mat.multiply(x) for x in range(4)])
total_obs_min = [trans_mat.multiply(x) for x in range(4)]
for filename,var in zip(['figure_15','test','test','test'],['so','o2','chl','ph']):
	fig = plt.figure(figsize=(18,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())

	obs_out = scipy.sparse.csc_matrix(total_obs_all).dot(float_mat.get_sensor(var))
	plottable = total_obs_all.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
	print(var)
	print(np.mean(plottable))
	plottable = np.ma.masked_less(plottable,1)
	XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax1)
	ax.pcolor(XX,YY,plottable,vmin=0,vmax=100,cmap=color_map.reversed())
	ax.annotate('a', xy = (0.17,0.8),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())

	obs_out = [scipy.sparse.csc_matrix(x).dot(float_mat.get_sensor(var)) for x in total_obs_min]
	for k in range(len(obs_out)):
		obs_out[k][obs_out[k]>1]=1
	first_min = np.array(obs_out[0].todense())*np.array(obs_out[1].todense())
	second_min = np.array(obs_out[2].todense())*np.array(obs_out[3].todense())
	minimum = (first_min*second_min)	
	print(var)
	print(np.mean(minimum))		
	plottable = trans_mat.trans_geo.transition_vector_to_plottable(minimum)*100

	plottable = np.ma.masked_less(plottable,1)
	plottable[plottable>100]=100
	XX,YY,ax = total_obs_all.trans_geo.plot_setup(ax=ax2)
	ax.pcolor(XX,YY,plottable,cmap=color_map.reversed(),norm=matplotlib.colors.LogNorm())
	ax.annotate('b', xy = (0.17,0.8),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	PCM = ax.get_children()[3]
	plt.subplots_adjust(left=0.06)
	fig.colorbar(PCM,ax=[ax1,ax2],label='Observation in Next Year (%)',fraction=0.10,)
	plt.savefig(file_handler.out_file(filename))
	plt.close()

fig = plt.figure(figsize=(14,14))
ax_list = [fig.add_subplot(3,1,(k+1), projection=ccrs.PlateCarree()) for k in range(3)]
for k,(var,letter) in enumerate(zip(['o2','chl','ph'],['a','b','c'])):
	ax = ax_list[k]
	obs_out = scipy.sparse.csc_matrix(total_obs_all).dot(float_mat.get_sensor(var))
	plottable = total_obs_all.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
	print(var)
	print(np.mean(plottable))
	plottable = np.ma.masked_less(plottable,1)
	plottable[plottable>100]=100
	XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)
	ax.pcolor(XX,YY,plottable,cmap=color_map.reversed(),norm=matplotlib.colors.LogNorm())
	ax.annotate(letter, xy = (0.17,0.8),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
PCM = ax_list[0].get_children()[3]
fig.colorbar(PCM,ax=ax_list,label='Observation in Next Year (%)',fraction=0.10)
plt.savefig(file_handler.out_file('figure_16'), bbox_inches='tight')
plt.close()