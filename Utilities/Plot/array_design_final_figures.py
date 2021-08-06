from TransitionMatrix.Utilities.Plot.argo_data import BGC
from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo,InverseInstance
from TransitionMatrix.Utilities.Compute.trans_read import TransMat
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader
from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list
import numpy as np
import scipy.sparse
import cartopy.crs as ccrs
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'array_design_final_figures')


def figure_1():
	full_argo_list()
	inverse_mat = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=2)
	trans_mat = TransMat.load_from_type(lat_spacing=2,lon_spacing=2,time_step=90)
	trans_mat.trans_geo.variable_list = inverse_mat.trans_geo.variable_list
	trans_mat.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	lats,lons = trans_mat.trans_geo.lats_lons()
	float_mat = BGC.recent_floats(trans_mat.trans_geo, BGCReader)
	total_obs = np.sum([trans_mat.multiply(x) for x in range(4)])


	plot_dict = {'so':'T/S','ph':'pH','chl':'Chlorophyl','o2':'Oxygen'}
	fig = plt.figure(figsize=(16,9))
	idx_list = [1,2,3,4,3]
	col_list = [2,2,2,2,1]
	
	ax_list = [fig.add_subplot(2,2,(k+1), projection=ccrs.PlateCarree()) for k in range(4)]
	for k,var in enumerate(trans_mat.trans_geo.variable_list[1:]):
		obs_out = scipy.sparse.csc_matrix(total_obs).dot(float_mat.get_sensor(var))
		plottable = total_obs.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
		ax = ax_list[k]
		XX,YY,ax = total_obs.trans_geo.plot_setup(ax=ax)
		ax.pcolor(XX,YY,plottable,vmin=0,vmax=100)
		ax.set_title(plot_dict[var])
	# plt.subplots_adjust(hspace=1.5)
	PCM = ax.get_children()[0]
	fig.colorbar(PCM,ax=ax_list,label='Observation in Next Year (%)',shrink=0.6)
	plt.savefig(file_handler.out_file('chance_of_observation'))
	plt.close()


BGCReader


float_mat = BGC.