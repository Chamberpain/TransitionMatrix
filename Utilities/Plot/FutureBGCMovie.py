import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors	
from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list,full_argo_list
from TransitionMatrix.Utilities.Inversion.target_load import InverseInstance,InverseGeo
from TransitionMatrix.Utilities.Plot.argo_data import BGC
from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')


def bgc_pdf():

	
	norm = matplotlib.colors.Normalize(0,400)
	colors = [[norm(0), "yellow"],
	          [norm(10), "lightgoldenrodyellow"],
	          [norm(30), "lightyellow"],
	          [norm(50), "powderblue"],
			  [norm(75), "skyblue"],
	          [norm(100), "deepskyblue"],
	          [norm(400), "dodgerblue"],
	          ]
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
	full_argo_list()
	inverse_mat = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=2)
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

	trans_mat.trans_geo.variable_list = inverse_mat.trans_geo.variable_list
	trans_mat.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	lats = trans_mat.trans_geo.get_lat_bins()
	lons = trans_mat.trans_geo.get_lon_bins()
	float_mat = BGC.recent_floats(trans_mat.trans_geo, BGCReader)
	total_obs = [trans_mat.multiply(x) for x in range(10)]

	plot_dict = {'so':'a','ph':'d','chl':'c','o2':'b'}

	for i,trans in enumerate(total_obs):
		for k,var in enumerate(['so','o2','chl','ph']):
			fig = plt.figure(figsize=(21,14))
			ax = fig.add_subplot(projection=ccrs.PlateCarree())
			obs_out = scipy.sparse.csc_matrix(trans).dot(float_mat.get_sensor(var))
			plottable = trans_mat.trans_geo.transition_vector_to_plottable(obs_out.todense())*100
			plottable[plottable<0.1]=0.1
			plottable = np.ma.masked_less(plottable,1)
			ax.annotate(var, xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
			XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)
			pcm = ax.pcolor(XX,YY,plottable,vmin=0.1,vmax=400,cmap='Greys',norm=matplotlib.colors.LogNorm(vmin=0.1, vmax=400))
			for idx in scipy.sparse.find(float_mat.get_sensor(var))[0]:
				point = list(trans_mat.trans_geo.total_list)[idx]
				ew_holder = ew[:i,idx]
				ns_holder = ns[:i,idx]
				lons = [point.longitude + x for x in ew_holder]
				lats = [point.latitude + x for x in ns_holder]
				ax.plot(lons,lats,'r',linewidth=2)
			fig.colorbar(pcm,orientation="horizontal",shrink=0.9,label='Chance of Observation in Cell')
			plt.title(str(90*i)+' Day Timestep')
			plt.savefig(file_handler.tmp_file(var+'/timestep_'+str(i)))
			plt.close()
	for k,var in enumerate(['so','o2','chl','ph']):
		os.chdir(file_handler.tmp_file(var+'/'))
		os.system("ffmpeg -r 11/12 -i timestep_%01d.png -vcodec mpeg4 -y movie.mp4")
