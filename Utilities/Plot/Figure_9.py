from TransitionMatrix.Utilities.TransMat import TransMat,TransitionGeo
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from TransitionMatrix.Utilities.TransPlot import TransPlot
from GeneralUtilities.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
from TransitionMatrix.Utilities.Plot.argo_data import Core,BGC
from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo,InverseInstance,HInstance,InverseCCS,InverseGOM,InverseSOSE
import scipy
import numpy as np 


plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

trans_mat = TransPlot.load_from_type(lon_spacing=lon,lat_spacing=lat,time_step=time)
XX,YY,ax = GlobalCartopy().get_map()
ew_data_list = []
ns_data_list = []
for k in range(10):
	holder = trans_mat.multiply(k)
	east_west, north_south = holder.return_mean()
	ew_data_list.append(east_west)
	ns_data_list.append(north_south)
ew = np.vstack(ew_data_list)
ns = np.vstack(ns_data_list)
for k,point in enumerate(trans_mat.trans_geo.total_list):
	ew_holder = ew[:,k]
	ns_holder = ns[:,k]
	lons = [point.longitude + x for x in ew_holder]
	lats = [point.latitude + x for x in ns_holder]
	ax.plot(lons,lats)