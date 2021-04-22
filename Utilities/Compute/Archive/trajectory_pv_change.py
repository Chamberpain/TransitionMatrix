import os
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import matplotlib.colors as colors
import scipy.io
import numpy as np
#input file names
grid_file_name = os.getenv("HOME")+'/iCloud/Data/Raw/SOSE/grid.mat'
sose_df_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose_particle_df.pickle'
traj_df_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/all_argo_traj.pickle'

#output file names
sose_pv_map_output_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose_pv_map.npy'
traj_pv_map_output_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/traj_pv_map.npy'
gps_pv_map_output_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/gps_pv_map.npy'
argos_pv_map_output_file_name = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/argos_pv_map.npy'

file_path_dict = {'TRAJ':traj_pv_map_output_file_name,'ARGOS':argos_pv_map_output_file_name,
	'GPS':gps_pv_map_output_file_name,'SOSE':sose_pv_map_output_file_name}

mat = scipy.io.loadmat(grid_file_name)
XC = mat['XC'][:,0]
YC = mat['YC'][0,:]
Depth = mat['Depth']


def recompile_map_files(type_):
	if type_=='TRAJ':
		type_list = ['GPS','ARGOS']
	else:
		type_list = [type_]
	dataframe = pd.concat([pd.read_pickle(sose_df_file_name),pd.read_pickle(traj_df_file_name)])
	dataframe = dataframe[dataframe['position type'].isin(type_list)]
	dataframe['bins_lat'] = pd.cut(dataframe.Lat,bins=YC,labels=YC[:-1])
	dataframe['bins_lon'] = pd.cut(dataframe.Lon,bins=XC,labels=XC[:-1])
	for i,lon in enumerate(XC[:-1]):
		print 'lon = ',lon
		for j,lat in enumerate(YC[:-1]):
			dataframe.loc[(dataframe.bins_lat==lat)&(dataframe.bins_lon==lon),'Depth']=Depth[i,j]

	dataframe['PV']=np.sin(np.deg2rad(dataframe.Lat))*omega*2/dataframe.Depth
	frames = []
	for cruise in dataframe.Cruise.unique():
		print 'we are on cruise ',cruise
		df_holder = dataframe[dataframe.Cruise==cruise]
		df_holder['PV Diff'] = df_holder.PV.diff()
		frames.append(df_holder)
	dataframe = pd.concat(frames)
	dataframe = df.dropna()
	pv_diff_map = np.zeros([len(XC[:-1]),len(YC[:-1])])
	for i,lon in enumerate(XC[:-1]):
		print 'lon = ',lon
		for j,lat in enumerate(YC[:-1]):
			pv_diff_map[i,j] = dataframe.loc[(dataframe.bins_lat==lat)&(dataframe.bins_lon==lon),'PV Diff'].mean()
	np.save(file_path_dict[type_],pv_diff_map)
	return pv_diff_map

def pv_difference_map(type_):
	try:
		pv_diff_map = np.load(file_path_dict[type_])
	except IOError:
		print type_+' could not be loaded and will need to be recompiled'
		pv_diff_map = recompile_map_files(type_)

	pv_diff_map = np.ma.masked_array(pv_diff_map,mask=Depth[:-1,:-1]<1000)
	pv_diff_map = np.ma.masked_invalid(pv_diff_map)

	plot_depth = np.ma.masked_array(Depth,mask=Depth<1000).T

	plt.figure()
	plt.subplot(2,2,1)
	m = Basemap(projection='spstere',boundinglat=-50,lon_0=90,resolution='l')
	m.drawcoastlines()
	m.fillcontinents(color='coral',lake_color='aqua')
	m.drawparallels(np.arange(-80.,81.,20.))
	m.drawmeridians(np.arange(-180.,181.,20.))
	# m.drawmapboundary(fill_color='aqua')
	XX,YY = np.meshgrid(XC[:-1],YC[:-1])
	m.pcolormesh(XX,YY,pv_diff_map.T,norm=colors.SymLogNorm(linthresh=10**(-12), 
		linscale=10**(-12),vmin=np.min(pv_diff_map), vmax=np.max(pv_diff_map)),
		latlon=True)
	plt.title('$\Delta$ PV from '+type_)
	# plt.colorbar()
	plt.subplot(2,2,2)
	m = Basemap(projection='spstere',boundinglat=-50,lon_0=90,resolution='l')
	m.drawcoastlines()
	m.fillcontinents(color='coral',lake_color='aqua')
	m.drawparallels(np.arange(-80.,81.,20.))
	m.drawmeridians(np.arange(-180.,181.,20.))
	# m.drawmapboundary(fill_color='aqua')
	XX,YY = np.meshgrid(XC,YC)
	m.pcolormesh(XX,YY,plot_depth,latlon=True)
	plt.title('SOSE Bathymetry')
	# plt.colorbar()

	plt.subplot(2,2,4)
	m = Basemap(projection='spstere',boundinglat=-50,lon_0=90,resolution='l')
	m.drawcoastlines()
	m.fillcontinents(color='coral',lake_color='aqua')
	m.drawparallels(np.arange(-80.,81.,20.))
	m.drawmeridians(np.arange(-180.,181.,20.))
	# m.drawmapboundary(fill_color='aqua')
	A,B = np.gradient(plot_depth,YC,XC)
	grad_plot_depth = A+B
	XX,YY = np.meshgrid(XC,YC)
	m.pcolormesh(XX,YY,grad_plot_depth,norm=colors.SymLogNorm(linthresh=10**(-3) 
		,linscale=10**(-3),vmin=grad_plot_depth.min()*300, vmax=grad_plot_depth.max()*300)
		,latlon=True)
	plt.title(r'$\nabla \circ$'+'SOSE Bathymetry')
	# plt.colorbar()

	plt.subplot(2,2,3)
	m = Basemap(projection='spstere',boundinglat=-50,lon_0=90,resolution='l')
	m.drawcoastlines()
	m.fillcontinents(color='coral',lake_color='aqua')
	m.drawparallels(np.arange(-80.,81.,20.))
	m.drawmeridians(np.arange(-180.,181.,20.))
	# m.drawmapboundary(fill_color='aqua')
	grad_plot_depth = np.gradient(plot_depth,axis=1)
	XX,YY = np.meshgrid(XC,YC)
	m.pcolormesh(XX,YY,grad_plot_depth,norm=colors.SymLogNorm(linthresh=10**(-3) 
		,linscale=10**(-3),vmin=grad_plot_depth.min()*300, vmax=grad_plot_depth.max()*300)
		,latlon=True)
	plt.title('$d/dx$ SOSE Bathymetry')
	# plt.colorbar()

pv_difference_map('SOSE')
pv_difference_map('GPS')
pv_difference_map('ARGOS')
pv_difference_map('TRAJ')
plt.show()