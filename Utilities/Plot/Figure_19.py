from GeneralUtilities.Filepath.search import find_files
import os
data_file_name = os.getenv("HOME")+'/Data/Raw/GOSHIP'
import pandas as pd
from TransitionMatrix.Utilities.TransMat import TransMat
import geopy
import numpy
import scipy.sparse
import matplotlib.pyplot as plt
from GeneralUtilities.Filepath.instance import FilePathHandler
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR

file_handler = FilePathHandler(ROOT_DIR,'final_figures')
time_dict = {'I05':1,'A12':1,'SR04':1,'A13.5':1,'P02':1,'MED01':1,'P04':1,'I09':2,'I08':2,'A16':2,'P16':3,'P15':3,'S04P':4,'ARC01':4,'P06':5}
color_map = plt. cm. get_cmap('viridis')

def read_goship(filename):
	try:
		frame = pd.read_csv(filename,comment='#',header=1)[['SECT_ID','LATITUDE','LONGITUDE']].dropna().drop_duplicates()
	except:
		frame = pd.read_csv(filename,header=1)[['EXPOCODE','LATITUDE','LONGITUDE']].dropna().drop_duplicates()
		frame['SECT_ID'] = 'I05'
		frame = frame[['SECT_ID','LATITUDE','LONGITUDE']]
	return frame

frames = find_files(data_file_name,'*hy1.csv',function=read_goship)
goship_df = pd.concat(frames)
trans_mat = TransMat.load_from_type(lat_spacing=3,lon_spacing=3,time_step=90)
lat_bins = trans_mat.trans_geo.get_lat_bins()
lon_bins = trans_mat.trans_geo.get_lon_bins()
point_list = [geopy.Point(lat_bins.find_nearest(x),lon_bins.find_nearest(y)) for x,y in zip(goship_df.LATITUDE.tolist(),goship_df.LONGITUDE.tolist())]
idx_list = []
for point in point_list:
	try:
		idx_list.append(trans_mat.trans_geo.total_list.index(point))
	except ValueError:
		idx_list.append(np.nan)
goship_df['idx_list'] = idx_list

for sect_holder in goship_df.SECT_ID.unique():
	sect = sect_holder.replace(' ','')
	try: 
		idx = sect.index('N')
		sect = sect[:idx]
	except ValueError:
		pass
	try: 
		idx = sect.index('S')
		if idx is 0:
			pass
		else:
			sect = sect[:idx]
	except ValueError:
		pass
	try:
		year = time_dict[sect]
		mask = goship_df.SECT_ID==sect_holder
		goship_df.loc[mask,'year']=year
	except:
		continue
goship_df = goship_df.dropna()
mat_list = [trans_mat.multiply(x) for x in range(20)]
total_obs_list = [np.sum(mat_list[:x]) for x in [4,8,12,16,19]]
moment_in_time_list = [mat_list[x] for x in [4,8,12,16,19]]
float_list = []
for year in goship_df.year.unique():
	holder = goship_df[goship_df.year==year]
	float_holder = np.zeros([len(trans_mat.trans_geo.total_list),1])
	for idx in holder.idx_list.tolist():
		float_holder[int(idx),0]=1
	float_list.append(scipy.sparse.csc_matrix(float_holder))

fig = plt.figure(figsize=(20,14))
ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
XX,YY,ax1 = trans_mat.trans_geo.plot_setup(ax=ax1)
total_obs = np.sum([scipy.sparse.csc_matrix(x).dot(y) for x,y in zip(total_obs_list,float_list)])
plottable = trans_mat.trans_geo.transition_vector_to_plottable(total_obs.todense())*100
plottable = np.ma.masked_equal(plottable,0)
ax1.pcolor(XX,YY,plottable,cmap=color_map.reversed())

ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
XX,YY,ax2 = trans_mat.trans_geo.plot_setup(ax=ax2)
moment_in_time = np.sum([scipy.sparse.csc_matrix(x).dot(y) for x,y in zip(moment_in_time_list,float_list)])
plottable = trans_mat.trans_geo.transition_vector_to_plottable(moment_in_time.todense())*100
plottable = np.ma.masked_equal(plottable,0)
ax2.pcolor(XX,YY,plottable,cmap=color_map.reversed())
PCM = ax2.get_children()[3]
fig.colorbar(PCM,ax=[ax1,ax2],fraction=0.10,label='Chance of Observation (%)')
ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=11,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
plt.savefig(file_handler.out_file('figure_20'))
plt.close()
