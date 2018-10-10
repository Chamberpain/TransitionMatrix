import pandas as pd
import oceans
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import numpy as np

df = pd.read_pickle('global_argo_traj')

A17_loc = [(-30.3200,-39.1930)
,(-32.104 ,-40.76)
,(-33.3090,  -41.8050)
,(-34.489 ,-42.841)
,(-35.1160,  -43.3510)
,(-37.487 ,-45.462)
,(-40.4850,  -48.0970)
,(-41.729 ,-49.091)
,(-42.9070,  -50.1980)
,(-44.705 ,-51.765)
,(-45.905 ,-52.812)
,(-47.1050,  -53.8640)]

AMT_28_loc = [(-27.0, -25.0),
(-30.0, -25.0),
(-33.0, -28.8),
(-36.0, -32.6),
(-39.0, -36.4),
(-42.0, -40.2),
(-45.0, -44.0),
(-48.0, -48.0)]

degree_size = 0.5

def dataframe_plot(df_):
	lon_list = df_.Lon.values
	lat_list = df_.Lat.values
	m.plot(lon_list,lat_list,latlon=True)

data_list = [(A17_loc,'A17 '),(AMT_28_loc,'AMT_28 ')]
for dlist,name in data_list:
	for loc in dlist:

		lat,lon = loc

		plt.figure()
		plt.title(name+str(lat)+' , '+str(lon))
		lllat=-55
		urlat=-20
		lllon=-60
		urlon=10
		m = Basemap(projection='cyl',llcrnrlat=lllat,urcrnrlat=urlat,llcrnrlon=lllon,urcrnrlon=urlon,resolution='c',lon_0=0,fix_aspect=False)
		m.drawcoastlines(linewidth=1.5)
		line_space=10
		fontsz=10
		parallels = np.arange(-90,0,float(line_space)/2)
		m.drawparallels(parallels,labels=[1,0,0,0],fontsize=fontsz)
		meridians = np.arange(-360.,360.,float(line_space))
		m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fontsz)

		m.plot(lon,lat,marker='s',color='gold',markersize=10,zorder=10,markeredgecolor='black',markeredgewidth=2,latlon=True)
		lat_min = lat- degree_size
		lat_max = lat+ degree_size
		lon_min = oceans.wrap_lon180(lon-degree_size)[0]
		lon_max = oceans.wrap_lon180(lon+degree_size)[0]
		df_cut = df[((df.Lon<lon_max)&(df.Lon>lon_min))&((df.Lat<lat_max)&(df.Lat>lat_min))]
		cruise_date_df = []
		for cruise in df_cut.Cruise.unique():
			date = df_cut[df_cut.Cruise==cruise].Date.min()
			token = df[(df.Date>date)&(df.Cruise==cruise)]
			if len(token)<=2:
				continue
			else:
				dataframe_plot(token)
		plt.savefig('./plots_for_lynne/'+name[:-1]+'_'+str(lat)+' , '+str(lon)+'.png')
		plt.close()