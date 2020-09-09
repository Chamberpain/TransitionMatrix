lllon = -80
urlon = -40	
lllat = -65
urlat=-50

def dataframe_plot(df_,m):
	lon_list = df_.longitude.values
	lat_list = df_.latitude.values
	m.plot(lon_list,lat_list,latlon=True)

plt.figure(figsize=(10,10))
file_ = 'traj_df.pickle'
dataframe = pd.read_pickle(file_)
m = Basemap(projection='cea',llcrnrlat=lllat,urcrnrlat=urlat,llcrnrlon=lllon,urcrnrlon=urlon,resolution='h',lon_0=0,fix_aspect=False)
m.drawcoastlines(linewidth=1.5)
line_space=20
fontsz=10
parallels = np.arange(-90,0,float(line_space)/2)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=fontsz)
meridians = np.arange(-360.,360.,float(line_space))
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=fontsz)
m.fillcontinents(color='dimgray',lake_color='darkgray')
df_cut = dataframe[((dataframe.longitude<urlon)&(dataframe.longitude>lllon))&((dataframe.latitude<urlat)&(dataframe.latitude>lllat))]
for cruise in df_cut.Cruise.unique():
	token = dataframe[(dataframe.Cruise==cruise)]
	cutoff = df_cut[df_cut.Cruise==cruise].index.min()
	token = token[token.index>=cutoff]
	if len(token)<=2:
		continue
	else:
		dataframe_plot(token,m)
plt.savefig('argo_traj_map_drake_passage')