import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.basemap import Basemap
import numpy as np 
import matplotlib.colors as colors


# this was written in python 3 because of the way that argo_traj_box saves the trajectory data
def map_setup():
    print('I am plotting Crete')
    m = Basemap(projection='cea',llcrnrlon=20.,llcrnrlat=30,urcrnrlon=30,urcrnrlat=40,fix_aspect=False,resolution='h')
    m.drawcoastlines()
    XX,YY = m(X,Y)
    meridians = np.arange(18,29.,4)
    parallels = np.arange(28,42,4)
    m.drawparallels(parallels,labels=[True,False,False,False],fontsize=18)
    m.drawmeridians(meridians,labels=[True,False,False,True],fontsize=18)
    return (m,XX,YY)        

file_ = '/Users/pchamberlain/Projects/argo_traj_box/traj_df.pickle'
df = pd.read_pickle(file_)
df = df[(df.latitude>30)&(df.latitude<40)&(df.longitude>20)&(df.longitude<30)]


for spacing in [.25,.5,1]:
    plt.figure()
    bins_lat = np.arange(28,42,spacing).tolist()
    bins_lon = np.arange(18,32,spacing).tolist()
    df['bins_lat'] = pd.cut(df.latitude,bins = bins_lat,labels=bins_lat[:-1])
    df['bins_lon'] = pd.cut(df.longitude,bins = bins_lon,labels=bins_lon[:-1])
    df['bin_index'] = [_ for _ in zip(df['bins_lat'].values,df['bins_lon'].values)]
    X,Y = np.meshgrid(bins_lon,bins_lat)
    m,XX,YY = map_setup()
    ZZ = np.zeros([len(bins_lat),len(bins_lon)])
    series = df.groupby('bin_index').count()['Cruise']
    for item in series.iteritems():
        tup,n = item
        ii_index = bins_lon.index(tup[1])
        qq_index = bins_lat.index(tup[0])
        ZZ[qq_index,ii_index] = n           
    ZZ = np.ma.masked_equal(ZZ,0)
    m.pcolor(XX,YY,ZZ,vmax=700,norm=colors.LogNorm())
    plt.colorbar(label='Profile Density')
    plt.title('Profile Density for '+str(spacing)+' Degree Grid',fontsize=22)
    plt.savefig('./plots_for_matt/'+str(spacing)+'.png')
    plt.figure('Histogram')
    plt.hist(ZZ.flatten(),alpha=0.3,label=str(spacing)+'$^\circ$ Grid')
plt.yscale('log')
plt.xlabel('Profile Density')
plt.ylabel('Number of Occurrences')
plt.legend()
plt.savefig('./plots_for_matt/hist.png')


plt.figure()
x,y = zip(*df[['longitude','latitude']].values) 
m,XX,YY = map_setup()
m.scatter(x,y,s=0.2,latlon=True)

for cruise in df.Cruise.unique():
    x,y = zip(*df[df.Cruise==cruise][['longitude','latitude']].head(1).values)
    m.scatter(x,y,c='r',s=2,latlon=True)
plt.savefig('./plots_for_matt/raw_profiles.pdf', bbox_inches='tight')


plt.figure()
m,XX,YY = map_setup()
for cruise in df.Cruise.unique():
    x,y = zip(*df[df.Cruise==cruise][['longitude','latitude']].values)
    m.plot(x,y,latlon=True)
plt.title('Raw Cruise Tracks',fontsize=22)
plt.savefig('./plots_for_matt/raw_cruise_tracks.png')
