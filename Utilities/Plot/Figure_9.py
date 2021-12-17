from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.TransGeo import TransitionGeo
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import scipy
import numpy as np 
import matplotlib.collections as mcoll
import cartopy.crs as ccrs


plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

def colorline(
        x, y, z=None, cmap='plasma', norm=plt.Normalize(0.0, 900.0),
        linewidth=1.5, alpha=1.0):
    if z is None:
        z = np.linspace(0.0, 900.0, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)
    ax = plt.gca()
    ax.add_collection(lc)
    return lc

def make_segments(x, y):
    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

lon = 2
lat = 2
time_step = 90
trans_mat = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
ew_data_list = []
ns_data_list = []
for k in range(10):
	holder = trans_mat.multiply(k)
	east_west, north_south = holder.return_mean()
	ew_data_list.append(east_west)
	ns_data_list.append(north_south)
ew = np.vstack(ew_data_list)
ns = np.vstack(ns_data_list)

fig = plt.figure(figsize=(14,14))
ax = fig.add_subplot(1,1,1, projection=ccrs.PlateCarree())
XX,YY,ax = trans_mat.trans_geo.plot_setup(ax=ax)
for k,point in enumerate(trans_mat.trans_geo.total_list):
	ew_holder = ew[:,k]
	ns_holder = ns[:,k]
	lons = [point.longitude + x for x in ew_holder]
	lats = [point.latitude + x for x in ns_holder]
	lc = colorline(lons,lats)
plt.colorbar(lc,label='Days Since Deployment')
plt.savefig(file_handler.out_file('figure_9'))
plt.close()