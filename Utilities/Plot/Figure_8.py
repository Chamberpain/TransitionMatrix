from matplotlib import patches
import shapely.geometry as sgeom
import numpy as np 
from TransitionMatrix.Utilities.TransMat import TransMat
from TransitionMatrix.Utilities.TransGeo import TransitionGeo
import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Data.Filepath.instance import FilePathHandler
import cartopy.crs as ccrs
from pyproj import Geod
import scipy.sparse

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'final_figures')

def quiver_plot(self):
	east_west_data, north_south_data = self.return_mean()
	east_west_data = east_west_data*self.trans_geo.lon_sep*111/(self.trans_geo.time_step*24)
	north_south_data = north_south_data*self.trans_geo.lat_sep*111/(self.trans_geo.time_step*24)
	quiver_e_w = self.trans_geo.transition_vector_to_plottable(east_west_data)
	quiver_n_s = self.trans_geo.transition_vector_to_plottable(north_south_data)


	std_number = 3.

	quiver_e_w = np.ma.masked_greater(quiver_e_w,np.mean(east_west_data)+std_number*np.std(east_west_data))
	quiver_e_w = np.ma.masked_less(quiver_e_w,np.mean(east_west_data)-std_number*np.std(east_west_data))

	quiver_n_s = np.ma.masked_greater(quiver_n_s,np.mean(north_south_data)+std_number*np.std(north_south_data))
	quiver_n_s = np.ma.masked_less(quiver_n_s,np.mean(north_south_data)-std_number*np.std(north_south_data))

	sf = 3

	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	geod = Geod(ellps='WGS84')
	XX,YY,ax1 = self.trans_geo.plot_setup(ax = ax1)
	q = ax1.quiver(XX[::sf,::sf],YY[::sf,::sf],quiver_e_w[::sf,::sf],quiver_n_s[::sf,::sf],scale=10)
	qk= plt.quiverkey (q,0.5, 1.02, 20/27.8, '20 cm s$^{-1}$', labelpos='N')
	plt.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)

	scale_factor = .001
	skip_number = 2
	row_list, column_list, data_array = scipy.sparse.find(self)
	lat_bins = self.trans_geo.get_lat_bins()[::skip_number]
	lon_bins = self.trans_geo.get_lon_bins()[::3*skip_number]
	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
	XX,YY,ax2 = self.trans_geo.plot_setup(ax = ax2)
	geoms = []
	for k,(point,ns_mean,ew_mean) in enumerate(zip(self.trans_geo.total_list,north_south_data,east_west_data)):
		lat = point.latitude
		lon = point.longitude

		if lat not in lat_bins:
			continue
		if lon not in lon_bins:
			continue


		mask = column_list == k
		if not mask.any():
			continue

		data = data_array[mask]

		ew_holder = self.trans_geo.east_west[row_list[mask],column_list[mask]]
		ns_holder = self.trans_geo.north_south[row_list[mask],column_list[mask]]

		x = []
		y = []
		for i,ew,ns in zip(data,ew_holder,ns_holder): 
			x+=[ew*i]
			y+=[ns*i]

		try:
			w,v = np.linalg.eig(np.cov(x,y))
		except:
			continue
		angle = np.degrees(np.arctan(v[1,np.argmax(w)]/v[0,np.argmax(w)]))
		
		axis1 = max(w)
		axis2 = min(w)

		axis1 = 2*max(w)*np.sqrt(5.991)*1000*np.cos(np.radians(lat))
		axis2 = 2*min(w)*np.sqrt(5.991)*1000


		print('angle = ',angle)
		print('axis1 = ',axis1)
		print('axis2 = ',axis2)
		try:
			lons, lats = ax2.ellipse(geod,lon, lat,axis1*10000,axis2*10000,phi=angle)
			holder = sgeom.Polygon(zip(lons, lats))
			if holder.area>300:
				continue
			if holder.length>150:
				continue
			geoms.append(sgeom.Polygon(zip(lons, lats)))
		except ValueError: 
			print(' there was a value error in the calculation of the transition_matrix')
			continue
	ax2.add_geometries(geoms, ccrs.Geodetic(), facecolor='blue', alpha=0.7)
	plt.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(file_handler.out_file('figure_8'))
	plt.close()

TransMat.quiver_plot = quiver_plot
trans_geo = TransMat.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
trans_geo.quiver_plot()