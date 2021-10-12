from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from TransitionMatrix.Utilities.Compute.trans_read import TransMat,TransitionGeo
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import scipy
import scipy.sparse.linalg
import matplotlib.colors as colors
import matplotlib.cm as cm
import os
import cartopy.crs as ccrs
from pyproj import Geod

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'transition_matrix_plot')


class TransPlot(TransMat):
	def __init__(self, *args,**kwargs):
		super().__init__(*args,**kwargs)

	def number_plot(self,ax=False): 
		number_matrix = self.new_sparse_matrix(self.number_data)
		k = number_matrix.sum(axis=0)
		k = k.T
		print(k)
		number_matrix_plot = self.trans_geo.transition_vector_to_plottable(k)
		XX,YY,ax = self.trans_geo.plot_setup(ax=ax)  
		number_matrix_plot = np.ma.masked_equal(number_matrix_plot,0)   #this needs to be fixed in the plotting routine, because this is just showing the number of particles remaining
		ax.pcolormesh(XX,YY,number_matrix_plot,cmap=plt.cm.magma,vmin=self.trans_geo.number_vmin,vmax=self.trans_geo.number_vmax)
		# plt.title('Transition Density',size=30)
		PCM = ax.get_children()[0]
		cbar = plt.colorbar(PCM,ax=ax)
		cbar.set_label(label='Transition Number',size=30)
		cbar.ax.tick_params(labelsize=30)
		return ax

	def distribution_and_mean_of_column(self,col_idx):
		from GeneralUtilities.Plot.Cartopy.eulerian_plot import PointCartopy
		idx_lat,idx_lon,dummy = tuple(self.trans_geo.total_list[col_idx])
		east_west, north_south = self.return_mean()
		x_mean = east_west[col_idx]+idx_lon
		y_mean = north_south[col_idx]+idx_lat
		XX,YY,ax = PointCartopy(self.trans_geo.total_list[col_idx],lat_grid = self.trans_geo.get_lat_bins(),lon_grid = self.trans_geo.get_lon_bins(),pad=20).get_map()
		plt.pcolormesh(XX,YY,self.trans_geo.transition_vector_to_plottable(np.array(self[:,col_idx].todense()).flatten()))
		plt.colorbar()
		plt.scatter(x_mean,y_mean,c='pink',linewidths=5,marker='x',s=80,zorder=10)


	def return_standard_error(self):
		number_matrix = self.new_sparse_matrix(self.number_data)
		self.trans_geo.get_direction_matrix()
		row_list, column_list, data_array = scipy.sparse.find(self)
		n_s_distance_weighted = self.trans_geo.north_south[row_list,column_list]*data_array
		e_w_distance_weighted = self.trans_geo.east_west[row_list,column_list]*data_array
		# this is like calculating x*f(x)
		n_s_mat = self.new_sparse_matrix(n_s_distance_weighted)
		E_y = np.array(n_s_mat.sum(axis=0)).flatten()
		e_w_mat = self.new_sparse_matrix(e_w_distance_weighted)
		E_x = np.array(e_w_mat.sum(axis=0)).flatten()
		#this is like calculating E(x) = sum(xf(x)) = mean
		ns_x_minus_mu = (self.trans_geo.north_south[row_list,column_list]-E_y[column_list])**2
		ew_x_minus_mu = (self.trans_geo.east_west[row_list,column_list]-E_x[column_list])**2
		std_data = (ns_x_minus_mu+ew_x_minus_mu)*data_array
		std_mat = self.new_sparse_matrix(std_data)
		sigma = np.array(np.sqrt(std_mat.sum(axis=0))).flatten()
		std_error = sigma/np.sqrt(number_matrix.sum(axis=0))
		return np.array(std_error).flatten()

	def standard_error_plot(self):

#Todo: the way this is calculated the diagonal goes to 
#infinity if the number of floats remaining in a bin goes 
#to zero. I suspect that I may not be calculating this properly
		standard_error = self.return_standard_error()
		standard_error_plot = self.trans_geo.transition_vector_to_plottable(standard_error)
		standard_error_plot = np.ma.masked_greater(standard_error_plot,.6)

		plt.figure('Standard Error',figsize=(10,10))
		# m.fillcontinents(color='coral',lake_color='aqua')
		# number_matrix_plot[number_matrix_plot>1000]=1000
		XX,YY,ax = self.trans_geo.plot_setup()  
		number_matrix = self.new_sparse_matrix(self.number_data)
		k = number_matrix.sum(axis=0)
		k = k.T
		standard_error_plot = np.ma.array(standard_error_plot,mask=self.trans_geo.transition_vector_to_plottable(k)==0)
		plt.pcolormesh(XX,YY,standard_error_plot*100,cmap=plt.cm.cividis,vmax=self.trans_geo.std_vmax)
		cbar = plt.colorbar()
		cbar.set_label(label='Mean Standard Error (%)',size=20)
		cbar.ax.tick_params(labelsize=30)
		plt.savefig(file_handler.out_file('standard_error_plot'))

		plt.close()

	def transition_matrix_plot(self):
		plt.figure(figsize=(10,10))
		k = np.diagonal(self.todense())
		transition_plot = self.trans_geo.transition_vector_to_plottable(k)
		XX,YY,ax = self.trans_geo.plot_setup()
		number_matrix = self.new_sparse_matrix(self.number_data)
		k = number_matrix.sum(axis=0)
		k = k.T

		transition_plot = np.ma.array(100*(1-transition_plot),
			mask=self.trans_geo.transition_vector_to_plottable(k)==0)
		plt.pcolormesh(XX,YY,transition_plot,vmin=0,vmax=100) # this is a plot for the tendancy of the residence time at a grid cell
		cbar = plt.colorbar()
		cbar.ax.tick_params(labelsize=30)
		cbar.set_label('% Particles Dispersed',size=30)
		# plt.title('1 - diagonal of transition matrix',size=30)
		plt.savefig(self.plot_folder()+'trans_diag_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.time_step)+'_location_'+str(self.traj_file_type)+'.png')
		plt.close()


	def abs_mean_and_dispersion(self,m=False,arrows=True,degree_sep=4,scale_factor=.2,ellipses=True,mask=None):
		"""
		This plots the mean transition quiver as well as the variance ellipses
		"""
# todo: need to check this over. I think there might be a bug.

		bins_lat,bins_lon = self.bins_generator(self.degree_bins)
		XX,YY = np.meshgrid(bins_lon,bins_lat)
		east_west,north_south = self.new_sparse_matrix(north_south_data)

		quiver_e_w = transition_vector_to_plottable(bins_lat,bins_lon,
			self.total_list,east_west.mean(axis=0).tolist()[0])
		quiver_n_s = transition_vector_to_plottable(bins_lat,bins_lon,
			self.total_list,north_south.mean(axis=0).tolist()[0])

		plt.figure(figsize=(20,20))
		plt.subplot(2,2,1)
		XX,YY,ax,fig = self.trans_geo.plot_setup(bins_lon,bins_lat,self.traj_file_type,
			fill_color=False)  
		mXX,mYY = m(XX,YY)
		m.quiver(mXX,mYY,quiver_e_w,quiver_n_s,scale = 0.01)
		plt.title('vector of average transition')
		n_s_disp = np.zeros(XX.shape)
		e_w_disp = np.zeros(XX.shape)

		for k,((x,y),ns_mean,ew_mean) in enumerate(zip(self.total_list,
			east_west.mean(axis=0).tolist()[0],north_south.mean(axis=0).tolist()[0])):
			mask = column_list == k
			if not mask.any():
				continue
			east_west[row_list[mask],column_list[mask]] \
				= east_west[row_list[mask],column_list[mask]]-ew_mean
			north_south[row_list[mask],column_list[mask]] \
				= north_south[row_list[mask],column_list[mask]]-ns_mean

		plt.subplot(2,2,2)
		dummy,dummy,m = basemap_setup(bins_lon,bins_lat,self.traj_file_type,fill_color=False)  
		out1 = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,abs(north_south).mean(axis=0).tolist()[0])
		m.pcolor(mXX,mYY,out1)
		plt.colorbar()
		plt.title('meridional dispersion')
		plt.subplot(2,2,3)
		dummy,dummy,m = basemap_setup(bins_lon,bins_lat,self.traj_file_type,fill_color=False)  
		out2 = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,abs(east_west).mean(axis=0).tolist()[0])
		m.pcolor(mXX,mYY,out2)
		plt.colorbar()
		plt.title('zonal dispersion')
		plt.subplot(2,2,4)
		dummy,dummy,m = basemap_setup(bins_lon,bins_lat,self.traj_file_type,fill_color=False)  
		m.pcolor(mXX,mYY,np.ma.masked_greater(out2/out1,7))
		plt.colorbar()
		plt.title('ratio of zonal to meridional')
		plt.savefig(self.plot_folder()+'transition_matrix_dispersion')


	def quiver_plot(self):
		from matplotlib import patches
		import shapely.geometry as sgeom

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
		qk= plt.quiverkey (q,0.5, 1.02, 1, '1 km hr$^{-1}$', labelpos='N')
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
		plt.savefig(file_handler.out_file('quiver_ellipse'))
		plt.close()


	def plot_latest_soccom_locations(self,m):
		try:
			self.soccom
		except AttributeError:
			self.soccom = SOCCOM(degree_bin_lat=self.bins_lat,degree_bin_lon=self.bins_lon)
		y,x = zip(*np.array(self.total_list)[self.soccom.vector>0])
		m.scatter(x,y,marker='*',color='b',s=34,latlon=True)
		return m

	def get_argo_vector():
		cdict = {'red':  ((0.0, 0.4, 1.0),
					 (1/6., 1.0, 1.0),
					 (1/2., 1.0, 0.8),
					 (5/6., 0.0, 0.0),
					 (1.0, 0.0, 0.0)),

				 'green':  ((0.0, 0.0, 0.0),
					 (1/6., 0.0, 0.0),
					 (1/2., 0.8, 1.0),
					 (5/6., 1.0, 1.0),
					 (1.0, 0.4, 0.0)),

				 'blue': ((0.0, 0.0, 0.0),
					 (1/6., 0.0, 0.0),
					 (1/2., 0.9, 0.9),
					 (5/6., 0.0, 0.0),
					 (1.0, 0.0, 0.0))
			}

		cmap = colors.LinearSegmentedColormap('my_colormap', cdict, 100) 
		float_vector = self.get_argo_vector()
		plt.figure()
		plot_vector = self.transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,float_vector)
		original_plot_vector = plot_vector
		plot_vector = np.ma.masked_equal(plot_vector,0)
		m,XX,YY = self.matrix.matrix_plot_setup()
		m.pcolor(XX,YY,plot_vector,cmap=cmap,vmax=2,vmin=0)
		plt.colorbar()
		plt.title('density/age for '+str(self.info.degree_bins)+' at 0 days')
		for _ in range(3):
			float_vector = self.matrix.transition_matrix.todense().dot(float_vector)
		plt.figure()
		plot_vector = self.transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,float_vector)
		plot_vector = np.ma.masked_equal(plot_vector,0)
		print(plot_vector.sum()-original_plot_vector.sum())
		m,XX,YY = self.matrix.matrix_plot_setup()
		m.pcolor(XX,YY,plot_vector,cmap=cmap,vmax=2,vmin=0)
		plt.colorbar()
		plt.title('density/age for '+str(self.info.degree_bins)+' at '+str((_+1)*self.info.date_span_limit)+' days')            
		plt.show()

	def future_agregation_plot(self):
		w = traj_class.transition_matrix
		vector = np.ones((w.shape[0],1))
		vector = vector/vector.sum()*100
		for k in range(20):
			months = 6*2**k
			w = w.dot(w)
			future_output = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,(w.dot(scipy.sparse.csc_matrix(vector))).todense())
			future_output = np.ma.masked_less(future_output,10**-2) 
			plt.figure(figsize=(10,10))
			m = Basemap(projection='cyl',fix_aspect=False)
			# m.fillcontinents(color='coral',lake_color='aqua')
			m.drawcoastlines()
			XX,YY = m(self.X,self.Y)
			m.pcolormesh(XX,YY,future_output,vmin=0,vmax=future_output.max())
			plt.colorbar(label='% particles')
			plt.title('Distribution after '+str(months)+' Months')
			plt.show()
  

def five_year_density_plot():
	holder = argos_class = TransMat.load_from_type(GeoClass=GPSGeo,lat_spacing = 2,lon_spacing = 3,time_step = 90)
	t = 90
	for k in range(8):
		print(k)
		t = 2*t
		holder = holder.dot(holder)
	trans_mat = holder
	plottable = scipy.sparse.csc_matrix(trans_mat).dot(scipy.sparse.csc_matrix(np.ones([trans_mat.shape[0],1]))) 
	plottable = trans_mat.trans_geo.transition_vector_to_plottable(plottable.todense())
	plottable = np.ma.masked_array(plottable,mask=abs(plottable)<10**-6)

	XX,YY,ax = trans_mat.trans_geo.plot_setup()	
	ax.pcolormesh(XX,YY,plottable*100,vmax=175,vmin=0)
	PCM = ax.get_children()[0]
	plt.colorbar(PCM,ax=ax,label='Relative Chance of Aggregation (%)')
	plt.savefig(file_handler.out_file('future_aggreation_gps'))
	plt.close()

def eig_all_plots(trans_mat):
	def plot_the_data(plottable):
		XX,YY = np.meshgrid(bins_lon,bins_lat)
		XX,YY,m = basemap_setup(bins_lat,bins_lon,trans_mat.traj_file_type)
		plottable = np.ma.masked_array(plottable,mask=abs(plottable)<10**-(2.5))
		m.pcolormesh(XX,YY,plottable,vmin=-10**-(1.5),vmax=10**-(1.5))
		plt.colorbar()
		# if not real_mask.all():
		#     m,e_w,n_s = TransPlot(matrix,total_list=trans_mat.total_list,lat_spacing=trans_mat.degree_bins[0]
		#         ,lon_spacing=trans_mat.degree_bins[1],traj_file_type='Argo').quiver_plot(m=m,ellipses=False,mask=mask)


	trans_mat = TransPlot.load_from_type(1,1,30)
	# transmat = transmat.remove_small_values(0.04)
	# transmat.rescale()
	plot_name = str(int(trans_mat.degree_bins[0]))+'_'+str(int(trans_mat.degree_bins[1]))+'_'+str(trans_mat.traj_file_type)
	r_e_vals,r_e_vecs = scipy.sparse.linalg.eigs(trans_mat.T,k=60) 

	r_idx = [x for _,x in sorted(zip([abs(k) for k in r_e_vals],range(len(r_e_vals))))]
	r_idx = r_idx[::-1]
	# l_idx = [x for _,x in sorted(zip([abs(k) for k in l_e_vals],range(len(l_e_vals))))]
	# l_idx = l_idx[::-1]
	mag_mask = 10**-3
	vmin_mag = 3.
	plt.plot(r_e_vals[r_idx])
	plt.savefig(plot_name+'_e_val_spectrum_r')
	plt.close()

	bins_lat,bins_lon = trans_mat.bins_generator(trans_mat.degree_bins)
	# bins_lat,bins_lon = trans_mat.bins_generator(trans_mat.degree_bins)
	for n in range(len(r_idx)):

		r_e_vec = r_e_vecs[:,r_idx[n]]
		r_e_val = r_e_vals[r_idx[n]]

		# matrix_component = r_e_val*np.outer(r_e_vec,r_e_vec)/(r_e_vec**2).sum()
		# matrix_component = matrix_component/abs(matrix_component).max()
		
		# eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
		# eig_vec_token_holder = np.ma.masked_inside(eig_vec_token,-mag_mask,mag_mask)
		# mag = eig_vec_token_holder.data.max()
		# real_mask = eig_vec_token_holder.mask
		plt.subplot(1,2,1)
		eig_vec_token = transition_vector_to_plottable(bins_lat,bins_lon,trans_mat.total_list,r_e_vec.real)
		plot_the_data(eig_vec_token)
		plt.title('real right evec')
		plt.subplot(1,2,2)
		eig_vec_token = transition_vector_to_plottable(bins_lat,bins_lon,trans_mat.total_list,r_e_vec.imag)
		plot_the_data(eig_vec_token)
		plt.title('imag right evec')

		# eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
		# eig_vec_token_holder = np.ma.masked_inside(eig_vec_token,-mag_mask,mag_mask)
		# mag = eig_vec_token_holder.data.max()
		# imag_mask = eig_vec_token_holder.mask
		plt.suptitle(str(r_e_val))
		plt.savefig(plot_name+'transition_'+str(n))
		plt.close()

	def eig_total_plot(self,label=''):
		eig_vals,eig_vecs = scipy.sparse.linalg.eigs(self,k=30)


		overlap_colors = [(0, 1, 0),(1, 1, 0),(0, 1, 1),(1, 0, 0),(0, 0, 1),(1, 0, 1),(0.2, 1, 0.5),(0.5, 1, 0),(0, 1, 0.5),(0.5, 0.5, 0.1),
		(0, 1, 0),(1, 1, 0),(0, 1, 1),(1, 0, 0),(0, 0, 1),(1, 0, 1),(0.2, 1, 0.5),(0.5, 1, 0),(0, 1, 0.5),(0.5, 0.5, 0.1),
		(0, 1, 0),(1, 1, 0),(0, 1, 1),(1, 0, 0),(0, 0, 1),(1, 0, 1),(0.2, 1, 0.5),(0.5, 1, 0),(0, 1, 0.5),(0.5, 0.5, 0.1),
		]
		cmap_name = 'my_list'
		n_bin = 2
		cmaps=[colors.LinearSegmentedColormap.from_list(cmap_name, [_,_] , N=n_bin) for _ in overlap_colors]
		XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type) 
		for k,eig_val in enumerate(eig_vals):
			eig_vec = eig_vecs[:,k]
			eig_vec_token = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,eig_vec)
			eig_vec_token = abs(eig_vec_token)
			eig_vec_token = np.ma.masked_less(eig_vec_token,eig_vec_token.mean()+1.1*eig_vec_token.std()) 
			m.pcolormesh(XX,YY,eig_vec_token,cmap=cmaps[k],alpha=0.7)
		plt.show()


		for k,eig_val in enumerate(eig_vals):
			print('plotting eigenvector '+str(k)+' for '+str(self.degree_bins)+' eigenvalue is '+str(eig_val))
			eig_vec = eig_vecs[:,k]
			assert (self.todense().dot(eig_vec.T)-eig_val*eig_vec).max()<10**-1


			plt.figure(figsize=(10,10))
			plt.subplot(3,1,1)
			self.eig_vec_plot(eig_vec.real)
			plt.title('left eig vec (real)')
			plt.subplot(3,1,2)
			self.eig_vec_plot(np.absolute(eig_vec))
			plt.title('left eig vec (absolute)')
			plt.subplot(3,1,3)
			plt.plot(np.abs(eig_vals)[::-1])
			plt.ylabel('Eigen Value')
			plt.xlabel('Eign Number')
			plt.ylim([0,1])
			plt.xlim([0,len(eig_vals)])
			plt.title('Eigen Value Spectrum')
			plt.suptitle('Eigen Value '+'{0:.4f} + {1:.4f}i'.format(float(eig_val.real),float(eig_val.imag)))
			plt.savefig('../plots/eig_vals_'+str(k)+'_time_step_'+str(self.time_step)+'.png')
			plt.close()

def SOCCOM_death_plot():
	from TransitionMatrix.Utilities.Plot.argo_data import BGC as SOCCOM
	from TransitionMatrix.Utilities.Plot.argo_data import Core as Argo
	from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
	from GeneralUtilities.Plot.Cartopy.regional_plot import SOSECartopy
	from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo
	from GeneralUtilities.Data.lagrangian.bgc.bgc_read import BGCReader
	from GeneralUtilities.Data.lagrangian.argo.argo_read import ArgoReader

	from GeneralUtilities.Data.lagrangian.argo.argo_read import full_argo_list
	from TransitionMatrix.Utilities.Inversion.target_load import InverseGeo,InverseInstance
	full_argo_list()

	def recent_bins_by_sensor(variable,lat_bins,lon_bins,float_type):
		date_list = BGCReader.get_recent_date_list()
		bin_list = BGCReader.get_recent_bins(lat_bins,lon_bins)
		sensor_list = BGCReader.get_sensors()
		sensor_mask = [variable in x for x in sensor_list]
		date_mask =[max(date_list)-datetime.timedelta(days=180)<x for x in date_list]
		if float_type == 'BGC':
			soccom_mask = ['SOCCOM' in x.meta.project_name for dummy,x in BGCReader.all_dict.items()]
			mask = np.array(sensor_mask)&np.array(date_mask)&np.array(soccom_mask)
		else:
			mask = np.array(sensor_mask)&np.array(date_mask)

		age_list = [(x.prof.date._list[-1]-x.prof.date._list[0]).days/365. for x in BGCReader.all_dict.values()]
		return (np.array(bin_list)[mask],1/(np.ceil(age_list)+1)[mask])

	def recent_floats(cls,GeoClass, FloatClass):
		out_list = []
		for variable in GeoClass.variable_list:
			float_var = GeoClass.variable_translation_dict[variable]
			var_grid,age_list = recent_bins_by_sensor(float_var,GeoClass.get_lat_bins(),GeoClass.get_lon_bins(),cls.traj_file_type)
			idx_list = [GeoClass.total_list.index(x) for x in var_grid if x in GeoClass.total_list]
			holder_array = np.zeros([len(GeoClass.total_list),1])
			for k,idx in enumerate(idx_list):
				holder_array[idx]+=age_list[k]
			out_list.append(holder_array)
		out = np.vstack(out_list)
		return cls(out,trans_geo=GeoClass)


	inverse_mat = InverseInstance.load_from_type(InverseGeo,lat_sep=2,lon_sep=2,l=300,depth_idx=2)
	traj_class_1 = TransPlot.load_from_type(lat_spacing=4,lon_spacing=4,time_step=180)
	traj_class_1.trans_geo.plot_class = SOSECartopy
	traj_class_1.trans_geo.variable_list = ['so']
	traj_class_1.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	float_vector_1 = recent_floats(SOCCOM,traj_class_1.trans_geo, BGCReader)

	traj_class_2 = TransPlot.load_from_type(lat_spacing=2,lon_spacing=2,time_step=180)
	traj_class_2.trans_geo.plot_class = GlobalCartopy
	traj_class_2.trans_geo.variable_list = ['so']
	traj_class_2.trans_geo.variable_translation_dict = inverse_mat.trans_geo.variable_translation_dict 
	float_vector_2 = recent_floats(Argo,traj_class_2.trans_geo, ArgoReader)

	bins_lat = traj_class.trans_geo.get_lat_bins()
	bins_lon = traj_class.trans_geo.get_lon_bins()
	fig = plt.figure(figsize=(14,14))
	ax1 = fig.add_subplot(2,1,1, projection=ccrs.PlateCarree())
	XX,YY,ax1 = traj_class_1.trans_geo.plot_setup(ax = ax1)
	plottable = traj_class_1.trans_geo.transition_vector_to_plottable(traj_class.todense().dot(float_vector_1.todense()))
	traj_class_1.traj_file_type = 'SOSE'
	ax1.pcolormesh(XX,YY,np.ma.masked_less(plottable,5*10**-3),cmap=plt.cm.YlGn,vmax=0.3)
	PCM = ax1.get_children()[0]
	row_idx,column_idx,data = scipy.sparse.find(float_vector_1)

	lats = [list(traj_class_1.trans_geo.total_list)[x].latitude for x in row_idx]
	lons = [list(traj_class_1.trans_geo.total_list)[x].longitude for x in row_idx]

	ax1.scatter(lons,lats,c='m',marker='*')

	plt.annotate('a', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2 = fig.add_subplot(2,1,2, projection=ccrs.PlateCarree())
	traj_class_2.traj_file_type = 'Argo'
	XX,YY,ax2 = traj_class_2.trans_geo.plot_setup(ax = ax2)
	plottable = traj_class_2.trans_geo.transition_vector_to_plottable(traj_class_2.todense().dot(float_vector_2.todense()))
	ax2.pcolormesh(XX,YY,np.ma.masked_less(plottable,5*10**-3),cmap=plt.cm.YlGn,vmax = 0.3)
	PCM = ax2.get_children()[0]
	fig.colorbar(PCM,ax=[ax1,ax2],fraction=0.10,label='Probability Density/Age')
	row_idx,column_idx,data = scipy.sparse.find(float_vector_2)

	lats = [list(traj_class_2.trans_geo.total_list)[x].latitude for x in row_idx]
	lons = [list(traj_class_2.trans_geo.total_list)[x].longitude for x in row_idx]

	ax2.scatter(lons,lats,c='r',s=4)
	plt.annotate('b', xy = (0.17,0.9),xycoords='axes fraction',zorder=11,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(file_handler.out_file('death_plot'))
	plt.close()

def number_matrix_plot():
	import sys
	from PIL import Image
	for traj in ['argo','SOSE']:
		holder = TransPlot.load_from_type(2,2,60,traj_type=traj) 
		holder.number_plot()
		holder.standard_error_plot()
		images = [Image.open(x) for x in [holder.std_error_file(),holder.number_file()]]
		widths, heights = zip(*(i.size for i in images))
		total_width = sum(widths)
		max_height = max(heights)

		new_im = Image.new('RGB', (total_width, max_height))

		x_offset = 0
		for im in images:
		  new_im.paste(im, (x_offset,0))
		  x_offset += im.size[0]
		new_im.save('/Users/pchamberlain/Projects/transition_matrix_paper/plots/'+traj+'number_plot.jpg')


def eigen_spectrum_plot():
	for lat,lon in [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]:
		for time in [30,60,90,120,140,160,180]:
			holder = TransPlot.load_from_type(lon,lat,time)
			# trans_mat = holder.multiply(multiplyer,0.0000001)
			e_vals,e_vecs =scipy.sparse.linalg.eigs(holder,k=40)
			plt.plot(np.sort(e_vals)[::-1],label=str(time)+' days')
		plt.legend()
		plt.savefig(holder.plot_folder()+'eigen_time_plot')
		plt.close()

def standard_error_plot():
	for traj_type in ['argo','SOSE']:
		error_list = []
		for lat,lon in [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]:
			for time in [30,60,90,120,140,160,180]:
				print(time)
				holder_low_res = TransPlot.load_from_type(lon,lat,time,traj_type=traj_type)
				std_lr = holder_low_res.return_standard_error()
				error_list.append((lat,lon,time,np.nanmean(std_lr),np.nanstd(std_lr)))
		start_idx = (np.array(range(6)))*7
		end_idx = (np.array(range(6))+1)*7
		fig,ax = plt.subplots()
		for ii,kk in zip(start_idx,end_idx):
			holder = error_list[ii:kk]
			label = str((holder[0][0],holder[0][1]))
			dummy,dummy,time,mean,std = zip(*holder)
			mean = np.array(mean)
			print(mean)
			print(time)
			# std = np.array(std)
			# ax.fill_between(time,mean-std,mean+std,alpha=0.2)
			ax.plot(time,mean)
			ax.scatter(time,mean,label=label)
		plt.legend()
		plt.xlabel('Timestep (days)')
		plt.ylabel('Mean Error')
		plt.savefig('/Users/pchamberlain/Projects/transition_matrix_paper/plots/'+traj_type+'_error_compare.jpg')
		plt.close()

def resolution_bias_plot():
	from transition_matrix.compute.compute_utils import matrix_size_match    
	for traj_type in ['argo','SOSE']:
		error_list = []
		for lat,lon in [(2,2),(3,3),(4,4),(2,3),(3,6)]:
			for time in [30,60,90,120,140,160,180]:
				print(time)
				print(lat)
				print(lon)
				holder_low_res = TransPlot.load_from_type(lon,lat,time,traj_type=traj_type)
				holder_low_res = holder_low_res.multiply(2)        
				holder_high_res = TransPlot.load_from_type(1,1,time,traj_type=traj_type)
				holder_high_res = holder_high_res.multiply(2)                
				holder_high_res = holder_high_res.reduce_resolution((lat,lon))

				holder_low_res,holder_high_res = matrix_size_match(holder_low_res,holder_high_res)
				east_west_lr, north_south_lr = holder_low_res.return_mean()
				east_west_hr, north_south_hr = holder_high_res.return_mean()

				list_low_res = []
				list_high_res = []
				for i in range(holder_low_res.shape[0]):
					list_low_res.append((east_west_lr[:,i].data.mean(),north_south_lr[:,i].data.mean()))
					list_high_res.append((east_west_hr[:,i].data.mean(),north_south_hr[:,i].data.mean()))
				e_w_lr, n_s_lr = zip(*list_low_res)
				e_w_hr, n_s_hr = zip(*list_high_res)
				e_w_diff = np.array(e_w_lr)-np.array(e_w_hr)
				n_s_diff = np.array(n_s_lr)-np.array(n_s_hr)
				total_diff = np.sqrt(e_w_diff**2+n_s_diff**2)
				error_list.append((lat,lon,time,total_diff.mean(),total_diff.std()))
		start_idx = (np.array(range(5)))*7
		end_idx = (np.array(range(5))+1)*7
		fig,ax = plt.subplots()
		for ii,kk in zip(start_idx,end_idx):
			holder = error_list[ii:kk]
			label = str((holder[0][0],holder[0][1]))
			dummy,dummy,time,mean,std = zip(*holder)
			mean = np.array(mean)
			print(mean)
			print(time)
			# std = np.array(std)
			# ax.fill_between(time,mean-std,mean+std,alpha=0.2)
			ax.plot(time,mean)
			ax.scatter(time,mean,label=label)
		plt.legend()
		plt.xlabel('Timestep (days)')
		plt.ylabel('Mean Error')
		plt.savefig('/Users/pchamberlain/Projects/transition_matrix_paper/plots/'+traj_type+'_resolution_error_compare.jpg')
		plt.close()