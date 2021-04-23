from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from TransitionMatrix.Utilities.Compute.trans_read import TransMat
import scipy
import scipy.sparse.linalg
import matplotlib.colors as colors
import matplotlib.cm as cm
from TransitionMatrix.Utilities.Plot.plot_utils import cartopy_setup,transition_vector_to_plottable
from TransitionMatrix.Utilities.Plot.argo_data import SOCCOM,Argo
import os
# class TrajectoryPlot(Trajectory):
#     pass
#     def profile_density_plot(self): #plot the density of profiles
#         ZZ = np.zeros([len(self.bins_lat),len(self.bins_lon)])
#         series = self.df.groupby('bin_index').count()['Cruise']
#         for item in series.iteritems():
#             tup,n = item
#             ii_index = self.bins_lon.index(tup[1])
#             qq_index = self.bins_lat.index(tup[0])
#             ZZ[qq_index,ii_index] = n           
#         plt.figure(figsize=(10,10))
#         XX,YY,m = basemap_setup(self.bins_lon,self.bins_lat,self.traj_file_type)    
#         ZZ = np.ma.masked_equal(ZZ,0)
#         m.pcolormesh(XX,YY,ZZ,vmin=0,cmap=plt.cm.magma)
#         plt.title('Profile Density',size=30)
#         plt.colorbar(label='Number of float profiles')
#         print 'I am saving argo dense figure'
#         plt.savefig('../plots/argo_dense_data/number_matrix_degree_bins_'+str(self.degree_bins)+'.png')
#         plt.close()

#     def deployment_density_plot(self): # plot the density of deployment locations
#         ZZ = np.zeros([len(self.bins_lat),len(self.bins_lon)])
#         series = self.df.drop_duplicates(subset=['Cruise'],keep='first').groupby('bin_index').count()['Cruise']
#         for item in series.iteritems():
#             tup,n = item
#             ii_index = self.bins_lon.index(tup[1])
#             qq_index = self.bins_lat.index(tup[0])
#             ZZ[qq_index,ii_index] = n  
#         plt.figure(figsize=(10,10))
#         XX,YY,m = basemap_setup(self.bins_lon,self.bins_lat,self.traj_file_type)    
#         ZZ = np.ma.masked_equal(ZZ,0)
#         m.pcolormesh(XX,YY,ZZ,vmin=0,vmax=30,cmap=plt.cm.magma)
#         plt.title('Deployment Density',size=30)
#         plt.colorbar(label='Number of floats deployed')
#         plt.savefig('../plots/deployment_number_data/number_matrix_degree_bins_'+str(self.degree_bins)+'.png')
#         plt.close()

#     def deployment_location_plot(self): # plot all deployment locations
#         plt.figure(figsize=(10,10))
#         XX,YY,m = basemap_setup(self.bins_lon,self.bins_lat,self.traj_file_type)    
#         series = self.df.drop_duplicates(subset=['Cruise'],keep='first')
#         m.scatter(series.Lon.values,series.Lat.values,s=0.2)
#         plt.title('Deployment Locations',size=30)
#         plt.savefig('../plots/deployment_number_data/deployment_locations.png')
#         plt.close()

class TransPlot(TransMat):
	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
		,time_step=None,number_data=None,traj_file_type=None,rescale=False):
		super(TransPlot,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type,rescale=rescale)
		from transition_matrix.definitions import ROOT_DIR
		self.base = ROOT_DIR+'/output/'+self.traj_file_type+'/plots/'
		#         self.multiplyer = time_multiplyer
#         matrix = copy.deepcopy(self)
#         for _ in range(time_multiplyer-1):
#             matrix = matrix.dot(self)
#         self = matrix
# routine to propogate transition matrix into the future, can be used for forward or inverse problem
	def plot_folder(self):
		if not os.path.isdir(self.base):
			os.mkdir(self.base)
		degree_bins = [str(self.degree_bins[0]),str(self.degree_bins[1])]
		folder = self.base+str(self.time_step)+'-'+str(self.degree_bins[0]).replace('.','_')+'-'+str(self.degree_bins[1]).replace('.','_')+'/'
		if not os.path.isdir(folder):
			os.mkdir(folder)
		return self.base+str(self.time_step)+'-'+str(self.degree_bins[0]).replace('.','_')+'-'+str(self.degree_bins[1]).replace('.','_')+'/'

	def number_folder(self):
		folder = self.plot_folder()+'number_matrix/'
		if not os.path.isdir(folder):
			os.mkdir(folder)
		return folder


	def number_file(self):
		return self.number_folder()+'number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.time_step)+'_location_'+str(self.traj_file_type)+'.png'

	def std_error_file(self):
		return self.number_folder()+'standard_error_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.time_step)+'_location_'+str(self.traj_file_type)+'.png'

	def remove_small_values(self,value):

		row_idx,column_idx,data = scipy.sparse.find(self)
		mask = data>value
		row_idx = row_idx[mask]
		column_idx = column_idx[mask]
		data = data[mask]
		arg = scipy.sparse.csc_matrix((data,(row_idx,column_idx)),shape=(len(self.total_list),len(self.total_list)))
		return TransPlot(arg,shape = self.shape,total_list=self.total_list,lat_spacing=self.degree_bins[0],
			lon_spacing=self.degree_bins[1],traj_file_type=self.traj_file_type,rescale=False)

	def multiply(self,mult,value=0.02):
		mat1 = self.remove_small_values(self.data.mean()/25)
		mat2 = self.remove_small_values(self.data.mean()/25)
		for k in range(mult):
			print('I am at ',k,' step in the multiplication')
			mat_holder = mat1.dot(mat2)
			mat1 = TransPlot(mat_holder,shape = self.shape,total_list=self.total_list,
			lat_spacing=self.degree_bins[0],lon_spacing=self.degree_bins[1],
			traj_file_type=self.traj_file_type,rescale=True)
			mat1 = mat1.remove_small_values(mat1.data.mean()/10)

		return mat1

	def number_plot(self): 
		self.number_matrix = self.new_sparse_matrix(self.number_data)
		k = self.number_matrix.sum(axis=0)
		k = k.T
		print(k)
		bins_lat,bins_lon = self.bins_generator(self.degree_bins)
		number_matrix_plot = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,k)
		plt.figure('number matrix',figsize=(10,10))
		XX,YY,m = basemap_setup(bins_lat,bins_lon,self.traj_file_type)  
		number_matrix_plot = np.ma.masked_equal(number_matrix_plot,0)   #this needs to be fixed in the plotting routine, because this is just showing the number of particles remaining
		if self.traj_file_type == 'SOSE':
			m.pcolormesh(XX,YY,number_matrix_plot,cmap=plt.cm.magma,vmin=0,vmax=2000)
		else:
			m.pcolormesh(XX,YY,number_matrix_plot,cmap=plt.cm.magma,vmin=0,vmax=300)
		# plt.title('Transition Density',size=30)
		cbar = plt.colorbar()
		cbar.set_label(label='Transition Number',size=30)
		cbar.ax.tick_params(labelsize=30)
		plt.annotate('B', xy = (0.17,0.9),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		


		plt.savefig(self.number_file())
		plt.close()

	def return_standard_error(self):
		number_matrix = self.new_sparse_matrix(self.number_data)
		self.get_direction_matrix()
		row_list, column_list, data_array = scipy.sparse.find(self)
		n_s_distance_weighted = self.north_south[row_list,column_list]*data_array
		e_w_distance_weighted = self.east_west[row_list,column_list]*data_array
		# this is like calculating x*f(x)
		n_s_mat = self.new_sparse_matrix(n_s_distance_weighted)
		E_y = np.array(n_s_mat.sum(axis=0)).flatten()
		e_w_mat = self.new_sparse_matrix(e_w_distance_weighted)
		E_x = np.array(e_w_mat.sum(axis=0)).flatten()
		#this is like calculating E(x) = sum(xf(x)) = mean


		ns_x_minus_mu = (self.north_south[row_list,column_list]-E_y[column_list])**2
		ew_x_minus_mu = (self.east_west[row_list,column_list]-E_x[column_list])**2
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
		bins_lat,bins_lon = self.bins_generator(self.degree_bins)
		standard_error_plot = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,standard_error)
		standard_error_plot = np.ma.masked_greater(standard_error_plot,.6)

		plt.figure('Standard Error',figsize=(10,10))
		# m.fillcontinents(color='coral',lake_color='aqua')
		# number_matrix_plot[number_matrix_plot>1000]=1000
		XX,YY,m = basemap_setup(bins_lat,bins_lon,self.traj_file_type)  
		q = self.number_matrix.sum(axis=0)
		q = q.T
		q = np.ma.masked_equal(q,0)
		standard_error_plot = np.ma.array(standard_error_plot,mask=transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,q)==0)
		if self.traj_file_type == 'SOSE':
			m.pcolormesh(XX,YY,standard_error_plot*100,cmap=plt.cm.cividis,vmax=15)
		else:
			m.pcolormesh(XX,YY,standard_error_plot*100,cmap=plt.cm.cividis,vmax=30)
		cbar = plt.colorbar()
		cbar.set_label(label='Mean Standard Error (%)',size=30)
		cbar.ax.tick_params(labelsize=30)
		plt.annotate('A', xy = (0.17,0.9),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		plt.savefig(self.std_error_file())

		plt.close()

	def transition_matrix_plot(self):
		plt.figure(figsize=(10,10))
		k = np.diagonal(self.todense())
		bins_lat,bins_lon = self.bins_generator(self.degree_bins)
		transition_plot = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,k[0])
		XX,YY,m = basemap_setup(bins_lat,bins_lon,self.traj_file_type)  
		transition_plot = np.ma.array(100*(1-transition_plot),
			mask=transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,self.number_data)==0)
		m.pcolormesh(XX,YY,transition_plot,vmin=0,vmax=100) # this is a plot for the tendancy of the residence time at a grid cell
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
		dummy,dummy,m = basemap_setup(bins_lon,bins_lat,self.traj_file_type,
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
		row_list, column_list, data_array = scipy.sparse.find(self)
		self.get_direction_matrix()
		bins_lat,bins_lon = self.bins_generator(self.degree_bins)
		XX,YY = np.meshgrid(bins_lon,bins_lat)


		east_west_data = self.east_west[row_list,column_list]*data_array
		north_south_data = self.north_south[row_list,column_list]*data_array
		east_west = self.new_sparse_matrix(east_west_data)
		north_south = self.new_sparse_matrix(north_south_data)

		eastwest_mean = []
		northsouth_mean = []
		for k in range(self.shape[0]):
			eastwest_mean.append(east_west[:,k].data.mean())
			northsouth_mean.append(north_south[:,k].data.mean())



		quiver_e_w = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,np.array(eastwest_mean)*self.degree_bins[0]*111/(self.time_step*24))
		quiver_n_s = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,np.array(northsouth_mean)*self.degree_bins[0]*111/(self.time_step*24))


		std_number = 1/4.

		quiver_e_w = np.ma.masked_greater(quiver_e_w,np.mean(eastwest_mean)+std_number*np.std(eastwest_mean))
		quiver_e_w = np.ma.masked_less(quiver_e_w,np.mean(eastwest_mean)-std_number*np.std(eastwest_mean))

		quiver_n_s = np.ma.masked_greater(quiver_n_s,np.mean(northsouth_mean)+std_number*np.std(northsouth_mean))
		quiver_n_s = np.ma.masked_less(quiver_n_s,np.mean(northsouth_mean)-std_number*np.std(northsouth_mean))



		plt.figure(figsize=(40,20))
		plt.subplot(1,2,1)
		scale_factor = 2
		dummy,dummy,m = basemap_setup(bins_lon[::scale_factor],bins_lat[::scale_factor],self.traj_file_type,fill_color=False)  
		mXX,mYY = m(XX[::scale_factor,::scale_factor],YY[::scale_factor,::scale_factor])
		q = m.quiver(mXX,mYY,quiver_e_w[::scale_factor,::scale_factor],quiver_n_s[::scale_factor,::scale_factor],scale = .6)
		qk= plt.quiverkey (q,0.5, 1.02, 0.2, '0.2 km/hr', labelpos='N')
		plt.annotate('A', xy = (0.17,0.9),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)



		scale_factor = 1.3
		skip_number = 2
		plt.subplot(1,2,2)
		lon_bins_holder = bins_lon[::skip_number]
		lat_bins_holder = bins_lat[::skip_number]
		dummy,dummy,m = basemap_setup(lon_bins_holder,lat_bins_holder,self.traj_file_type,fill_color=False)  
		east_west = self.new_sparse_matrix(east_west_data)
		north_south = self.new_sparse_matrix(north_south_data)
		# ew_mask = np.array(east_west[row_list,column_list]!=0)[0]
		# ns_mean = np.array(north_south[row_list,column_list]!=0)[0]
		for k,((lat,lon),ns_mean,ew_mean) in enumerate(zip(self.total_list,
			northsouth_mean,eastwest_mean)):

			if lon not in lon_bins_holder:
				continue
			if lat not in lat_bins_holder:
				continue
			if lon>160:
				continue
			if lon<-160:
				continue
			mask = column_list == k
			if not mask.any():
				continue

			data = data_array[mask]

			ew_holder = self.east_west[row_list[mask],column_list[mask]]
			ns_holder = self.north_south[row_list[mask],column_list[mask]]

			x = []
			y = []
			for i,ew,ns in zip(data,ew_holder,ns_holder): 
				x+=[ew]*int(i)
				y+=[ns]*int(i)

			try:
				w,v = np.linalg.eig(np.cov(x,y))
			except:
				continue
			angle = np.degrees(np.arctan(v[1,np.argmax(w)]/v[0,np.argmax(w)]))+90
			axis1 = np.log(2*max(w)*np.sqrt(5.991))

			axis2 = np.log(2*min(w)*np.sqrt(5.991))

			print('angle = ',angle)
			print('axis1 = ',axis1)
			print('axis2 = ',axis2)
			try:
				poly = m.ellipse(lon, lat, axis1*scale_factor,axis2*scale_factor, 80,phi=angle+90,line=False, facecolor='green', zorder=3,alpha=0.35)
			except ValueError:
				print(' there was a value error in the calculation of the transition_matrix')
				continue
		plt.annotate('B', xy = (0.17,0.9),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
		plt.savefig('/Users/pchamberlain/Projects/transition_matrix_paper/plots/quiver.jpg')
		plt.close()

	# def ellipse_plot(self):

#     def quiver_plot(self,m=False,arrows=True,degree_sep=4,scale_factor=.2,ellipses=True,mask=None):
#         """
#         This plots the mean transition quiver as well as the variance ellipses
#         """
# # todo: need to check this over. I think there might be a bug.
#         row_list, column_list, data_array = scipy.sparse.find(self)
#         self.get_direction_matrix()
#         east_west_data = self.east_west[row_list,column_list]*data_array
#         north_south_data = self.north_south[row_list,column_list]*data_array
#         east_west = self.new_sparse_matrix(east_west_data)
#         north_south = self.new_sparse_matrix(north_south_data)

#         print('I have succesfully multiplied the transition_matrices')


#         if not m:
#             bins_lat,bins_lon = self.bins_generator(self.degree_bins)
#             dummy,dummy,m = basemap_setup(bins_lon,bins_lat,self.traj_file_type,fill_color=False)  
#         m.drawmapboundary()
#         m.fillcontinents()
#         Y_ = np.arange(-68.0,66.0,degree_sep)
#         X_ = np.arange(-170.0,170.0,2*degree_sep)
#         if mask is not None:
#             bins_lat,bins_lon = self.bins_generator(self.degree_bins)
#             lon_mask = [x in X_ for x in bins_lon]
#             lat_mask = [x in Y_ for x in bins_lat]
#             XX,YY = np.meshgrid(lon_mask,lat_mask)
#             mask = mask[XX&YY].reshape(np.sum(lat_mask),np.sum(lon_mask))

#         XX,YY = np.meshgrid(X_,Y_)
#         n_s = np.zeros(XX.shape)
#         e_w = np.zeros(XX.shape)
#         XX,YY = m(XX,YY)
#         for i,lat in enumerate(Y_):
#             for k,lon in enumerate(X_):
#                 print('lat = ',lat)
#                 print('lon = ',lon)
#                 try:
#                     index = self.total_list.tolist().index([lat,lon])
#                 except ValueError:
#                     print('There was a value error in the total list')
#                     continue
#                 n_s[i,k] = north_south[:,index].mean()
#                 e_w[i,k] = east_west[:,index].mean()
#                 if ellipses:
#                     y = north_south[:,index].data
#                     x = east_west[:,index].data
#                     mask = (x!=0)|(y!=0) 
#                     x = x[mask]
#                     y = y[mask]
#                     try:
#                         w,v = np.linalg.eig(np.cov(x,y))
#                     except:
#                         continue
#                     angle = np.degrees(np.arctan(v[1,np.argmax(w)]/v[0,np.argmax(w)]))+90
#                     axis1 = np.log(2*max(w)*np.sqrt(5.991))

#                     axis2 = np.log(2*min(w)*np.sqrt(5.991))

#                     print('angle = ',angle)
#                     print('axis1 = ',axis1)
#                     print('axis2 = ',axis2)
#                     try:
#                         poly = m.ellipse(lon, lat, axis1*scale_factor,axis2*scale_factor, 80,phi=angle,line=False, facecolor='green', zorder=3,alpha=0.35)
#                     except ValueError:
#                         print(' there was a value error in the calculation of the transition_matrix')
#                         continue
#         if arrows:
#             e_w = np.ma.array(e_w,mask=mask)
#             n_s = np.ma.array(n_s,mask=mask)
#             mag = np.sqrt(e_w**2+n_s**2)

#             m.quiver(XX,YY,e_w/mag,n_s/mag,scale = 50)
#             return (m,e_w,n_s)

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

	def reduce_resolution(self,new_degree_bins):
		lat_mult = new_degree_bins[0]/self.degree_bins[0]
		lon_mult = new_degree_bins[1]/self.degree_bins[1]

		reduced_res_lat_bins,reduced_res_lon_bins = self.bins_generator(new_degree_bins)
		lat_bins,lon_bins = zip(*self.total_list)
		lat_idx = np.digitize(lat_bins,reduced_res_lat_bins)
		lon_idx = np.digitize(lon_bins,reduced_res_lon_bins)
		reduced_res_bins=[(reduced_res_lat_bins[i],reduced_res_lon_bins[j]) for i,j in zip(lat_idx,lon_idx)]
		reduced_res_total_list = list(Set(reduced_res_bins))
		translation_list = [reduced_res_total_list.index(x) for x in reduced_res_bins]
		check = [(np.array(translation_list)==x).sum()<=(lat_mult*lon_mult) for x in range(len(reduced_res_total_list))]
		assert all(check)
		translation_dict = dict(zip(range(len(self.total_list)),translation_list))
		
		old_row_idx,old_column_idx,old_data = scipy.sparse.find(self)
		new_row_idx = np.array([translation_dict[ii] for ii in old_row_idx])
		new_col_idx = np.array([translation_dict[ii] for ii in old_column_idx])
		out_data = []
		out_col = []
		out_row = []
		for row_dummy, col_dummy in list(Set(zip(new_row_idx,new_col_idx))):
			mask = (new_row_idx==row_dummy)&(new_col_idx==col_dummy)
			data_dummy = old_data[mask]
			assert len(data_dummy)<=(lat_mult*lon_mult)**2
#squared because of the possibilty of inter box exchange 
			out_data.append(data_dummy.sum()*1/(lat_mult*lon_mult))
			out_col.append(col_dummy)
			out_row.append(row_dummy)
		mat_dim = len(reduced_res_total_list)
		return TransPlot((out_data,(out_row,out_col
			)),shape=(mat_dim,mat_dim),total_list=reduced_res_total_list,
			lat_spacing=new_degree_bins[0],lon_spacing=new_degree_bins[1],time_step=self.time_step,number_data=None,
			traj_file_type=self.traj_file_type,rescale=True)    

def OP_eig_plot():
	def plot_the_data(plottable):
		XX,YY = np.meshgrid(bins_lon,bins_lat)
		plt.pcolormesh(XX,YY,plottable)
		# XX,YY,m = basemap_setup(bins_lat,bins_lon,'Argo') 
		# m.pcolormesh(XX,YY,plottable,vmin=vmin,vmax=vmax)
		plt.colorbar()
		# if not real_mask.all():
		#     m,e_w,n_s = TransPlot(matrix,total_list=trans_mat.total_list,lat_spacing=trans_mat.degree_bins[0]
		#         ,lon_spacing=trans_mat.degree_bins[1],traj_file_type='Argo').quiver_plot(m=m,ellipses=False,mask=mask)


	for op_type in ['OP_OneDDiffusion','OP_OneDJet','OP_OneDDoubleJet','OP_OneDDoubleJetDiffusion']:
		trans_mat = TransPlot.load_from_type(4,0.5,30,op_type)
		filename = trans_mat.make_filename(traj_type=str(trans_mat.traj_file_type),
			degree_bins=trans_mat.degree_bins,time_step=trans_mat.time_step)
		root_dir = os.path.dirname(filename)+'/'

		r_e_vals,r_e_vecs = scipy.sparse.linalg.eigs(trans_mat,k=50) 
		l_e_vals,l_e_vecs = scipy.sparse.linalg.eigs(trans_mat.T,k=50)         
		r_idx = [x for _,x in sorted(zip([abs(k) for k in r_e_vals],range(len(r_e_vals))))]
		r_idx = r_idx[::-1]
		l_idx = [x for _,x in sorted(zip([abs(k) for k in l_e_vals],range(len(l_e_vals))))]
		l_idx = l_idx[::-1]
		plt.plot(r_e_vals[r_idx])
		plt.savefig(root_dir+op_type+'e_val_spectrum_r')
		plt.close()
		plt.plot(l_e_vals[l_idx])
		plt.savefig(root_dir+op_type+'e_val_spectrum_l')
		plt.close()

		y,x = zip(*trans_mat.total_list)
		bins_lat = sorted(np.unique(y)) 
		bins_lon = sorted(np.unique(x)) 
		# bins_lat,bins_lon = trans_mat.bins_generator(trans_mat.degree_bins)
		for n in range(len(r_idx)):
			r_e_vec = r_e_vecs[:,r_idx[n]]
			r_e_val = r_e_vals[r_idx[n]]
			l_e_vec = l_e_vecs[:,l_idx[n]]
			l_e_val = l_e_vals[l_idx[n]]

			# matrix_component = r_e_val*np.outer(r_e_vec,r_e_vec)/(r_e_vec**2).sum()
			# matrix_component = matrix_component/abs(matrix_component).max()
			
			eig_vec_token = transition_vector_to_plottable(bins_lat,bins_lon,trans_mat.total_list,r_e_vec.real)
			# eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
			# eig_vec_token_holder = np.ma.masked_inside(eig_vec_token,-mag_mask,mag_mask)
			# mag = eig_vec_token_holder.data.max()
			# real_mask = eig_vec_token_holder.mask
			plt.subplot(2,2,1)
			plot_the_data(eig_vec_token)
			plt.title('real right evec')


			eig_vec_token = transition_vector_to_plottable(bins_lat,bins_lon,trans_mat.total_list,r_e_vec.imag)
			# eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
			# eig_vec_token_holder = np.ma.masked_inside(eig_vec_token,-mag_mask,mag_mask)
			# mag = eig_vec_token_holder.data.max()
			# imag_mask = eig_vec_token_holder.mask
			plt.subplot(2,2,2)
			plot_the_data(eig_vec_token)
			plt.title('imag right evec')

			
			eig_vec_token = transition_vector_to_plottable(bins_lat,bins_lon,trans_mat.total_list,l_e_vec.real)
			# eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
			# eig_vec_token_holder = np.ma.masked_inside(eig_vec_token,-mag_mask,mag_mask)
			# mag = eig_vec_token_holder.data.max()
			# real_mask = eig_vec_token_holder.mask
			plt.subplot(2,2,3)
			plot_the_data(eig_vec_token)
			plt.title('real left evec')

			eig_vec_token = transition_vector_to_plottable(bins_lat,bins_lon,trans_mat.total_list,l_e_vec.imag)
			# eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
			# eig_vec_token_holder = np.ma.masked_inside(eig_vec_token,-mag_mask,mag_mask)
			# mag = eig_vec_token_holder.data.max()
			# imag_mask = eig_vec_token_holder.mask

			plt.subplot(2,2,4)
			plot_the_data(eig_vec_token)
			plt.title('imag left evec')
			plt.suptitle(r_e_val)
			plt.savefig(str(int(trans_mat.degree_bins[0]))+'_'+str(int(trans_mat.degree_bins[1]))+'_'+str(trans_mat.traj_file_type)+'transition_'+str(n))
			plt.close()

def five_year_density_plot():
	holder = TransPlot.load_from_type(1,1,180)
	trans_mat = holder.multiply(9,0.0000001)
	plottable = trans_mat.dot(scipy.sparse.csc_matrix(np.ones([trans_mat.shape[0],1]))) 
	bins_lat,bins_lon = trans_mat.bins_generator(trans_mat.degree_bins)
	XX,YY,m = basemap_setup(bins_lat,bins_lon,trans_mat.traj_file_type)
	plottable = transition_vector_to_plottable(bins_lat,bins_lon,trans_mat.total_list,plottable.todense())
	plottable = np.ma.masked_array(plottable,mask=abs(plottable)<10**-6)
	m.pcolormesh(XX,YY,plottable*100)
	plt.colorbar(label='Relative Chance of Aggregation (%)')
	plt.savefig('future_aggreation')
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
	traj_class = TransPlot.load_from_type(4,4,180) 
	float_vector = SOCCOM.recent_floats(traj_class.degree_bins,traj_class.total_list)
	bins_lat,bins_lon = traj_class.bins_generator(traj_class.degree_bins)
	plt.figure(figsize=(20,10))
	plt.subplot(1,2,1)
	XX,YY,m = basemap_setup(bins_lat,bins_lon,traj_class.traj_file_type)
	plottable = transition_vector_to_plottable(bins_lat,bins_lon,traj_class.total_list,traj_class.dot(float_vector).todense())
	traj_class.traj_file_type = 'SOSE'
	XX,YY,m = basemap_setup(bins_lat,bins_lon,traj_class.traj_file_type)  
	m.pcolormesh(XX,YY,np.ma.masked_less(plottable,5*10**-3),cmap=plt.cm.YlGn,vmax = plottable.max()/2)
	plt.colorbar(label='Probability Density/Age')
	m = float_vector.scatter_plot(m=m)
	plt.annotate('A', xy = (0.17,0.9),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.subplot(1,2,2)
	traj_class.traj_file_type = 'Argo'
	float_vector = Argo.recent_floats(traj_class.degree_bins,traj_class.total_list)
	XX,YY,m = basemap_setup(bins_lat,bins_lon,traj_class.traj_file_type)
	plottable = transition_vector_to_plottable(bins_lat,bins_lon,traj_class.total_list,traj_class.dot(float_vector).todense())
	XX,YY,m = basemap_setup(bins_lat,bins_lon,traj_class.traj_file_type)  
	m.pcolormesh(XX,YY,np.ma.masked_less(plottable,5*10**-3),cmap=plt.cm.YlGn,vmax = plottable.max()/2)
	plt.colorbar(label='Probability Density/Age')
	m = float_vector.scatter_plot(m=m)
	plt.annotate('B', xy = (0.17,0.9),xycoords='axes fraction',zorder=10,size=32,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig('/Users/pchamberlain/Projects/transition_matrix_paper/plots/survival')
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

def gps_argos_compare():
	from transition_matrix.compute.compute_utils import matrix_size_match
	argos_class = TransPlot.load_from_type(2,2,180,'argo_argos') 
	argos_class.abs_mean_and_dispersion()
	gps_class = TransPlot.load_from_type(2,2,180,'argo_gps')
	gps_class.abs_mean_and_dispersion()
	argos, gps = matrix_size_match(gps_class,argos_class) 
	difference_data = argos - gps
	difference = TransPlot(difference_data,shape = argos.shape,total_list=argos.total_list,lat_spacing=argos.degree_bins[1],lon_spacing=argos.degree_bins[0],time_step=argos.time_step,
		traj_file_type='argos_gps_difference') 
	difference.save()
	difference.abs_mean_and_dispersion()

def season_compare():
	winter_class = TransMatrix(degree_bin_lat=2,degree_bin_lon=3,date_span_limit=100,traj_file_type='Argo',season = [11,12,1,2])
	summer_class = TransMatrix(degree_bin_lat=2,degree_bin_lon=3,date_span_limit=100,traj_file_type='Argo',season = [5,6,7,8])
	winter_class, summer_class = matrix_size_match(winter_class,summer_class)
	matrix_difference_compare(winter_class.matrix.transition_matrix,summer_class.matrix.transition_matrix)




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

def bias_plot():
	from transition_matrix.compute.compute_utils import matrix_size_match    
	for traj_type in ['SOSE','argo']:
		error_list = []
		for lat,lon in [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]:
			for time, multiplyer in [(30,11),(60,5),(90,3),(120,2)]:
				holder_low_res = TransPlot.load_from_type(lon,lat,time,traj_type=traj_type)
				holder_low_res = holder_low_res.multiply(multiplyer)        
				holder_high_res = TransPlot.load_from_type(lon,lat,180,traj_type=traj_type)
				holder_high_res = holder_high_res.multiply(1)        

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

				holder_low_res = TransPlot.load_from_type(lon,lat,time)
				bins_lat,bins_lon = holder_low_res.bins_generator(holder_low_res.degree_bins)
				XX,YY,m = basemap_setup(bins_lat,bins_lon,holder_high_res.traj_file_type)
				plottable = transition_vector_to_plottable(bins_lat,bins_lon,holder_high_res.total_list,e_w_diff)
				m.pcolormesh(XX,YY,np.ma.masked_array(plottable,mask=np.abs(plottable)>(e_w_diff.mean()+2*e_w_diff.std())))
				plt.colorbar()
				plt.savefig(holder_low_res.plot_folder()+'e-w-diff_high_temp_res')
				plt.close()

				XX,YY,m = basemap_setup(bins_lat,bins_lon,holder_high_res.traj_file_type)
				plottable = transition_vector_to_plottable(bins_lat,bins_lon,holder_high_res.total_list,n_s_diff)
				m.pcolormesh(XX,YY,np.ma.masked_array(plottable,mask=np.abs(plottable)>(n_s_diff.mean()+2*n_s_diff.std())))
				plt.colorbar()
				plt.savefig(holder_low_res.plot_folder()+'n-s-diff_high_temp_res')
				plt.close()

				XX,YY,m = basemap_setup(bins_lat,bins_lon,holder_high_res.traj_file_type)
				plottable = transition_vector_to_plottable(bins_lat,bins_lon,holder_high_res.total_list,total_diff)
				m.pcolormesh(XX,YY,np.ma.masked_array(plottable,mask=np.abs(plottable)>(total_diff.mean()+2*total_diff.std())))
				plt.colorbar()
				plt.savefig(holder_low_res.plot_folder()+'total-diff_high_temp_res')
				plt.close()
				error_list.append((lat,lon,time,total_diff.mean(),total_diff.std()))
		start_idx = (np.array(range(6)))*4
		end_idx = (np.array(range(6))+1)*4
		fig,ax = plt.subplots()
		for ii,kk in zip(start_idx,end_idx):
			holder = error_list[ii:kk]
			label = str((holder[0][0],holder[0][1]))
			dummy,dummy,time,mean,std = zip(*holder)
			mean = np.array(mean)
			# std = np.array(std)
			# ax.fill_between(time,mean-std,mean+std,alpha=0.2)
			ax.plot(time,mean)
			ax.scatter(time,mean,label=label)
		plt.legend()
		plt.xlabel('Timestep (days)')
		plt.ylabel('Mean Error')
		plt.savefig('/Users/pchamberlain/Projects/transition_matrix_paper/plots/'+traj_type+'bias_compare.jpg')

def mean_data(matrix):
	return matrix.sum(axis=0)/np.array([matrix[:,x].count_nonzero() for x in range(matrix.shape[0])])


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


def SOSE_compare():
	from transition_matrix.compute.compute_utils import matrix_size_match
	holder = TransPlot.load_from_type(2,2,120,traj_type='argo')
	holder1 = TransPlot.load_from_type(2,2,120,traj_type='SOSE')
	sose,argo = matrix_size_match(holder,holder1)
	plot = TransPlot(argo-sose,total_list=sose.total_list,lat_spacing=2,lon_spacing=2,time_step=120,traj_file_type='SOSE')
	self = plot
	row_list, column_list, data_array = scipy.sparse.find(self)
	self.get_direction_matrix()
	bins_lat,bins_lon = self.bins_generator(self.degree_bins)
	XX,YY = np.meshgrid(bins_lon,bins_lat)


	east_west_data = self.east_west[row_list,column_list]*data_array
	north_south_data = self.north_south[row_list,column_list]*data_array
	east_west = self.new_sparse_matrix(east_west_data)
	north_south = self.new_sparse_matrix(north_south_data)

	eastwest_mean = []
	northsouth_mean = []
	for k in range(self.shape[0]):
		eastwest_mean.append(east_west[:,k].data.mean())
		northsouth_mean.append(north_south[:,k].data.mean())



	quiver_e_w = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,np.array(eastwest_mean)*self.degree_bins[0]*111/(self.time_step*24))
	quiver_n_s = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,np.array(northsouth_mean)*self.degree_bins[0]*111/(self.time_step*24))


	std_number = 1/4.

	quiver_e_w = np.ma.masked_greater(quiver_e_w,np.mean(eastwest_mean)+std_number*np.std(eastwest_mean))
	quiver_e_w = np.ma.masked_less(quiver_e_w,np.mean(eastwest_mean)-std_number*np.std(eastwest_mean))

	quiver_n_s = np.ma.masked_greater(quiver_n_s,np.mean(northsouth_mean)+std_number*np.std(northsouth_mean))
	quiver_n_s = np.ma.masked_less(quiver_n_s,np.mean(northsouth_mean)-std_number*np.std(northsouth_mean))

	scale_factor = 2
	dummy,dummy,m = basemap_setup(bins_lon[::scale_factor],bins_lat[::scale_factor],self.traj_file_type,fill_color=False)  
	mXX,mYY = m(XX[::scale_factor,::scale_factor],YY[::scale_factor,::scale_factor])
	q = m.quiver(mXX,mYY,quiver_e_w[::scale_factor,::scale_factor],quiver_n_s[::scale_factor,::scale_factor],scale = .6)
	qk= plt.quiverkey (q,0.5, 1.02, 0.02, '0.02 km/hr', labelpos='N')
	plt.savefig('/Users/pchamberlain/Projects/transition_matrix_paper/plots/mean_sose_difference.jpg')
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



# base = '/Users/pchamberlain/Projects/transition_matrix/output/aoml/'
# for token in ['60-[2.0, 2.0].npz']:   
#     for location in ['aoml']:
#         i = TransPlot.load(base+token)
#         i.traj_file_type = location
#         i.number_plot()
#         i.transition_matrix_plot()
#         i.standard_error_plot()

# for token in [(2,2),(2,3),(3,3)]:
#     lat,lon = token
#     for date in [180]:
#         traj_class = argo_traj_plot(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=date)
#         traj_class.plot_cm26_cor_ellipse('co2')