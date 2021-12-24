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
from GeneralUtilities.Plot.Cartopy.eulerian_plot import GlobalCartopy
from 

plt.rcParams['font.size'] = '16'
file_handler = FilePathHandler(ROOT_DIR,'transition_matrix_plot')


class TransPlot(TransMat):
	def __init__(self, *args,**kwargs):
		super().__init__(*args,**kwargs)





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
  