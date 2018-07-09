import pandas as pd
import numpy as np
import pickle
import matplotlib
# matplotlib.use('agg') #server
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap,shiftgrid
from matplotlib.patches import Polygon
from mpl_toolkits.basemap import pyproj
import matplotlib.colors
import sys,os
import glob
from collections import OrderedDict
# sys.path.append(os.path.abspath("../"))
# import soccom_proj_settings
# import oceans
import pickle
import datetime
import scipy.sparse
import scipy.optimize
from itertools import groupby  
from matplotlib.colors import LogNorm
from scipy.interpolate import griddata
import random
import math

"""compiles and compares transition matrix from trajectory data. """
class Basemap(Basemap):
    def ellipse(self, x0, y0, a, b, n, phi=0, ax=None, **kwargs):
        """
        Draws a polygon centered at ``x0, y0``. The polygon approximates an
        ellipse on the surface of the Earth with semi-major-axis ``a`` and 
        semi-minor axis ``b`` degrees longitude and latitude, made up of 
        ``n`` vertices.

        For a description of the properties of ellipsis, please refer to [1].

        The polygon is based upon code written do plot Tissot's indicatrix
        found on the matplotlib mailing list at [2].

        Extra keyword ``ax`` can be used to override the default axis instance.

        Other \**kwargs passed on to matplotlib.patches.Polygon

        RETURNS
            poly : a maptplotlib.patches.Polygon object.

        REFERENCES
            [1] : http://en.wikipedia.org/wiki/Ellipse


        """
        ax = kwargs.pop('ax', None) or self._check_ax()
        g = pyproj.Geod(a=self.rmajor, b=self.rminor)
        # Gets forward and back azimuths, plus distances between initial
        # points (x0, y0)
        azf, azb, dist = g.inv([x0, x0], [y0, y0], [x0+a, x0], [y0, y0+b])
        tsid = dist[0] * dist[1] # a * b

        # Initializes list of segments, calculates \del azimuth, and goes on 
        # for every vertex
        seg = []
        AZ = np.linspace(azf[0], 360. + azf[0], n)
        for i, az in enumerate(AZ):
            # Skips segments along equator (Geod can't handle equatorial arcs).
            if np.allclose(0., y0) and (np.allclose(90., az) or
                np.allclose(270., az)):
                continue

            # In polar coordinates, with the origin at the center of the 
            # ellipse and with the angular coordinate ``az`` measured from the
            # major axis, the ellipse's equation  is [1]:
            #
            #                           a * b
            # r(az) = ------------------------------------------
            #         ((b * cos(az))**2 + (a * sin(az))**2)**0.5
            #
            # Azymuth angle in radial coordinates and corrected for reference
            # angle.
            azr = 2. * np.pi / 360. * (phi+az + 90.)
            A = dist[0] * np.sin(azr)
            B = dist[1] * np.cos(azr)
            r = tsid / (B**2. + A**2.)**0.5
            lon, lat, azb = g.fwd(x0, y0, az, r)
            x, y = self(lon, lat)

            # Add segment if it is in the map projection region.
            if x < 1e20 and y < 1e20:
                seg.append((x, y))
        # print seg
        poly = Polygon(seg, **kwargs)
        ax.add_patch(poly)

        # Set axes limits to fit map region.
        self.set_axes_limits(ax=ax)

        return poly

        
def find_nearest(items, pivot):
	return min(items, key=lambda x: abs(x - pivot))

def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape']).T #this transpose is very important because otherwise these transition matrices load wrong and send flow the wrong way.

__file__ = os.getenv("HOME")+'/Projects/transition_matrix/transition_matrix_compute.py' #this is a hack to get this thing to run in terminal
bad_cruise_list = ['1900857','1900978','1901149','1901552','3900743','3901062','4901246','5903883','6900938','6901004']

class argo_traj_data:
	def __init__(self,degree_bins=1,date_span_limit=60):
		print 'I have started argo traj data'
		self.base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/'

		self.degree_bins = float(degree_bins)
		self.date_span_limit = date_span_limit
		self.bins_lat = np.arange(-90,90.1,self.degree_bins).tolist()
		self.bins_lon = np.arange(-180,180.1,self.degree_bins).tolist()
		if 180.0 not in self.bins_lon:
			print '180 is not divisable by the degree bins chosen'		#need to add this logic for the degree bin choices that do not end at 180. 
			raise
		self.X,self.Y = np.meshgrid(self.bins_lon,self.bins_lat)
		self.df = pd.read_pickle(self.all_argo_traj_df_string).sort_values(by=['Cruise','Date'])
		self.df = self.df[~self.df.Cruise.isin(bad_cruise_list)]
		self.df['bins_lat'] = pd.cut(self.df.Lat,bins = self.bins_lat,labels=self.bins_lat[:-1])
		self.df['bins_lon'] = pd.cut(self.df.Lon,bins = self.bins_lon,labels=self.bins_lon[:-1])
		self.df = self.df.dropna(subset=['bins_lon','bins_lat']) # get rid of binned values outside of the domain
		self.df = self.df.reset_index(drop=True)
		self.df['bin_index'] = zip(self.df['bins_lat'].values,self.df['bins_lon'].values)

		#implement tests of the dataset so that all values are within the known domain
		assert self.df.Lon.min() >= -180
		assert self.df.Lon.max() <= 180
		assert self.df.Lat.max() <= 90
		assert self.df.Lat.min() >=-90

		try:
			self.df_transition = pd.read_pickle(self.base_file+'transition_df_degree_bins_'+str(degree_bins)+'.pickle')
			assert (self.df_transition['date span']>=0).all() #require time to go in the right direction

		except IOError: #this is the case that the file could not load
			print 'i could not load the transition df, I am recompiling with degree step size', self.degree_bins,''
			self.recompile_transition_df()
		self.end_bin_string = 'end bin '+str(self.date_span_limit)+' day' # we compute the transition df for many different date spans, here is where we find that column
		self.df_transition = self.df_transition.dropna(subset=[self.end_bin_string])
		self.identify_problems_df_transition()
		while len(self.df_transition)!=len(self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]):	#need to loop this
			self.df_transition = self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]
		self.total_list = [list(x) for x in self.df_transition['start bin'].unique()] 

		try: # try to load the transition matrix
			self.transition_matrix = load_sparse_csr(self.base_file+'transition_matrix_degree_bins_'+str(degree_bins)+'_time_step_'+str(date_span_limit)+'.npz')
		except IOError: # if the matrix cannot load, recompile
			print 'i could not load the transition matrix, I am recompiling with degree step size', self.degree_bins,' and time step ',self.date_span_limit
			self.recompile_transition_matrix()

		# self.transition_eig_vals = np.linalg.eigvals(self.transition_matrix.todense())
		# if self.transition_eig_vals[self.transition_eig_vals==0]:
		# 	print 'ding ding ding ding'
		# 	print '*******************'
		# 	print 'there are zero values in the transition matrix'
		# 	raise

	def recompile_transition_df(self):
		"""
		from self.df, calculate the dataframe that is used to create the transition matrix

		input: dump - logical variable that is used to determine whether to save the dataframe
		"""
		cruise_list = []
		start_bin_list = []
		final_bin_list = []
		end_bin_list = []
		date_span_list = []
		start_date_list = []
		position_type_list = []
		time_delta_list = np.arange(20,300,20)
		k = len(self.df.Cruise.unique())
		for n,cruise in enumerate(self.df.Cruise.unique()):
			print 'cruise is ',cruise,'there are ',k-n,' cruises remaining'
			print 'start bin list is ',len(start_bin_list),' long'
			mask = self.df.Cruise==cruise  		#pick out the cruise data from the df
			df_holder = self.df[mask]
			max_date = df_holder.Date.max()
			df_holder['diff']= (df_holder.bins_lat.diff().apply(lambda x: 1 if abs(x) else 0)+df_holder.bins_lon.diff().apply(lambda x: 1 if abs(x) else 0)).cumsum()
			position_type = df_holder['position type'].values[0]
			group_library = df_holder.groupby('diff').groups
			diff_group = np.sort(df_holder.groupby('diff').groups.keys()).tolist()
			diff_group.pop() #this is the last grid box the float moved into and we can assume it died
			for g in diff_group:
				df_token = df_holder[df_holder.index.isin(group_library[g])]
				start_date = df_token.Date.min()
				cruise_list.append(cruise)
				start_date_list.append(start_date)
				start_bin_list.append(df_token.bin_index.values[0])
				position_type_list.append(position_type)
				location_tuple = []
				for time_addition in [datetime.timedelta(days=x) for x in time_delta_list]:
					final_date = start_date+time_addition
					if final_date>max_date:
						location_tuple.append(np.nan)
					else:
						location_tuple.append(df_holder[df_holder.Date==find_nearest(df_holder.Date,final_date)].bin_index.values[0]) # find the nearest date and record the bin index
				final_bin_list.append(tuple(location_tuple))
				
				try:
					advance = df_holder[df_holder.index==(df_token.tail(1).index.values[0]+1)]
					date_span_list.append((advance.Date.min()-df_token.Date.min()).days)
					end_bin_list.append(advance['bin_index'].values[0])
				except IndexError:
					date_span_list.append(np.nan)
					end_bin_list.append(np.nan)
				if any(len(lst) != len(position_type_list) for lst in [start_date_list, start_bin_list, end_bin_list, date_span_list, final_bin_list]):
					print 'position list length ,',len(position_type_list)
					print 'start date list ',len(start_date_list)
					print 'start bin list ',len(start_bin_list)
					print 'end bin list ',len(end_bin_list)
					print 'date span list ',len(final_bin_list)
					raise
		df_dict = {}
		df_dict['Cruise']=cruise_list
		df_dict['position type']=position_type_list
		df_dict['date'] = start_date_list
		df_dict['start bin'] = start_bin_list
		for time_delta,bin_list in zip(time_delta_list,zip(*final_bin_list)):
			bins_string = 'end bin '+str(time_delta)+' day'
			df_dict[bins_string]=bin_list
		df_dict['end bin'] = end_bin_list
		df_dict['date span'] = date_span_list
		self.df_transition = pd.DataFrame(df_dict)
		self.df_transition.to_pickle(self.base_file+'transition_df_degree_bins_'+str(degree_bins)+'.pickle')

	def identify_problems_df_transition(self,plot=False):
		degree_max = self.date_span_limit/3.
		lat1,lon1 = zip(*self.df_transition['start bin'].values)
		lat2,lon2 = zip(*self.df_transition[self.end_bin_string].values)
		lat_diff = abs(np.array(lat1)-np.array(lat2))
		lon_diff = abs(np.array(lon1)-np.array(lon2))
		lon_diff[lon_diff>180] = abs(lon_diff[lon_diff>180]-360)
		distance = np.sqrt((lon_diff*np.cos(lat1))**2+lat_diff**2)
		self.df_transition['distance_check'] = distance 
		cruise_list = self.df_transition[self.df_transition.distance_check>degree_max].Cruise.unique()
		if plot:
			for cruise in cruise_list:
				df_holder = self.df[self.df.Cruise==cruise]
				df_holder1 = self.df_transition[self.df_transition.Cruise==cruise]
				x = df_holder.Lon.values
				y = df_holder.Lat.values
				plt.plot(x,y)
				plt.scatter(x,y)
				plt.title('Max distance is '+str(df_holder1.distance_check.max()))
				plt.show()
		self.df_transition = self.df_transition[~self.df_transition.Cruise.isin(cruise_list)]


	def recompile_transition_matrix(self,dump=True):
		num_list = []
		num_list_index = []
		data_list = []
		row_list = []
		column_list = []

		k = len(self.df_transition['start bin'].unique())
		for n,ii in enumerate(self.df_transition['start bin'].unique()):
			print 'made it through ',n,' bins. ',(k-n),' remaining'
			ii_index = self.total_list.index(list(ii))
			frame = self.df_transition[self.df_transition['start bin']==ii]	# data from of floats that start in looped grid cell
			frame_cut = frame[frame[self.end_bin_string]!=ii].dropna(subset=[self.end_bin_string]) #this is all floats that go out of the looped grid cell
			num_list_index.append(ii_index) # we only need to save this once, because we are only concerned about the diagonal
			num_list.append(len(frame_cut)) # this is where we save the data density of every cell
			if not frame_cut.empty:
				print 'the frame cut was not empty'
				test_list = []
				row_list.append(ii_index)	#compute the on diagonal elements
				column_list.append(ii_index)	#compute the on diagonal elemnts
				data = (len(frame)-len(frame_cut))/float(len(frame)) # this is the percentage of floats that stay in the looped grid cell
				data_list.append(data)
				test_list.append(data)
				print 'total number of row slots is ',len(frame_cut[self.end_bin_string].unique())
				for qq in frame_cut[self.end_bin_string].unique():
					qq_index = self.total_list.index(list(qq))
					row_list.append(qq_index)
					column_list.append(ii_index)
					data = (len(frame_cut[frame_cut[self.end_bin_string]==qq]))/float(len(frame))
					data_list.append(data)
					test_list.append(data)
			else:
				print 'the frame cut was empy, so I will calculate the scaled dispersion'
				test_list = []
				diagonal = 1
				for qq in frame['end bin'].unique():
					try:
						qq_index = self.total_list.index(list(qq))
					except ValueError:
						print qq,' was not in the list'
						continue
					frame_holder = frame[frame['end bin']==qq]
					percentage_of_floats = len(frame_holder)/float(len(frame))
					frame_holder['time step percent'] = self.date_span_limit/frame_holder['date span']
					frame_holder[frame_holder['time step percent']>1]=1
					off_diagonal = len(frame_holder)/float(len(frame))*frame_holder['time step percent'].mean()
					diagonal -= off_diagonal
					row_list.append(qq_index)
					column_list.append(ii_index)
					data_list.append(off_diagonal)
					test_list.append(off_diagonal)
					print off_diagonal
				print diagonal
				row_list.append(ii_index)
				column_list.append(ii_index)
				data_list.append(diagonal)
				test_list.append(diagonal)
			assert abs(sum(test_list)-1)<0.01	#ensure that all columns scale to 1
			assert ~np.isnan(data_list).any()
			assert (np.array(data_list)<=1).all()
			assert (np.array(data_list)>=0).all()
		
		self.transition_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(self.total_list),len(self.total_list)))
		# self.transition_matrix = self.add_noise(self.transition_matrix)
		self.number_matrix = scipy.sparse.csc_matrix((num_list,(num_list_index,num_list_index)),shape=(len(self.total_list),len(self.total_list)))
		if dump:
			assert (np.abs(self.transition_matrix.todense().sum(axis=0)-1)<0.01).all() #this is in the dump subroutine for the case that it is recompiled for the data withholding experiment.
			save_sparse_csr(self.base_file+'transition_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',self.transition_matrix)
			save_sparse_csr(self.base_file+'number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',self.number_matrix)


	def data_withholding_routine(self,test_df):
		pdf_list = []
		cruise_len = len(test_df.Cruise.unique())
		for t,cruise in enumerate(test_df.Cruise.unique()):
			print cruise_len-t,' cruises remaining'
			token = test_df[test_df.Cruise==cruise]
			token['days from start'] = (token.Date-token.Date.min()).dt.days
			try:
				start_index= self.total_list.index(list(token.bin_index.values[0]))
			except ValueError:
				pdf_list+=[0,0,0,0,0,0]
				print 'i have incountered at error, adding zeros to the pdf list'
			for k in range(7)[1:]:
				self.load_w(k)
				pdf = self.w[:,start_index].todense()
				pdf_mean = pdf[pdf>0].mean()
				delta_days = find_nearest(token['days from start'].values,self.date_span_limit*k) 
				assert delta_days>=0
				try:
					end_index = self.total_list.index(list(token[token['days from start']==delta_days].bin_index.values[0]))
					pdf_list.append(pdf[end_index].item()/pdf_mean)
				except ValueError:
					pdf_list.append(0)
					print 'i have encountered an error, adding a zero to the list'
		return pdf_list

	def z_test(self,p_1,p_2,n_1,n_2):
		p_1 = np.ma.array(p_1,mask = (n_1==0))
		n_1 = np.ma.array(n_1,mask = (n_1==0))
		p_2 = np.ma.array(p_2,mask = (n_2==0))
		n_2 = np.ma.array(n_2,mask = (n_2==0))		
		z_stat = (p_1-p_2)/np.sqrt(self.transition_matrix.todense()*(1-self.transition_matrix.todense())*(1/n_1+1/n_2))
		assert (np.abs(z_stat)<1.96).data.all()

	def data_withholding_setup(self):
		self.delete_w()
		mylist = traj_class.df_transition.Cruise.unique().tolist()
		not_mylist = [mylist[i] for i in sorted(random.sample(xrange(len(mylist)), int(round(len(mylist)/20)))) ]
		df_holder = self.df[self.df.Cruise.isin(not_mylist)]
		null_hypothesis = self.data_withholding_routine(df_holder)

		self.delete_w()
		self.df_transition = self.df_transition[~self.df_transition.Cruise.isin(not_mylist)]
		self.df_transition = self.df_transition.reset_index(drop=True)
		while len(self.df_transition)!=len(self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]):	#need to loop this
			self.df_transition = self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]
		self.recompile_transition_matrix(dump=False)
		data_withheld = self.data_withholding_routine(df_holder)
		plt.hist(null_hypothesis,bins=50,density=True,color='r',alpha=0.5,label='Null Hypothesis')
		plt.hist(data_withheld,bins=50,density=True,color='b',alpha=0.5,label='Data Withheld')
		plt.legend()
		plt.xlabel('Normalized Probability')
		plt.title('Data Withholding Experiment Comparison')
		plt.show()

	def add_noise(self,matrix,noise=0.05):
		print 'adding matrix noise'
		east_west,north_south = self.get_direction_matrix()
		direction_mat = -abs(east_west)**2-abs(north_south)**2
		direction_mat = noise*np.exp(direction_mat)
		direction_mat[direction_mat<noise*np.exp(-20)]=0
		matrix += scipy.sparse.csc_matrix(direction_mat)
		return self.rescale_matrix(matrix)

	def rescale_matrix(self,matrix_):
		print 'rescaling the matrix'
		mat_sum = matrix_.todense().sum(axis=0)
		scalefactor,dummy = np.meshgrid(1/mat_sum,1/mat_sum)
		return scipy.sparse.csc_matrix(mat_sum*scalefactor)


	def delete_w(self):
		for w in glob.glob(os.path.dirname(os.path.realpath(__file__))+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_*.npz'):
			os.remove(w)


	def load_w(self,number,dump=True):

		print 'in w matrix, current number is ',number
		try:
			self.w = load_sparse_csr(os.path.dirname(os.path.realpath(__file__))+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz')
			print 'w matrix successfully loaded'
		except IOError:
			print 'w matrix could not be loaded and will be recompiled'
			if number == 0:
				self.w = self.transition_matrix 
			else:
				self.load_w(number-1,dump=True)		# recursion to get to the first saved transition matrix 
				self.w = self.w.dot(self.transition_matrix)		# advance the transition matrix once
			 	# self.w = self.rescale_matrix(self.w)

			if dump:
				save_sparse_csr(os.path.dirname(os.path.realpath(__file__))+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz',self.w)	#and save
				# self.diagnose_matrix(self.w,os.path.dirname(os.path.realpath(__file__))+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.png')

	def transition_vector_to_plottable(self,vector):
		plottable = np.zeros([len(self.bins_lat),len(self.bins_lon)])
		for n,tup in enumerate(self.total_list):
			ii_index = self.bins_lon.index(tup[1])
			qq_index = self.bins_lat.index(tup[0])
			plottable[qq_index,ii_index] = vector[n]
		return plottable

	def df_to_plottable(self,df_):
		plottable = np.zeros([len(self.bins_lat),len(self.bins_lon)])
		k = len(df_.bin_index.unique())
		for n,tup in enumerate(df_.bin_index.unique()):
			print k-n, 'bins remaining'
			ii_index = self.bins_lon.index(tup[1])
			qq_index = self.bins_lat.index(tup[0])
			plottable[qq_index,ii_index] = len(df_[(df_.bin_index==tup)])
		return plottable		

	def argo_dense_plot(self):
		ZZ = self.df_to_plottable(self.df)
		plt.figure(figsize=(10,10))
		m = Basemap(projection='cyl',fix_aspect=False)
		# m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		XX,YY = m(self.X,self.Y)
		ZZ = np.ma.masked_equal(ZZ,0)
		m.pcolormesh(XX,YY,ZZ,vmin=0,vmax=4000,cmap=plt.cm.magma)
		plt.title('Profile Density',size=30)
		plt.colorbar(label='Number of float profiles')

	def dep_number_plot(self):
		frames = []
		k = len(self.df.Cruise.unique())
		for n,cruise in enumerate(self.df.Cruise.unique()):
			print k-n, 'cruises remaining'
			frames.append(self.df[self.df.Cruise==cruise].head(1))
		ZZ = self.df_to_plottable(pd.concat(frames))
		plt.figure(figsize=(10,10))
		m = Basemap(projection='cyl',fix_aspect=False)
		# m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		XX,YY = m(self.X,self.Y)
		ZZ = np.ma.masked_equal(ZZ,0)
		m.pcolormesh(XX,YY,ZZ,vmin=0,vmax=30,cmap=plt.cm.magma)
		plt.title('Deployment Density',size=30)
		plt.colorbar(label='Number of floats deployed')


	def trans_number_matrix_plot(self):	
		try:
			self.number_matrix = load_sparse_csr('./number_matrix_data/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
		except IOError:
			print 'the number matrix could not be loaded'
			self.recompile_transition_matrix(dump=True)
		k = np.diagonal(self.number_matrix.todense())
		number_matrix_plot = self.transition_vector_to_plottable(k)
		plt.figure(figsize=(10,10))
		m = Basemap(projection='cyl',fix_aspect=False)
		# m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		XX,YY = m(self.X,self.Y)
		# number_matrix_plot[number_matrix_plot>1000]=1000
		number_matrix_plot = np.ma.masked_equal(number_matrix_plot,0)
		m.pcolormesh(XX,YY,number_matrix_plot,vmin=0,vmax=80,cmap=plt.cm.magma)
		plt.title('Transition Density',size=30)
		plt.colorbar(label='Number of float transitions')
		plt.savefig('./number_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
		plt.close()

	def transition_matrix_plot(self,filename,load_number_matrix=True):
		if load_number_matrix:
			self.number_matrix = load_sparse_csr('./number_matrix_data/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')			
		plt.figure(figsize=(10,10))
		k = np.diagonal(self.transition_matrix.todense())
		transition_plot = self.transition_vector_to_plottable(k)
		m = Basemap(projection='cyl',fix_aspect=False)
		# m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		XX,YY = m(self.X,self.Y)
		transition_plot = np.ma.array((1-transition_plot),mask=self.transition_vector_to_plottable(np.diagonal(self.number_matrix.todense()))==0)
		m.pcolormesh(XX,YY,transition_plot,vmin=0,vmax=1) # this is a plot for the tendancy of the residence time at a grid cell
		plt.colorbar(label='% particles dispersed')
		plt.title('1 - diagonal of transition matrix',size=30)
		plt.savefig('./transition_plots/'+filename+'_diag_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
		plt.close()
		self.diagnose_matrix(self.transition_matrix,'./transition_plots/'+filename+'_diagnose_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')


	def matrix_compare(self,matrix_a,matrix_b,num_matrix_a,num_matrix_b,title,save_name): #this function accepts sparse matrices

		vmax = 0.4
		transition_matrix_plot = matrix_a-matrix_b
		# transition_matrix_plot[transition_matrix_plot>0.25]=0.25
		# transition_matrix_plot[transition_matrix_plot<-0.25]=-0.25
		k = np.diagonal(transition_matrix_plot.todense())

		transition_plot = self.transition_vector_to_plottable(k)
		num_matrix_a = self.transition_vector_to_plottable(np.diagonal(num_matrix_a.todense()))
		num_matrix_b = self.transition_vector_to_plottable(np.diagonal(num_matrix_b.todense()))
		transition_plot = np.ma.array(transition_plot,mask=(num_matrix_a==0)|(num_matrix_b==0)|np.isnan(transition_plot))
		print 'maximum of comparison is ', transition_plot.max()
		print 'minimum of comparison is ', transition_plot.min()

		plt.figure(figsize=(10,10))
		m = Basemap(projection='cyl',fix_aspect=False)
		# m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		m.drawmapboundary(fill_color='grey')
		XX,YY = m(self.X,self.Y)
		m.pcolormesh(XX,YY,transition_plot,vmin=-vmax,vmax=vmax,cmap=plt.cm.seismic) # this is a plot for the tendancy of the residence time at a grid cell
		plt.colorbar(label='% particles dispersed')
		plt.title(title,size=30)
		plt.savefig(save_name)
		plt.close()


	def gps_argos_compare(self):
		try:
			transition_matrix_argos = load_sparse_csr('./argos_gps_data/transition_matrix_argos_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			number_matrix_argos = load_sparse_csr('./number_matrix_data/number_matrix_argos_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			transition_matrix_gps = load_sparse_csr('./argos_gps_data/transition_matrix_gps_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			number_matrix_gps = load_sparse_csr('./number_matrix_data/number_matrix_gps_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
		except IOError:
			print 'the gps and argos transition matrices could not be loaded and will be recompiled'
			df = self.df_transition
			df_argos = df[df['position type']=='ARGOS']
			self.df_transition = df_argos
			self.recompile_transition_matrix(dump=False)
			transition_matrix_argos = self.transition_matrix
			number_matrix_argos = self.number_matrix
			df_gps = df[df['position type']=='GPS']
			self.df_transition = df_gps
			self.recompile_transition_matrix(dump=False)
			transition_matrix_gps = self.transition_matrix
			number_matrix_gps = self.number_matrix
			save_sparse_csr('./argos_gps_data/transition_matrix_argos_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',transition_matrix_argos)
			save_sparse_csr('./number_matrix_data/number_matrix_argos_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',number_matrix_argos)
			save_sparse_csr('./argos_gps_data/transition_matrix_gps_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',transition_matrix_gps)
			save_sparse_csr('./number_matrix_data/number_matrix_gps_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',number_matrix_gps)
		self.matrix_compare(transition_matrix_argos,transition_matrix_gps,number_matrix_argos,number_matrix_gps,'Dataset Difference (GPS - ARGOS)','./dataset_difference/dataset_difference_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
		self.diagnose_matrix(transition_matrix_argos,'./dataset_difference/argos_diagnose_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
		self.diagnose_matrix(transition_matrix_gps,'./dataset_difference/gps_diagnose_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
		self.z_test(transition_matrix_gps.todense(),transition_matrix_argos.todense(),number_matrix_gps.todense(),number_matrix_argos.todense())



	def seasonal_compare(self):
		print 'I am now comparing the seasonal nature of the dataset'
		try:
			transition_matrix_summer = load_sparse_csr('./sum_wint_data/transition_matrix_summer_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			number_matrix_summer = load_sparse_csr('./number_matrix_data/number_matrix_summer_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			transition_matrix_winter = load_sparse_csr('./sum_wint_data/transition_matrix_winter_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			number_matrix_winter = load_sparse_csr('./number_matrix_data/number_matrix_winter_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
		except IOError:
			print 'the summer and winter transition matrices could not be loaded and will be recompiled'
			df = self.df_transition
			df_winter = df[df.date.dt.month.isin([11,12,1,2])]
			self.df_transition = df_winter
			self.recompile_transition_matrix(dump=False)
			transition_matrix_winter = self.transition_matrix
			number_matrix_winter = self.number_matrix
			df_summer = df[df.date.dt.month.isin([5,6,7,8])]
			self.df_transition = df_summer
			self.recompile_transition_matrix(dump=False)
			transition_matrix_summer = self.transition_matrix
			number_matrix_summer = self.number_matrix

			save_sparse_csr('./sum_wint_data/transition_matrix_winter_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',transition_matrix_winter)
			save_sparse_csr('./number_matrix_data/number_matrix_winter_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',number_matrix_winter)
			save_sparse_csr('./sum_wint_data/transition_matrix_summer_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',transition_matrix_summer)
			save_sparse_csr('./number_matrix_data/number_matrix_summer_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',number_matrix_summer)

		self.transition_matrix=transition_matrix_summer
		self.number_matrix=number_matrix_summer
		self.transition_matrix_plot('summer',load_number_matrix=False)
		self.transition_matrix=transition_matrix_winter
		self.number_matrix=number_matrix_winter
		self.transition_matrix_plot('winter',load_number_matrix=False)
		self.matrix_compare(transition_matrix_winter,transition_matrix_summer,number_matrix_winter,number_matrix_summer,'Seasonal Difference (Summer - Winter)','./seasonal/seasonal_difference_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
		self.z_test(transition_matrix_summer.todense(),transition_matrix_winter.todense(),number_matrix_summer.todense(),number_matrix_winter.todense())

	def get_direction_matrix(self):
		lat_list, lon_list = zip(*self.total_list)
		lon_max = 180/self.degree_bins
		east_west = []
		north_south = []
		k = len(self.total_list)
		for n,item in enumerate(self.total_list):
			print 'made it through ',n,' there are ',(k-n),' remaining'
			lat,lon = item
			e_w = (np.array(lon_list)-lon)/self.degree_bins
			e_w[e_w>lon_max]=e_w[e_w>lon_max]-2*lon_max
			e_w[e_w<-lon_max]=e_w[e_w<-lon_max]+2*lon_max
			east_west.append(e_w)
			north_south.append((np.array(lat_list)-lat)/self.degree_bins)

		east_west = np.array(east_west).T  #this is because np.array makes lists of lists go the wrong way
		north_south = np.array(north_south).T
		return (east_west,north_south)

	def quiver_plot(self,matrix):
		east_west,north_south = self.get_direction_matrix()
		trans_mat = matrix.todense()
		# print trans_mat.shape
		# print east_west.shape
		# print north_south.shape
		# e_w_max = max(abs(east_west[trans_mat!=0]))
		# n_s_max = max(abs(north_south[trans_mat!=0]))
		# print 'the farthest east-west distance is', e_w_max
		# print 'the farthest north-south distance is', n_s_max
		# # assert e_w_max < 20
		# # assert n_s_max < 20
		east_west = np.multiply(trans_mat,east_west)
		north_south = np.multiply(trans_mat,north_south)
		# print np.sum(east_west,axis=0).tolist()[0]
		# print np.sum(east_west,axis=0).shape

		m = Basemap(projection='cyl',fix_aspect=False)
		# m.fillcontinents(color='coral',lake_color='aqua')


		m.drawcoastlines()
		Y = np.arange(-68,66,4)
		X = np.arange(-162,162,4)
		XX,YY = np.meshgrid(X,Y)
		n_s = np.zeros(XX.shape)
		e_w = np.zeros(XX.shape)
		XX,YY = m(XX,YY)
		for i,lat in enumerate(Y):
			for k,lon in enumerate(X):
				print 'lat = ',lat
				print 'lon = ',lon 
				try:
					index = self.total_list.index([lat,lon])
				except ValueError:
					print 'There was a value error in the total list'
					continue
				n_s[i,k] = north_south[:,index].mean()
				e_w[i,k] = east_west[:,index].mean()
				y = north_south[:,index]
				x = east_west[:,index]
				mask = (x!=0)|(y!=0) 
				x = x[mask]
				y = y[mask]
				try: 
					w,v = np.linalg.eig(np.cov(x,y))
				except:
					continue
				angle = np.degrees(np.arctan(v[1,np.argmax(w)]/v[0,np.argmax(w)]))
				axis1 = 2*max(w)*np.sqrt(5.991)
				axis2 = 2*min(w)*np.sqrt(5.991)
				print 'angle = ',angle
				print 'axis1 = ',axis1
				print 'axis2 = ',axis2
				if axis1>0.5: 
					print 'Axis 1 is too big'
					continue
				if axis2>0.1: 
					print 'Axis 2 is too big'
					continue

				try:
					poly = m.ellipse(lon, lat, axis1*20,axis2*20, 80,phi=angle, facecolor='green', zorder=3,alpha=0.35)
				except ValueError:
					continue
		# east_west = self.transition_vector_to_plottable(np.sum(east_west,axis=0).tolist()[0])
		# north_south = self.transition_vector_to_plottable(np.sum(north_south,axis=0).tolist()[0])
		# self.number_matrix = load_sparse_csr('./number_matrix_data/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')


		# number_mask = self.transition_vector_to_plottable(np.diagonal(self.number_matrix.todense()))==0
		# east_west = np.ma.array(east_west,mask=number_mask)
		# north_south = np.ma.array(north_south,mask=number_mask)
		# m.quiver(XX,YY,east_west,north_south,scale=25) # this is a plot for the tendancy of the residence time at a grid cell
		e_w = np.ma.array(e_w,mask=(e_w==0))
		n_s = np.ma.array(n_s,mask=(n_s==0))
		m.quiver(XX,YY,e_w*5000,n_s*5000,scale=25)
		plt.title('Variance Ellipses of Transition Matrix')

	def diagnose_matrix(self,matrix,filename):
		plt.figure(figsize=(10,10))
		plt.subplot(2,2,1)
		plt.spy(matrix)
		plt.subplot(2,2,2)
		self.quiver_plot(matrix)
		plt.subplot(2,2,3)
		row,col = zip(*np.argwhere(matrix))  
		plt.hist([len(list(group)) for key, group in groupby(np.sort(col))],log=True)
		plt.title('histogram of how many cells are difused into')
		plt.subplot(2,2,4)
		plt.hist(matrix.data[~np.isnan(matrix.data)],log=True)
		plt.title('histogram of transition matrix weights')
		plt.savefig(filename)
		plt.close()

	def get_latest_soccom_float_locations(self,plot=False,individual = False):
		"""
		This function gets the latest soccom float locations and returns a vector of the pdf after it has been multiplied by w
		"""

		try:
			self.float_vector = np.load('soccom_initial_degree_bins_'+str(self.degree_bins)+'.npy')
			dummy,indexes = zip(*np.argwhere(self.float_vector))
		except IOError:
			print 'initial soccom locations could not be loaded and need to be recompiled'
			df = pd.read_pickle('soccom_all.pickle')
			# df['Lon']=oceans.wrap_lon180(df['Lon'])
			mask = df.Lon>180
			df.loc[mask,'Lon']=df[mask].Lon-360
			frames = []
			for cruise in df.Cruise.unique():                            
				df_holder = df[df.Cruise==cruise]
				frame = df_holder[df_holder.Date==df_holder.Date.max()].drop_duplicates(subset=['Lat','Lon'])
				if (frame.Date>(df.Date.max()-datetime.timedelta(days=30))).any():  
					frames.append(frame)
				else:
					continue
			df = pd.concat(frames)
			lats = [find_nearest(self.bins_lat,x) for x in df.Lat.values]
			lons = [find_nearest(self.bins_lon,x) for x in df.Lon.values]
			indexes = []
			self.float_vector = np.zeros([1,len(self.total_list)])
			for x in zip(lats,lons):
				try: 
					indexes.append(self.total_list.index(list(x)))
					self.float_vector[0,indexes[-1]]=1
				except ValueError:	# this is for the case that the starting location of the the soccom float is not in the transition matrix
					print 'I have incountered a value error and cannot continue...'
					
					raise
			np.save('soccom_initial_degree_bins_'+str(self.degree_bins)+'.npy',self.float_vector)


		if individual:
			indexes = [indexes[individual]]
			self.float_vector = np.zeros([1,len(self.total_list)])
			self.float_vector[0,indexes[-1]]=1
		float_vector = scipy.sparse.csc_matrix(self.float_vector)
		float_result = self.w.dot(scipy.sparse.csr_matrix(float_vector.T))
		t = []

		if plot:
			if individual:
				lat,lon = self.total_list[indexes[-1]]
				print lat
				print lon 
				lynne_coord =[-70,50,-80,-32]
				lllat=lynne_coord[2]
				urlat=lynne_coord[3]
				lllon=lynne_coord[0]
				urlon=lynne_coord[1]
				lon_0=0
				t = Basemap(projection='mill',llcrnrlat=lllat,urcrnrlat=urlat,llcrnrlon=lllon,urcrnrlon=urlon,resolution='i',lon_0=lon_0,fix_aspect=False)
			else:
				t = Basemap(projection='cyl',lon_0=0,fix_aspect=False,)
			# t.fillcontinents(color='coral',lake_color='aqua')
			t.drawcoastlines()
			for index in indexes:
				lat,lon = self.total_list[index]
				x,y = t(lon,lat)
				t.plot(x,y,'b*',markersize=14)
		return t,float_result.todense()


	def plot_latest_soccom_locations(self,debug = False,individual=False):
		plt.figure()
		t,float_result_sparse = self.get_latest_soccom_float_locations(plot=True,individual=individual)
		float_result = self.transition_vector_to_plottable(float_result_sparse.todense().reshape(len(self.total_list)).tolist()[0])

		# float_result = np.log(np.ma.array(float_result,mask=(float_result<0.001)))
		plot_max = float_result.max()
		plot_min = plot_max-3*float_result.std()
		# float_result = np.ma.array(float_result,mask=(float_result<plot_min))

		XX,YY = t(self.X,self.Y)
		t.pcolormesh(XX,YY,float_result,vmax=0.2,vmin=0,cmap=plt.cm.Greens)
		plt.title('SOCCOM Sampling PDF')
		if debug:
			plt.figure()
			plt.hist(float_result_sparse.data[~np.isnan(float_result_sparse.data)],log=True)
			plt.show()
		plt.savefig('./soccom_plots/soccom_sampling_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(num)+'.png')
		plt.close()

	def pco2_mag_plot(self):
		df = pd.read_csv('../eulerian_plot/basemap/data/sumflux_2006c.txt', skiprows = 51,sep=r"\s*")
		y = np.sort(df.LAT.unique())
		x = np.sort(df.LON.unique())
		XC,YC = np.meshgrid(x,y)
		CO2 = np.zeros([len(y),len(x)])
		di = df.iterrows()
		for i in range(len(df)):
			row = next(di)[1]
			CO2[(row['LON']==XC)&(row['LAT']==YC)] = row['DELTA_PCO2']
		CO2, x = shiftgrid(180, CO2, x, start=False)
		x = x+5
		x[0] = -180
		x[-1] = 180
		co2_vector = np.zeros([len(self.total_list),1])
		for n,(lat,lon) in enumerate(self.total_list):
			lon_index = find_nearest(lon,x)
			lat_index = find_nearest(lat,y)
			co2_vector[n] = CO2[lat_index,lon_index]
		co2_plot = abs(self.transition_vector_to_plottable(co2_vector))
		plt.figure()
		m = Basemap(projection='cyl',fix_aspect=False)
		# m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		XX,YY = m(self.X,self.Y)

		m.pcolormesh(XX,YY,co2_plot,cmap=plt.cm.PRGn) # this is a plot for the tendancy of the residence time at a grid cell
		plt.colorbar(label='CO2 Flux $gm C/m^2/yr$')



	def get_optimal_float_locations(self,target_vector,desired_vector_in_plottable=True):
		""" accepts a target vecotr in sparse matrix form, returns the ideal float locations and weights to sample that"""

		# self.w[self.w<0.007]=0
		target_vector = abs(target_vector/target_vector.max()) # normalize the target_vector
		m,soccom_float_result = self.get_latest_soccom_float_locations(plot=True)
		float_result = soccom_float_result/soccom_float_result.max()
		target_vector = target_vector-float_result
		target_vector = np.array(target_vector)
		print type(self.w)
		print type(target_vector)
		#desired_vector, residual = scipy.optimize.nnls(np.matrix(self.w.todense()),np.squeeze(target_vector))
		optimize_fun = scipy.optimize.lsq_linear(self.w,np.squeeze(target_vector),bounds=(0,self.degree_bins),verbose=2)
		desired_vector = optimize_fun.x

		y,x = zip(*np.array(self.total_list)[desired_vector>0.1])
		if desired_vector_in_plottable:
			return (m,x,y,self.transition_vector_to_plottable(desired_vector))
		else:
			return (m,x,y,desired_vector, target_vector)

	def cm2p6(self,filename):
		x = np.load('xt_ocean')
		y = np.load('yt_ocean')
		field = np.load(filename)
		shifted_field,x = shiftgrid(-180,field,x,start=True)
		field_vector = np.zeros([len(self.total_list),1])
		for n,(lat,lon) in enumerate(self.total_list):
			lon_index = x.tolist().index(find_nearest(x,lon))
			lat_index = y.tolist().index(find_nearest(y,lat))

			field_vector[n] = shifted_field[lat_index,lon_index]
		return field_vector

	def pco2_var_plot(self,cost_plot=False):
		plt.figure()
		if cost_plot:
			plt.subplot(2,1,1)
		field_vector = self.cm2p6('mean_pco2.dat')
		field_plot = abs(self.transition_vector_to_plottable(field_vector))
		m,x,y,desired_vector,target_vector =  self.get_optimal_float_locations(field_vector,desired_vector_in_plottable=(not cost_plot)) #if cost_plot, the vector will return desired vector in form for calculations
		XX,YY = m(self.X,self.Y)
		m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Purples) # this is a plot for the tendancy of the residence time at a grid cell
		m.scatter(x,y,marker='*',color='g',s=34)
		if cost_plot:
			float_number = []
			variance_misfit = []
			total_var = np.sum(target_vector.dot(target_vector.T))
			for limit in np.arange(1 ,0,-0.01):
				print 'limit is ',limit
				
				token_vector = np.zeros(len(desired_vector))
				token_vector[desired_vector>limit] = [math.ceil(k) for k in desired_vector[desired_vector>limit]]
				float_number.append(token_vector.sum())
				token_result = target_vector - self.w.todense().dot(np.matrix(token_vector).T)
				token_result[token_result<0] = 0
				variance_misfit.append(100-np.sum(token_result.dot(token_result.T))/total_var*100)
				print 'variance misfit is ',variance_misfit[-1]
			plt.subplot(2,1,2)
			plt.plot(float_number,variance_misfit)
			plt.ylabel('pCO$^2$ Variance Constrained (%)')
			plt.xlabel('Number of Additional Floats Deployed')
		plt.show()

		
	def o2_var_plot(self,line=([20, -55], [-30, -15])):
		floor = 10**-11
		plt.figure()
		if line:
			plt.subplot(2,1,1)
		field_vector = self.cm2p6('mean_o2.dat')
		field_vector[field_vector<floor]=floor
		field_plot = abs(self.transition_vector_to_plottable(field_vector))
		m,x,y,desired_vector =  self.get_optimal_float_locations(field_vector)
		XX,YY = m(self.X,self.Y)
		m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Reds) # this is a plot for the tendancy of the residence time at a grid cell
		m.scatter(x,y,marker='*',color='g',s=34)
		if line:
			lat,lon = line
			x,y = m(lon,lat)
			m.plot(x,y,'o-')
			plt.subplot(2,1,2)
			lat = np.linspace(lat[0],lat[1],50)
			lon = np.linspace(lon[0],lon[1],50)
			points = np.array(zip(lat,lon))
			grid_z0 = griddata((self.X.flatten(),self.Y.flatten()),desired_vector.flatten(),points,method='linear')
			plt.plot((lat-lat.min())*111,grid_z0)
			plt.xlabel('Distance Along Cruise Track (km)')
			plt.ylabel('Variance Constrained')
		plt.show()

	def hybrid_var_plot(self):
		plt.figure()
		field_vector_pco2 = self.cm2p6('mean_pco2.dat')
		field_vector_o2 = self.cm2p6('mean_o2.dat')
		field_vector_pco2[field_vector_pco2>(10*field_vector_pco2.std()+field_vector_pco2.mean())]=0
		field_vector = field_vector_pco2/(2*field_vector_pco2.max())+field_vector_o2/(2*field_vector_o2.max())
		field_plot = abs(self.transition_vector_to_plottable(field_vector))

		m,x,y,desired_vector = self.get_optimal_float_locations(field_vector)
		XX,YY = m(self.X,self.Y)
		m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Greens) # this is a plot for the tendancy of the residence time at a grid cell
		m.scatter(x,y,marker='*',color='r',s=30)		
		plt.show()

	def load_corr_matrix(self,variable):
		try:
			self.cor_matrix = load_sparse_csr(variable+'_cor_matrix_degree_bins_'+str(self.degree_bins)+'.npz')
		except IOError:
			with open(variable+"_array_list", "rb") as fp:   
				array_list = pickle.load(fp)

			with open(variable+"_position_list", "rb") as fp:
				position_list = pickle.load(fp)

			base_list,alt_list = zip(*position_list)
			base_lat,base_lon = zip(*base_list)
			alt_lat,alt_lon = zip(*alt_list)

			base_lon = np.array(base_lon); alt_lon = np.array(alt_lon)
			base_lat = np.array(base_lat); alt_lat = np.array(alt_lat)
			array_list = np.array(array_list)
			base_lon[base_lon<-180] = base_lon[base_lon<-180]+360
			alt_lon[alt_lon<-180] = alt_lon[alt_lon<-180]+360

			total_lat,total_lon = zip(*self.total_list)
			subsampled_bins_lon = [find_nearest(np.unique(base_lon), i) for i in np.unique(total_lon)]
			subsampled_bins_lat = [find_nearest(np.unique(base_lat), i) for i in np.unique(total_lat)]
			base_mask = np.isin(base_lon,subsampled_bins_lon)&np.isin(base_lat,subsampled_bins_lat)

			subsampled_bins_lon = [find_nearest(np.unique(alt_lon), i) for i in np.unique(total_lon)]
			subsampled_bins_lat = [find_nearest(np.unique(alt_lat), i) for i in np.unique(total_lat)]
			alt_mask = np.isin(alt_lon,subsampled_bins_lon)&np.isin(alt_lat,subsampled_bins_lat)

			mask = alt_mask&base_mask

			base_lon = base_lon[mask]; alt_lon = alt_lon[mask]
			base_lat = base_lat[mask]; alt_lat = alt_lat[mask]
			array_list = array_list[mask]

			base_lon = [find_nearest(self.bins_lon, i) for i in base_lon]
			base_lat = [find_nearest(self.bins_lat, i) for i in base_lat]			
			alt_lon = [find_nearest(self.bins_lon, i) for i in alt_lon]
			alt_lat = [find_nearest(self.bins_lat, i) for i in alt_lat]			

			column_list = [x for x in range(len(self.total_list))] # initialize these lists with 1 along the diagonal to represent perfect correlation in grid cell
			row_list = [x for x in range(len(self.total_list))]
			data_list = [1 for x in range(len(self.total_list))]

			for base,alt,val in zip(zip(base_lat,base_lon),zip(alt_lat,alt_lon),array_list):
				b_lat, b_lon = base
				try:
					col_temp = [self.total_list.index(list(base))]
					row_temp = [self.total_list.index(list(alt))]
					value_temp = [val]
				except ValueError:
					print 'I was unsuccessful'
					print b_lat
					print b_lon
					continue
				else:
					print 'I sucessfully added to the lists'
					print base
					print alt
					column_list += col_temp
					row_list += row_temp
					data_list += value_temp
			
			self.cor_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(self.total_list),len(self.total_list)))
			save_sparse_csr(variable+'_cor_matrix_degree_bins_'+str(self.degree_bins)+'.npz', self.cor_matrix)

	def california_plot_for_lynne(self):
		locs = pd.read_excel('../california_current_float_projection/ca_current_test_locations_2018-05-14.xlsx')
		for n,(lat,lon) in locs[['lat','lon']].iterrows():
			lon = -lon
			lat1 = find_nearest(self.bins_lat,lat)
			lon1 = find_nearest(self.bins_lon,lon)
			try:
				index = self.total_list.index([lat1,lon1])
			except ValueError:
				print 'lat and lon not in index'
				continue
			for num in np.arange(12):
				print 'num is ',num
				self.load_w(num)
				m = Basemap(llcrnrlon=-150.,llcrnrlat=21.,urcrnrlon=-115.,urcrnrlat=48,projection='cyl')
				# m.fillcontinents(color='coral',lake_color='aqua')
				m.drawcoastlines()
				XX,YY = m(self.X,self.Y)
				x,y = m(lon,lat)
				m.plot(x,y,'yo',markersize=14)
				float_vector = np.zeros([1,len(self.total_list)])

				float_vector[0,index]=1
				float_vector = scipy.sparse.csc_matrix(float_vector)
				float_result = self.w.dot(scipy.sparse.csr_matrix(float_vector.T))
				float_result = self.transition_vector_to_plottable(float_result.todense().reshape(len(self.total_list)).tolist()[0])

				XX,YY = m(self.X,self.Y)
				m.pcolormesh(XX,YY,float_result,vmax=0.05,vmin=0,cmap=plt.cm.Greens)
				plt.savefig('lynne_plot_'+str(n)+'_w_'+str(num))
				plt.close()



traj_class = argo_traj_data()
# traj_class.load_w(5)
# traj_class.pco2_var_plot(line_plot=True)
# plt.show()
# traj_class.quiver_plot(traj_class.transition_matrix)
# plt.show()




# degree_stepsize = 2
# time_stepsize = 30
# traj_class = argo_traj_data(degree_bins=degree_stepsize,date_span_limit=time_stepsize)
# num =5 
# traj_class.load_corr_matrix('pco2')
# field_vector = traj_class.cm2p6('mean_o2.dat')
# field_vector = traj_class.cor_matrix.dot(field_vector)
# field_plot = abs(traj_class.transition_vector_to_plottable(field_vector))
# m = Basemap(projection='cyl',fix_aspect=False)
# m.fillcontinents(color='coral',lake_color='aqua')
# m.drawcoastlines()
# XX,YY = m(traj_class.X,traj_class.Y)
# m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Purples) # this is a plot for the tendancy of the residence time at a grid cell
# # traj_class.load_w(num)
# # traj_class.o2_var_plot(line = ([15,-55],[-30,05]))


# if __name__ == "__main__":
# 	degree_stepsize = float(sys.argv[1])
# 	print 'degree stepsize is ',degree_stepsize
# 	time_stepsize = 30
# 	traj_class = argo_traj_data(degree_bins=degree_stepsize,date_span_limit=time_stepsize)
# 	traj_class.recompile_transition_df()
# 	# for time_stepsize in np.arange(60,300,15):
# 	# 	print 'time stepsize is ',time_stepsize
# 	# 	traj_class = argo_traj_data(degree_bins=degree_stepsize,date_span_limit=time_stepsize)
# 	# 	traj_class.transition_matrix_plot('transition_matrix')
# 	# 	plt.figure()
# 	# 	traj_class.quiver_plot(traj_class.transition_matrix)
# 	# 	plt.savefig('./transition_plots/quiver_diag_degree_bins_'+str(traj_class.degree_bins)+'_time_step_'+str(traj_class.date_span_limit)+'.png')
# 	# 	plt.close()
# 	# 	traj_class.number_matrix_plot()
# 	# 	traj_class.seasonal_compare()
# 	# 	traj_class.gps_argos_compare()
# 	for num in range(20):
# 		traj_class.load_w(num)
# 		traj_class.plot_latest_soccom_locations()