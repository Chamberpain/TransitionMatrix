import pandas as pd
import numpy as np
import pickle
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap,shiftgrid
import matplotlib.colors
import sys,os
from collections import OrderedDict
# sys.path.append(os.path.abspath("../"))
# import soccom_proj_settings
# import oceans
import pickle
import datetime
import scipy.sparse

"compiles and compares transition matrix from trajectory data. "



def save_sparse_csr(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csr(filename):
    loader = np.load(filename)
    return scipy.sparse.csr_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

class argo_traj_data:
	def __init__(self,degree_bins=2,date_span_limit=40):
		self.degree_bins = degree_bins
		self.date_span_limit = date_span_limit
		self.bins_lat = np.arange(-90,90.1,self.degree_bins).tolist()
		self.bins_lon = np.arange(-180,180.1,self.degree_bins).tolist()
		self.X,self.Y = np.meshgrid(self.bins_lon,self.bins_lat)

		self.df = pd.read_pickle('global_argo_traj')
		self.df['bins_lat'] = pd.cut(self.df.Lat,bins = self.bins_lat,labels=self.bins_lat[:-1])
		self.df['bins_lon'] = pd.cut(self.df.Lon,bins = self.bins_lon,labels=self.bins_lon[:-1])

		self.df = self.df.dropna(subset=['bins_lon','bins_lat'])
		self.df = self.df.reset_index(drop=True)
		self.df['bin_index'] = zip(self.df['bins_lat'].values,self.df['bins_lon'].values)
		self.total_list = self.df.drop_duplicates(subset=['bins_lat','bins_lon'])[['bins_lat','bins_lon']].values.tolist()

		assert self.df.Lon.min() >= -180
		assert self.df.Lon.max() <= 180
		assert self.df.Lat.max() <= 90
		assert self.df.Lat.min() >=-90

		try:
			self.df_transition = pd.read_pickle('transition_df_degree_bins_'+str(self.degree_bins)+'.pickle')
		except IOError:
			print 'i could not load the transition df, I am recompiling with degree step size', self.degree_bins,''
			self.recompile_transition_df()
		try:
			self.transition_matrix = load_sparse_csr('transition_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
		except IOError:
			print 'i could not load the transition matrix, I am recompiling with degree step size', self.degree_bins,' and time step ',self.date_span_limit
			self.recompile_transition_matrix()


	def recompile_transition_df(self,dump=True):
		"""
		from self.df, calculate the dataframe that is used to create the transition matrix

		input: dump - logical variable that is used to determine whether to save the dataframe
		"""
		start_bin_list = []
		final_bin_list = []
		start_date_list = []
		position_type_list = []
		time_delta_list = np.arange(10,150,15)
		k = len(self.df.Cruise.unique())
		for n,cruise in enumerate(self.df.Cruise.unique()[:5]):
			print 'cruise is ',cruise,'there are ',k-n,' cruises remaining'
			print 'start bin list is ',len(start_bin_list),' long'
			mask = self.df.Cruise==cruise
			df_holder = self.df[mask]
			max_date = df_holder.Date.max()
			df_holder['diff']= (df_holder.bins_lat.diff().apply(lambda x: 1 if abs(x) else 0)+df_holder.bins_lon.diff().apply(lambda x: 1 if abs(x) else 0)).cumsum()
			position_type = df_holder['Position Type'].values[0]
			diff_group = OrderedDict(df_holder.groupby('diff').groups)
			diff_group.popitem() #this is the case where the float died in the last grid cell and we eliminate it
			for g in diff_group:
				df_token = df_holder.groupby('diff').get_group(g)
				start_date = df_token.Date.min()
				start_date_list.append(start_date)
				start_bin_list.append(df_token.bin_index.values[0])
				position_type_list.append(position_type)
				location_tuple = []
				for time_addition in [datetime.timedelta(days=x) for x in time_delta_list]:
					final_date = start_date+time_addition
					if final_date>max_date:
						location_tuple.append(np.nan)
					location_tuple.append(df_holder[df_holder.Date==find_nearest(df_holder.Date,final_date)].bin_index.values[0]) # find the nearest date and record the bin index
				final_bin_list.append(tuple(location_tuple))
				if len(position_type_list)!=len(start_bin_list):
					print 'There is a bug in the code'
					raise
		df_dict = {}
		df_dict['position type']=position_type_list
		df_dict['date'] = start_date_list
		df_dict['start_bin'] = start_bin_list
		for time_delta,bin_list in zip(time_delta_list,zip(*final_bin_list)):
			bins_string = 'end bin '+str(time_delta)+' day'
			df_dict[bins_string]=bin_list
		self.df_transition = pd.DataFrame(df_dict)

		if dump:
			self.df_transition.to_pickle('transition_df_degree_bins_'+str(self.degree_bins)+'.pickle')

	def recompile_transition_matrix(self,dump=True):
		
		num_list = []
		num_list_index = []
		data_list = []
		row_list = []
		column_list = []
		end_bin_string = 'end bin '+str(self.date_span_limit)+' day'
		k = len(self.df_transition['start bin'].unique())
		for n,ii in enumerate(self.df_transition['start bin'].unique()):
			print 'made it through ',n,' bins. ',(k-n),' remaining'
			ii_index = self.total_list.index(list(ii))
			date_span_addition = 0 
			frame = self.df_transition[self.df_transition['start bin']==ii]
			num_list.append(len(frame)) # this is where we save the data density of every cell
			num_list_index.append(ii_index) # we only need to save this once, because we are only concerned about the diagonal
			frame_cut = frame[frame[end_bin_string]!=ii]
			if not frame_cut.empty:
				print 'the frame cut was not empty'
				test_list = []
				row_list.append(ii_index)
				column_list.append(ii_index)
				data = (len(frame)-len(frame_cut))/float(len(frame))
				data_list.append(data)
				test_list.append(data)
				for qq in frame_cut[end_bin_string].unique():
					qq_index = self.total_list.index(list(qq))
					row_list.append(qq_index)
					column_list.append(ii_index)
					data = (len(frame_cut[frame_cut[end_bin_string]==qq]))/float(len(frame))
					data_list.append(data)
					test_list.append(data)
				if abs(sum(test_list)-1)>0.01:
						print 'ding ding ding ding'
						print '*******************'
						print 'We have a problem'
						print sum(test_list)
						raise
			else:
				print 'the frame cut was empy, so I will calculate the scaled dispersion'
				test_list = []
				diagonal = 1
				for qq in frame[end_bin_string].unique():
					qq_index = self.total_list.index(list(qq))
					frame_holder = frame[frame[end_bin_string]==qq]
					percentage_of_floats = len(frame_holder)/float(len(frame))
					frame_holder['time step percent'] = self.date_span_limit/frame_holder['date span']
					frame_holder[frame_holder['time step percent']>1]=1
					off_diagonal = len(frame_holder)/float(len(frame))*frame_holder['time step percent'].mean()
					diagonal -= off_diagonal
					row_list.append(qq_index)
					column_list.append(ii_index)
					data_list.append(off_diagonal)
					test_list.append(off_diagonal)

				row_list.append(ii_index)
				column_list.append(ii_index)
				data_list.append(diagonal)
				test_list.append(diagonal)
				if abs(sum(test_list)-1)>0.01:
						print 'ding ding ding ding'
						print '*******************'
						print 'We have a problem'
						print sum(test_list)
						raise


		self.transition_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(self.total_list),len(self.total_list)))
		self.number_matrix = scipy.sparse.csc_matrix((num_list,(num_list_index,num_list_index)),shape=(len(self.total_list),len(self.total_list)))
		if dump:
			save_sparse_csr('transition_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',self.transition_matrix)
			save_sparse_csr('number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',self.number_matrix)

	def load_w(self,number,dump=True):
		print 'recompiling w matrix, current number is ',number
		try:
			self.w = load_sparse_csr('w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz')
		except IOError:
			if number == 0:
				self.w = self.transition_matrix 
			else:
				self.load_w(number-1,dump=True)		# recursion to get to the first saved transition matrix 
				self.w = self.w.dot(self.transition_matrix)		# advance the transition matrix once
		if dump:
			save_sparse_csr('w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz',self.w)	#and save



	def number_plot_plot(self):	
		try:
			self.plot_array_ = np.load('number_plot_array.dat')
		except IOError:
			self.number_plot_process()
		plt.figure(figsize=(10,10))
		m = Basemap(projection='cyl',fix_aspect=False)
		m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		XX,YY = m(self.X,self.Y)
		self.plot_array_[self.plot_array_>1000]=1000
		self.plot_array_ = np.ma.masked_equal(self.plot_array_,0)
		levs = np.linspace(self.plot_array_.min(),self.plot_array_.max(),50)
		m.pcolormesh(XX,YY,self.plot_array_.T,levels=levs,cmap=plt.cm.magma)
		plt.title('Data Density',size=30)
		plt.colorbar(label='Number of float tracks')
		plt.savefig('number_plot.png')
		plt.show()

	def transition_vector_to_plottable(self,vector):
		plottable = np.zeros([len(self.bins_lat),len(self.bins_lon)])
		for n,tup in enumerate(self.total_list):
			ii_index = self.bins_lon.index(tup[1])
			qq_index = self.bins_lat.index(tup[0])
			plottable[qq_index,ii_index] = vector[n]
		return plottable

	def transition_plot_array_diagonal(self):
		try:
			self.plot_array_ = np.load('number_plot_array.dat')
		except IOError:
			self.number_plot_process()
		plt.figure(figsize=(10,10))
		k = np.diagonal(self.transition_matrix)
		transition_plot = self.transition_vector_to_plottable(k)
		m = Basemap(projection='cyl',fix_aspect=False)
		m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		XX,YY = m(self.X,self.Y)
		transition_plot = np.ma.array((1-transition_plot),mask=self.plot_array_.T==0)
		trans_max = abs(transition_plot).max()
		levs = np.linspace(0,trans_max,50)
		m.pcolormesh(XX,YY,transition_plot,levels=levs) # this is a plot for the tendancy of the residence time at a grid cell
		plt.colorbar(label='% particles dispersed')
		plt.title('1 - diagonal of transition matrix',size=30)
		plt.savefig('transition_matrix_diag.png')

	def float_pdf(self):
		for ii in np.random.choice(range(len(self.total_list)),10,replace=False):
		    print ii
		    float_vector = np.zeros([len(self.total_list),1])
		    float_vector[ii,0]=1
		    print self.w.shape
		    print float_vector.shape
		    float_result = self.w.dot(float_vector)
		    plt.figure()
		    print float_result.shape
		    print float_result.reshape(len(self.total_list)).tolist()[0]
		    float_result = self.transition_vector_to_plottable(float_result.reshape(len(self.total_list)).tolist()[0])
		    m = Basemap(projection='cyl',lon_0=0)
		    m.fillcontinents(color='coral',lake_color='aqua')
		    m.drawcoastlines()
		    XX,YY = m(self.X,self.Y)

		    m.pcolormesh(XX,YY,float_result) # this is a plot for the tendancy of the residence time at a grid cell    
		    lat,lon = self.total_list[ii]
		    x,y = m(lon,lat)
		    m.plot(x,y,'y*',markersize=20)
		    plt.title('1 year particle deployment at '+str(lat)+' lat,'+str(lon)+' lon',size=30)
		plt.show()


	def matrix_compare(self,matrix_a,matrix_b,num_matrix_a,num_matrix_b,title,save_name): #this function accepts sparse matrices
		transition_matrix_plot = matrix_a-matrix_b
		# transition_matrix_plot[transition_matrix_plot>0.25]=0.25
		# transition_matrix_plot[transition_matrix_plot<-0.25]=-0.25
		trans_max = abs(transition_matrix_plot).max()
		k = np.diagonal(transition_matrix_plot.todense())
		transition_plot = self.transition_vector_to_plottable(k)
		num_matrix_a = self.transition_vector_to_plottable(np.diagonal(num_matrix_a.todense()))
		num_matrix_b = self.transition_vector_to_plottable(np.diagonal(num_matrix_b.todense()))
		transition_plot = np.ma.array(transition_plot,mask=(num_matrix_a==0)|(num_matrix_b==0))
		plt.figure(figsize=(10,10))
		m = Basemap(projection='cyl',fix_aspect=False)
		m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		m.drawmapboundary(fill_color='grey')
		XX,YY = m(self.X,self.Y)
		m.pcolormesh(XX,YY,transition_plot,vmin=-trans_max,vmax=trans_max,cmap=plt.cm.seismic) # this is a plot for the tendancy of the residence time at a grid cell
		plt.colorbar(label='% particles dispersed')
		plt.title(title,size=30)
		plt.savefig(save_name)
		plt.close()


	def gps_argos_compare(self):
		try:
			transition_matrix_argos = load_sparse_csr('transition_matrix_argos_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			number_matrix_argos = load_sparse_csr('number_matrix_argos_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			transition_matrix_gps = load_sparse_csr('transition_matrix_gps_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			number_matrix_gps = load_sparse_csr('number_matrix_gps_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
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
			save_sparse_csr('transition_matrix_argos_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',transition_matrix_argos)
			save_sparse_csr('number_matrix_argos_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',number_matrix_argos)
			save_sparse_csr('transition_matrix_gps_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',transition_matrix_gps)
			save_sparse_csr('number_matrix_gps_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',number_matrix_gps)
		self.matrix_compare(transition_matrix_argos,transition_matrix_gps,number_matrix_argos,number_matrix_gps,'Dataset Difference (GPS - ARGOS)','dataset_difference_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')



	def seasonal_compare(self):
		try:
			transition_matrix_summer = load_sparse_csr('transition_matrix_summer_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			number_matrix_summer = load_sparse_csr('number_matrix_summer_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			transition_matrix_winter = load_sparse_csr('transition_matrix_winter_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
			number_matrix_winter = load_sparse_csr('number_matrix_winter_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
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

			save_sparse_csr('transition_matrix_winter_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',transition_matrix_winter)
			save_sparse_csr('number_matrix_winter_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',number_matrix_winter)
			save_sparse_csr('transition_matrix_summer_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',transition_matrix_summer)
			save_sparse_csr('number_matrix_summer_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz',number_matrix_summer)
		self.matrix_compare(transition_matrix_winter,transition_matrix_summer,number_matrix_winter,number_matrix_summer,'Seasonal Difference (Summer - Winter)','seasonal_difference_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')


	def find_nearest(items, pivot):
		return min(items, key=lambda x: abs(x - pivot))

	def get_latest_soccom_float_locations(self,plot=False):
		"""
		This function gets the latest soccom float locations and returns a vector of thier pdf after it has been multiplied by w
		"""
		df = pd.read_pickle('soccom_all.pickle')
		# df['Lon']=oceans.wrap_lon180(df['Lon'])
		frames = []
		for cruise in df.Cruise.unique():                            
			df_holder = df[df.Cruise==cruise]
			frame = df_holder[df_holder.Date==df_holder.Date.max()].drop_duplicates(subset=['Lat','Lon'])
			if (frame.Date>(df.Date.max()-datetime.timedelta(days=30))).any():  
				frames.append(frame)
			else:
				continue
		df = pd.concat(frames)
		lats = [self.bins_lat[self.find_nearest(x,self.bins_lat)] for x in df.Lat.values]
		lons = [self.bins_lon[self.find_nearest(x,self.bins_lon)] for x in df.Lon.values]
		indexes = []
		float_vector = np.zeros([len(self.total_list),1])
		for x in zip(lats,lons):
			try: 
				indexes.append(self.total_list.index(list(x)))
				float_vector[indexes[-1],0]=1
			except ValueError:	# this is for the case that the starting location of the the soccom float is not in the transition matrix
				continue
		float_result = scipy.sparse.csr_matrix(float_vector).transpose().dot(self.w)
		t = []
		if plot:
			t = Basemap(projection='cyl',lon_0=0,fix_aspect=False,)
			t.fillcontinents(color='coral',lake_color='aqua')
			t.drawcoastlines()
			for index in indexes:
			    lat,lon = self.total_list[index]
			    x,y = t(lon,lat)
			    t.plot(x,y,'y*',markersize=10)
		return t,float_result


	def plot_latest_soccom_locations(self):
		plt.figure(figsize=(10,10))
		t,float_result = self.get_latest_soccom_float_locations(plot=True)
		float_result = self.transition_vector_to_plottable(float_result.todense().reshape(len(self.total_list)).tolist()[0])
		float_result = np.log(np.ma.array(float_result,mask=(float_result<0.001)))
		XX,YY = t(self.X,self.Y)
		t.pcolormesh(XX,YY,float_result,cmap=plt.cm.Greens)
		plt.title('SOCCOM Sampling PDF')
		plt.savefig('soccom_sampling_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(num)+'.png')

	def co2_plot(self):
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
			lon_index = self.find_nearest(lon,x)
			lat_index = self.find_nearest(lat,y)
			co2_vector[n] = CO2[lat_index,lon_index]
		co2_plot = abs(self.transition_vector_to_plottable(co2_vector))
		plt.figure()
		m = Basemap(projection='cyl',fix_aspect=False)
		m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		XX,YY = m(self.X,self.Y)
		co2_max = co2_vector.max()
		co2_vector = abs(co2_vector/co2_max) # normalize the co2_vector
		m.pcolormesh(XX,YY,co2_plot,cmap=plt.cm.PRGn) # this is a plot for the tendancy of the residence time at a grid cell
		plt.colorbar(label='CO2 Flux $gm C/m^2/yr$')

		self.w[self.w<0.007]=0
		dummy,soccom_float_result = self.get_latest_soccom_float_locations(plot=False)
		float_result = soccom_float_result/soccom_float_result.max()
		co2_vector = co2_vector-float_result
		co2_vector = co2_vector*co2_max
		co2_vector = np.array(co2_vector)

		desired_vector, residual = scipy.optimize.nnls(self.w,np.squeeze(co2_vector))
		print len(desired_vector)





		truth_list = np.array((desired_vector>0).astype(int))
		truth_list.reshape([len(self.total_list),1])

		print np.array(self.total_list)[desired_vector>0]



		y,x = zip(*np.array(self.total_list)[desired_vector>0])

		desired_plot = self.transition_vector_to_plottable(desired_vector)
		desired_plot = desired_plot
		# plt.figure()
		# m = Basemap(projection='cyl',fix_aspect=False)
		# m.fillcontinents(color='coral',lake_color='aqua')
		# m.drawcoastlines()
		# XX,YY = m(self.X,self.Y)
		# X,Y = m(x,y)
		# m.contourf(XX,YY,desired_plot) # this is a plot for the tendancy of the residence time at a grid cell
		m.scatter(x,y,marker='*',color='y',s=18)
		plt.title('Optimal Distribution of Global BGC Argo to Capture CO2 Flux')
		plt.savefig('optimal_sampling.png')
		plt.show()

if __name__ == "__main__":
	degree_stepsize = float(sys.argv[1])
	print 'degree stepsize is ',degree_stepsize
	for time_stepsize in np.arange(10,80,20):
		print 'time stepsize is ',time_stepsize
		traj_class = argo_traj_data(degree_bins=degree_stepsize,date_span_limit=time_stepsize)
		for num in range(30):
			traj_class.load_w(number=num,dump=True)
			traj_class.plot_latest_soccom_locations()