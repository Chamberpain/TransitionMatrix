import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap,shiftgrid
from scipy.sparse import csr_matrix
import matplotlib.colors
import sys,os
sys.path.append(os.path.abspath("../"))
import soccom_proj_settings
import oceans
import datetime
import scipy

"compiles and compares transition matrix from trajectory data. "


class argo_traj_data:
	def __init__(self,degree_bins=2,date_span_limit=40):
		self.degree_bins = degree_bins
		self.date_span_limit = date_span_limit
		self.bins_lat = np.arange(-90,90.1,self.degree_bins).tolist()
		self.bins_lon = np.arange(-180,180.1,self.degree_bins).tolist()
		self.X,self.Y = np.meshgrid(self.bins_lon,self.bins_lat)

		self.df = pd.read_pickle('/Users/paulchamberlain/Data/global_argo_traj')
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
			print 'i could not load the transition df, I am recompiling with degree step size', self.degree_bins,
			self.recompile_transition_df()
		try:
			self.transition_matrix = np.load('transition_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.dat')
		except IOError:
			print 'i could not load the transition df, I am recompiling with degree step size', self.degree_bins,' and time step ',self.date_span_limit
			self.recompile_transition_matrix()
		try:
			self.w = np.load("w.dat")
		except:
			print 'could not load w'
			pass

	def recompile_transition_df(self,dump=True):
		"""
		from self.df, calculate the dataframe that is used to create the transition matrix

		input: dump - logical variable that is used to determine whether to save the dataframe
		"""
		start_bin_list = []
		final_bin_list = []
		date_span_list = []
		final_date_list = []
		position_type_list = []
		k = len(self.df.Cruise.unique())
		for n,cruise in enumerate(self.df.Cruise.unique()):
		    print 'cruise is ',cruise,'there are ',k-n,' cruises remaining'
		    print 'start bin list is ',len(start_bin_list),' long'
		    mask = self.df.Cruise==cruise
		    df_holder = self.df[mask]
		    df_holder['diff']= (df_holder.bins_lat.diff().apply(lambda x: 1 if abs(x) else 0)+df_holder.bins_lon.diff().apply(lambda x: 1 if abs(x) else 0)).cumsum()
		    position_type = df_holder['Position Type'].values[0]
		    for g in df_holder.groupby('diff').groups:
		        df_token = df_holder.groupby('diff').get_group(g)
		        if df_holder[df_holder.index==(df_token.index.values[-1] + 1)].empty:  #this is the case where the float died in the grid cell
		            continue
		        else:
		            date_span = df_token.Date.values[-1]-df_token.Date.values[0]
		            start_bin = df_token.bin_index.values[0]
		            final_bin = df_holder[df_holder.index==(df_token.index.values[-1] + 1)].bin_index.values[0]
		            final_date = df_holder[df_holder.index==(df_token.index.values[-1] + 1)].Date.values[0]
		            date_span_list.append(date_span)
		            start_bin_list.append(start_bin)
		            final_bin_list.append(final_bin)
		            final_date_list.append(final_date)
		            position_type_list.append(position_type)
		            if len(position_type_list)!=len(start_bin_list):
		            	print 'There is a bug in the code'
		            	raise


		date_span_list = [abs(x.astype('timedelta64[D]')/np.timedelta64(1, 'D')) for x in date_span_list]
		self.df_transition = pd.DataFrame({'date':final_date_list,'position type':position_type_list,'start bin':start_bin_list,'date span':date_span_list,'end bin':final_bin_list})
		if dump:
			self.df_transition.to_pickle('transition_df_degree_bins_'+str(self.degree_bins)+'.pickle')

	def recompile_transition_matrix(self,dump=True):
		self.transition_matrix = np.identity(len(self.total_list))
		self.number_list = np.zeros([len(self.total_list)])
		k = len(self.df_transition['start bin'].unique())
		for n,ii in enumerate(self.df_transition['start bin'].unique()):
		    print 'made it through ',n,' bins. ',(k-n),' remaining'
		    ii_index = self.total_list.index(list(ii))
		    date_span_addition = 0 
		    frame = self.df_transition[self.df_transition['start bin']==ii]
		    self.number_list[ii_index]=len(frame)
		    frame_cut = frame[frame['date span']<=self.date_span_limit]
		    while frame_cut.empty:
		    	date_span_addition += 10
		    	frame_cut = frame[frame['date span']<=(date_span_limit+ date_span_addition)]
		    	print 'we need to increase the time delta, because we have an error'
		    	print 'we have increased the date_span addition by ',date_span_addition
		    self.transition_matrix[ii_index,ii_index]=(len(frame)-len(frame_cut))/float(len(frame))
		    for qq in frame_cut['end bin'].unique():
		        qq_index = self.total_list.index(list(qq))
		        self.transition_matrix[qq_index,ii_index]=len(frame_cut[frame_cut['end bin']==qq])/float(len(frame))
		    if abs(sum(self.transition_matrix[:,ii_index])-1)>0.01 :
		        print 'Something went wrong and the transition doesnt add up'
		        print sum(self.transition_matrix[:,ii_index])
		        raise
		if dump:
			self.transition_matrix.dump('transition_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.dat')

	def recompile_w(self,number):
		self.w = csr_matrix(self.transition_matrix)
		for n in range(number):
		    print n
		    self.w = self.w.dot(self.w)
		self.w = self.w.todense()	
		self.w.dump('w.dat')


	def number_plot_process(self):
		self.plot_array_ = np.zeros([len(self.bins_lon),len(self.bins_lat)])
		for ii in self.df.bins_lat.unique():
		    print 'working on lat ',ii
		    i_index = self.bins_lat.index(ii)
		    for jj in self.df.bins_lon.unique():
		        j_index = self.bins_lon.index(jj)
		        self.plot_array_[j_index,i_index] = self.df[(self.df.bins_lon==jj)&(self.df.bins_lat==ii)].Lat.count()
		self.plot_array_.dump('number_plot_array.dat')

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
		m.contourf(XX,YY,self.plot_array_.T,levels=levs,cmap=plt.cm.magma)
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
		m.contourf(XX,YY,transition_plot,levels=levs) # this is a plot for the tendancy of the residence time at a grid cell
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

		    m.contourf(XX,YY,float_result) # this is a plot for the tendancy of the residence time at a grid cell    
		    lat,lon = self.total_list[ii]
		    x,y = m(lon,lat)
		    m.plot(x,y,'y*',markersize=20)
		    plt.title('1 year particle deployment at '+str(lat)+' lat,'+str(lon)+' lon',size=30)
		plt.show()

	def gps_argos_compare(self,recompile=False):
		if recompile:
			df = self.df_transition
			df_argos = df[df['position type']=='ARGOS']
			self.df_transition = df_argos
			self.recompile_transition_matrix(dump=False)
			transition_matrix_argos = self.transition_matrix
			argos_number_list = self.transition_vector_to_plottable(self.number_list)
			df_gps = df[df['position type']=='GPS']
			self.df_transition = df_gps
			self.recompile_transition_matrix(dump=False)
			transition_matrix_gps = self.transition_matrix
			gps_number_list = self.transition_vector_to_plottable(self.number_list)

			transition_matrix_argos.dump('transition_matrix_argos')
			argos_number_list.dump('argos_number_list')
			transition_matrix_gps.dump('transition_matrix_gps')
			gps_number_list.dump('gps_number_list')

		else:
			transition_matrix_argos = np.load('transition_matrix_argos')
			argos_number_list = np.load('argos_number_list')
			transition_matrix_gps = np.load('transition_matrix_gps')
			gps_number_list = np.load('gps_number_list')

		transition_matrix_plot = transition_matrix_argos-transition_matrix_gps
		# transition_matrix_plot[transition_matrix_plot>0.25]=0.25
		# transition_matrix_plot[transition_matrix_plot<-0.25]=-0.25
		trans_max = abs(transition_matrix_plot).max()
		k = np.diagonal(transition_matrix_plot)
		transition_plot = self.transition_vector_to_plottable(k)
		transition_plot = np.ma.array(transition_plot,mask=(gps_number_list==0)|(argos_number_list==0))
		plt.figure(figsize=(10,10))
		m = Basemap(projection='cyl',fix_aspect=False)
		m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		m.drawmapboundary(fill_color='grey')
		XX,YY = m(self.X,self.Y)
		levs = np.linspace(-trans_max,trans_max,50)
		m.contourf(XX,YY,transition_plot,levels=levs,cmap=plt.cm.seismic) # this is a plot for the tendancy of the residence time at a grid cell
		plt.colorbar(label='% particles dispersed')
		plt.title('Dataset Difference (GPS - ARGOS)',size=30)
		plt.savefig('dataset_difference.png')
		plt.show()


	def seasonal_compare(self,recompile=False):
		if recompile:
			df = self.df_transition
			df_winter = df[df.date.dt.month.isin([11,12,1,2])]
			self.df_transition = df_winter
			self.recompile_transition_matrix(dump=False)
			transition_matrix_winter = self.transition_matrix
			winter_number_list = self.transition_vector_to_plottable(self.number_list)
			df_summer = df[df.date.dt.month.isin([5,6,7,8])]
			self.df_transition = df_summer
			self.recompile_transition_matrix(dump=False)
			transition_matrix_summer = self.transition_matrix
			summer_number_list = self.transition_vector_to_plottable(self.number_list)

			transition_matrix_winter.dump('transition_matrix_winter.dat')
			winter_number_list.dump('winter_number_list.dat')
			transition_matrix_summer.dump('transition_matrix_summer.dat')
			summer_number_list.dump('summer_number_list')

		else:
			transition_matrix_winter = np.load('transition_matrix_winter.dat')
			winter_number_list = np.load('winter_number_list.dat')
			transition_matrix_summer = np.load('transition_matrix_summer.dat')
			summer_number_list = np.load('summer_number_list')

		transition_matrix_plot = transition_matrix_winter-transition_matrix_summer
		trans_max = abs(transition_matrix_plot).max()

		# transition_matrix_plot[transition_matrix_plot>0.5]=0.5
		# transition_matrix_plot[transition_matrix_plot<-0.5]=-0.5

		k = np.diagonal(transition_matrix_plot)
		transition_plot = self.transition_vector_to_plottable(k)
		transition_plot = np.ma.array(transition_plot,mask=(winter_number_list==0)|(summer_number_list==0))
		plt.figure(figsize=(10,10))
		m = Basemap(projection='cyl',fix_aspect=False,)
		m.fillcontinents(color='coral',lake_color='aqua')
		m.drawcoastlines()
		m.drawmapboundary(fill_color='grey')
		XX,YY = m(self.X,self.Y)
		levs = np.linspace(-trans_max,trans_max,50)
		m.contourf(XX,YY,transition_plot,levels=levs,cmap=plt.cm.seismic) # this is a plot for the tendancy of the residence time at a grid cell
		plt.colorbar(label='% particles dispersed')
		plt.title('Seasonal Difference (Summer - Winter)',size=30)
		plt.savefig('seasonal_difference.png')
		plt.show()

	def find_nearest(self,array,value):
	    idx = (np.abs(array-value)).argmin()
	    return idx

	def get_latest_soccom_float_locations(self,plot=False):
		df = pd.read_pickle(soccom_proj_settings.soccom_drifter_file)
		df['Lon']=oceans.wrap_lon180(df['Lon'])
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
			# try: 
			indexes.append(self.total_list.index(list(x)))
			float_vector[indexes[-1],0]=1
			# except ValueError:
			# 	continue
		float_result = self.w.dot(float_vector)
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
		float_result = self.transition_vector_to_plottable(float_result.reshape(len(self.total_list)).tolist()[0])
		float_result = np.log(np.ma.array(float_result,mask=(float_result<0.001)))
		XX,YY = t(self.X,self.Y)
		t.contourf(XX,YY,float_result,cmap=plt.cm.Greens)
		plt.title('SOCCOM Sampling PDF')
		plt.savefig('soccom_sampling.png')
		plt.show()

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
		m.contourf(XX,YY,co2_plot,cmap=plt.cm.PRGn) # this is a plot for the tendancy of the residence time at a grid cell
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

for degree_stepsize in np.arange(0.5,5,0.5):
	for time_stepsize in np.arange(10,80,5):
		m = argo_traj_data(degree_bins=degree_stepsize,date_span_limit=time_stepsize)