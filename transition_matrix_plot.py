from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from transition_matrix_compute import argo_traj_data,load_sparse_csr,save_sparse_csr
from transition_matrix_compute import find_nearest
import numpy as np
from scipy.interpolate import griddata
import pandas as pd
import pyproj
from matplotlib.patches import Polygon
from itertools import groupby
import pickle
import scipy

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


class argo_traj_data(argo_traj_data):

    def z_test(self,p_1,p_2,n_1,n_2):
        p_1 = np.ma.array(p_1,mask = (n_1==0))
        n_1 = np.ma.array(n_1,mask = (n_1==0))
        p_2 = np.ma.array(p_2,mask = (n_2==0))
        n_2 = np.ma.array(n_2,mask = (n_2==0))      
        z_stat = (p_1-p_2)/np.sqrt(self.transition_matrix.todense()*(1-self.transition_matrix.todense())*(1/n_1+1/n_2))
        assert (np.abs(z_stat)<1.96).data.all()

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


    def data_withholding_setup(self):
        self.delete_w()
        mylist = traj_class.df_transition.Cruise.unique().tolist()
        not_mylist = [mylist[i] for i in sorted(random.sample(xrange(len(mylist)), int(round(len(mylist)/20)))) ]
        df_holder = self.df[self.df.Cruise.isin(not_mylist)]
        null_hypothesis = self.data_withholding_routine(df_holder)

        self.delete_w()
        self.df_transition = self.df_transition[~self.df_transition.Cruise.isin(not_mylist)]
        self.df_transition = self.df_transition.reset_index(drop=True)
        while len(self.df_transition)!=len(self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]): #need to loop this
            self.df_transition = self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]
        self.recompile_transition_matrix(dump=False)
        data_withheld = self.data_withholding_routine(df_holder)
        plt.hist(null_hypothesis,bins=50,density=True,color='r',alpha=0.5,label='Null Hypothesis')
        plt.hist(data_withheld,bins=50,density=True,color='b',alpha=0.5,label='Data Withheld')
        plt.legend()
        plt.xlabel('Normalized Probability')
        plt.title('Data Withholding Experiment Comparison')
        plt.show()

    def transition_matrix_plot(self,filename,load_number_matrix=True):
        if load_number_matrix:
            self.number_matrix = load_sparse_csr(self.base_file+'transition_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')            
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
        plt.savefig(self.base_file+'transition_plots/'+filename+'_diag_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
        plt.close()
        self.diagnose_matrix(self.transition_matrix,self.base_file+'/transition_plots/'+filename+'_diagnose_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')

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
        ZZ = np.zeros([len(self.bins_lat),len(self.bins_lon)])
        series = self.df.groupby('bin_index').count()['Cruise']
        for item in series.iteritems():
            tup,n = item
            ii_index = self.bins_lon.index(tup[1])
            qq_index = self.bins_lat.index(tup[0])
            ZZ[qq_index,ii_index] = n           
        plt.figure(figsize=(10,10))
        m = Basemap(projection='cyl',fix_aspect=False)
        # m.fillcontinents(color='coral',lake_color='aqua')
        m.drawcoastlines()
        XX,YY = m(self.X,self.Y)
        ZZ = np.ma.masked_equal(ZZ,0)
        m.pcolormesh(XX,YY,ZZ,vmin=0,cmap=plt.cm.magma)
        plt.title('Profile Density',size=30)
        plt.colorbar(label='Number of float profiles')
        print 'I am saving argo dense figure'
        plt.savefig(self.base_file+'argo_dense_data/number_matrix_degree_bins_'+str(self.degree_bins)+'.png')
        plt.close()

    def dep_number_plot(self):
        ZZ = np.zeros([len(self.bins_lat),len(self.bins_lon)])
        series = self.df.drop_duplicates(subset=['Cruise'],keep='first').groupby('bin_index').count()['Cruise']
        for item in series.iteritems():
            tup,n = item
            ii_index = self.bins_lon.index(tup[1])
            qq_index = self.bins_lat.index(tup[0])
            ZZ[qq_index,ii_index] = n  
        plt.figure(figsize=(10,10))
        m = Basemap(projection='cyl',fix_aspect=False)
        # m.fillcontinents(color='coral',lake_color='aqua')
        m.drawcoastlines()
        XX,YY = m(self.X,self.Y)
        ZZ = np.ma.masked_equal(ZZ,0)
        m.pcolormesh(XX,YY,ZZ,vmin=0,vmax=30,cmap=plt.cm.magma)
        plt.title('Deployment Density',size=30)
        plt.colorbar(label='Number of floats deployed')
        plt.savefig(self.base_file+'deployment_number_data/number_matrix_degree_bins_'+str(self.degree_bins)+'.png')
        plt.close()

        plt.figure(figsize=(10,10))
        m = Basemap(projection='cyl',fix_aspect=False)
        # m.fillcontinents(color='coral',lake_color='aqua')
        m.drawcoastlines()
        series = self.df.drop_duplicates(subset=['Cruise'],keep='first')
        m.scatter(series.Lon.values,series.Lat.values,s=0.2)
        plt.title('Deployment Locations',size=30)
        plt.savefig(self.base_file+'deployment_number_data/deployment_locations.png')
        plt.close()

    def trans_number_matrix_plot(self): 
        try:
            self.number_matrix = load_sparse_csr(self.base_file+'transition_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
            self.standard_error = load_sparse_csr(self.base_file+'transition_matrix/standard_error_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
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
        m.pcolormesh(XX,YY,number_matrix_plot,cmap=plt.cm.magma)
        plt.title('Transition Density',size=30)
        plt.colorbar(label='Number of float transitions')
        plt.savefig(self.base_file+'/number_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
        plt.close()

        k = np.diagonal(self.standard_error.todense())
        standard_error_plot = self.transition_vector_to_plottable(k)
        plt.figure(figsize=(10,10))
        m = Basemap(projection='cyl',fix_aspect=False)
        # m.fillcontinents(color='coral',lake_color='aqua')
        m.drawcoastlines()
        XX,YY = m(self.X,self.Y)
        # number_matrix_plot[number_matrix_plot>1000]=1000
        standard_error_plot = np.ma.masked_equal(standard_error_plot,0)
        m.pcolormesh(XX,YY,standard_error_plot,cmap=plt.cm.cividis)
        plt.title('Standard Error',size=30)
        plt.colorbar(label='Standard Error')
        plt.savefig(self.base_file+'/number_matrix/standard_error_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
        plt.close()

    def deployment_locations(self):
        plt.figure(figsize=(10,10))
        m = Basemap(projection='cyl',fix_aspect=False)
        # m.fillcontinents(color='coral',lake_color='aqua')
        m.drawcoastlines()
        m.scatter(self.df.Lon.values,self.df.Lat.values,s=0.2)
        plt.title('Float Profiles',size=30)
        plt.savefig(self.base_file+'deployment_number_data/profile_locations.png')
        plt.close()


    def matrix_compare(self,matrix_a,matrix_b,num_matrix_a,num_matrix_b,title,save_name): #this function accepts sparse matrices
        vmax = 0.4
        transition_matrix_plot = matrix_a-matrix_b
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
        """
        This compares the transition matrices created by GPS and ARGOS tracked floats
        """
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
        self.matrix_compare(transition_matrix_argos,transition_matrix_gps,number_matrix_argos,number_matrix_gps,
            'Dataset Difference (GPS - ARGOS)','./dataset_difference/dataset_difference_degree_bins_'+
            str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
        self.diagnose_matrix(transition_matrix_argos,'./dataset_difference/argos_diagnose_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
        self.diagnose_matrix(transition_matrix_gps,'./dataset_difference/gps_diagnose_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
        self.z_test(transition_matrix_gps.todense(),transition_matrix_argos.todense(),number_matrix_gps.todense(),number_matrix_argos.todense())



    def seasonal_compare(self):
        """
        This compares the transition matrices generated by NDJF and MJJA
        """
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

    def SOSE_transition_matrix_plot(self):
        """
        uses dataframe from sose trajectories to build a transition matrix 
        """
        self.df = pd.read_pickle(self.base_file+'sose_particle_df.pickle').sort_values(by=['Cruise','Date'])
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
            self.df_transition = pd.read_pickle(self.base_file+'sose_transition_df_degree_bins_'+str(self.degree_bins)+'.pickle')

        except IOError: #this is the case that the file could not load
            print 'i could not load the transition df, I am recompiling with degree step size', self.degree_bins,''
            self.recompile_transition_df(dump=False)
            self.df_transition.to_pickle(self.base_file+'sose_transition_df_degree_bins_'+str(self.degree_bins)+'.pickle')
        self.df_transition = self.df_transition.dropna(subset=[self.end_bin_string])
        self.identify_problems_df_transition()
        while len(self.df_transition)!=len(self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]): #need to loop this
            self.df_transition = self.df_transition[self.df_transition[self.end_bin_string].isin(self.df_transition['start bin'].unique())]
        self.total_list = [list(x) for x in self.df_transition['start bin'].unique()] 

        try: # try to load the transition matrix
            self.transition_matrix = load_sparse_csr(self.base_file+'transition_matrix/sose_transition_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(date_span_limit)+'.npz')
        except IOError: # if the matrix cannot load, recompile
            print 'i could not load the transition matrix, I am recompiling with degree step size', self.degree_bins,' and time step ',self.date_span_limit
            self.recompile_transition_matrix(dump=False)
            save_sparse_csr(self.base_file+'transition_matrix/sose_transition_matrix_degree_bins_'+str(degree_bins)+'_time_step_'+str(date_span_limit)+'.npz',self.transition_matrix)

    def get_direction_matrix(self):
        """
        This creates a matrix that shows the number of grid cells north and south, east and west.
        """
        lat_list, lon_list = zip(*self.total_list)
        pos_max = 180/self.degree_bins #this is the maximum number of bins possible
        output_list = []
        for position_list in [lat_list,lon_list]:
            token_array = np.zeros([len(position_list),len(position_list)])
            for token in np.unique(position_list):
                print token
                index_list = np.where(np.array(position_list)==token)[0]

                token_list = (np.array(position_list)-token)/self.degree_bins #the relative number of degree bins
                token_list[token_list>pos_max]=token_list[token_list>pos_max]-2*pos_max #the equivalent of saying -360 degrees
                token_list[token_list<-pos_max]=token_list[token_list<-pos_max]+2*pos_max #the equivalent of saying +360 degrees

                token_array[:,index_list]=np.array([token_list.tolist(),]*len(index_list)).transpose() 
            output_list.append(token_array)
        north_south,east_west = output_list
        self.east_west = east_west
        self.north_south = north_south  #this is because np.array makes lists of lists go the wrong way
        assert (self.east_west<=180/self.degree_bins).all()
        assert (self.east_west>=-180/self.degree_bins).all()
        assert (self.north_south>=-180/self.degree_bins).all()
        assert (self.north_south<=180/self.degree_bins).all()

    def quiver_plot(self,matrix,arrows=True,degree_sep=4,scale_factor=20):
        """
        This plots the mean transition quiver as well as the variance ellipses
        """
# todo: need to check this over. I think there might be a bug.
        trans_mat = matrix.todense()
        east_west = np.multiply(trans_mat,self.east_west)
        north_south = np.multiply(trans_mat,self.north_south)
        ew_test = scipy.sparse.csc_matrix(east_west)
        ns_test = scipy.sparse.csc_matrix(north_south)
        # ew_test.data = ew_test.data**2
        # ns_test.data = ns_test.data**2
        # test = ew_test + ns_test
        # test = np.sqrt(test)
        # east_west[test.todense()>8]=0
        # north_south[test.todense()>8]=0
        m = Basemap(projection='cyl',fix_aspect=False)
        m.fillcontinents(color='coral',lake_color='aqua')

        m.drawcoastlines()
        Y_ = np.arange(-68.0,66.0,degree_sep)
        X_ = np.arange(-162.0,162.0,degree_sep)
        XX,YY = np.meshgrid(X_,Y_)
        n_s = np.zeros(XX.shape)
        e_w = np.zeros(XX.shape)
        XX,YY = m(XX,YY)
        for i,lat in enumerate(Y_):
            for k,lon in enumerate(X_):
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
                try:
                    poly = m.ellipse(lon, lat, axis1*scale_factor,axis2*scale_factor, 80,phi=angle, facecolor='green', zorder=3,alpha=0.35)
                except ValueError:
                    print ' there was a value error in the calculation of the transition_matrix'
                    continue
        e_w = np.ma.array(e_w,mask=(e_w==0))
        n_s = np.ma.array(n_s,mask=(n_s==0))
        if arrows:
            m.quiver(XX,YY,e_w*5000,n_s*5000,scale=25)

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
                except ValueError:  # this is for the case that the starting location of the the soccom float is not in the transition matrix
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
        field_vector = self.cm2p6(self.base_file+'mean_pco2.dat')
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
        field_vector = self.cm2p6(self.base_file+'mean_o2.dat')
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
        field_vector_pco2 = self.cm2p6(self.base_file+'mean_pco2.dat')
        field_vector_o2 = self.cm2p6(self.base_file+'mean_o2.dat')
        field_vector_pco2[field_vector_pco2>(10*field_vector_pco2.std()+field_vector_pco2.mean())]=0
        field_vector = field_vector_pco2/(2*field_vector_pco2.max())+field_vector_o2/(2*field_vector_o2.max())
        field_plot = abs(self.transition_vector_to_plottable(field_vector))

        m,x,y,desired_vector = self.get_optimal_float_locations(field_vector)
        XX,YY = m(self.X,self.Y)
        m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Greens) # this is a plot for the tendancy of the residence time at a grid cell
        m.scatter(x,y,marker='*',color='r',s=30)        
        plt.show()

    def load_corr_array(self,variable,lat,lon):
        cor_folder = './'+str(variable)+'/'
        lon_ = lon
        if lon_>80:
            lon_ -= 360
        if lon_<-260:
            lon_ += 360
        file_ = cor_folder+str(lat)+'_'+str(lon_)+'.npy'
        try:
            array_ = np.load(file_)
        except IOError:
            print 'could not load file, lon is ',lon_,' lat is ',lat
            index_list = [self.total_list.index([lat,lon])]
            data_list = [1]
            return (data_list,index_list)
        lat_list = np.arange(float(lat),float(lat)-12,(-1*self.degree_bins))[::-1].tolist()+np.arange(float(lat),float(lat)+12,self.degree_bins)[1:].tolist()
        column_index_list = np.arange(lat-12,lat+12,0.5).tolist()
        lon_list = np.arange(float(lon),float(lon)-12,(-1*self.degree_bins))[::-1].tolist()+np.arange(float(lon),float(lon)+12,self.degree_bins)[1:].tolist()
        row_index_list = np.arange(lon-12,lon+12,0.5).tolist()
        data_list = []
        index_list = []
        Y,X = np.meshgrid(lat_list,lon_list)
        for token in zip(Y.flatten(),X.flatten()):
            token = list(token)
            # try:
            row = row_index_list.index(token[1])
            column = column_index_list.index(token[0])
            if token[1]>180:
                token[1]-=360
            if token[1]<-180:
                token[1]+=360
            try:
                index_list.append(self.total_list.index(token))
                data_list.append(array_[row,column])
            except ValueError:
                print 'there was a mistake in the index_list'
                print token
                continue
        assert len(data_list)>0
        assert len(index_list)>0
        return (data_list,index_list)

    def load_corr_matrix(self,variable):
        try:
            self.cor_matrix = load_sparse_csr(self.base_file+variable+'_cor_matrix_degree_bins_'+str(self.degree_bins)+'.npz')
        except IOError:
            np.load(variable+'_corr.npy')
            data_list = []
            row_list = []
            column_list = []
            for n,token in enumerate(self.total_list):
                print n
                base_lat,base_lon = token
                data_list_token,row_list_token = self.load_corr_array(variable,base_lat,base_lon)
                print row_list_token
                row_list += row_list_token
                column_list += [n]*len(row_list_token)
                data_list += data_list_token            
            self.cor_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(self.total_list),len(self.total_list)))
            save_sparse_csr(self.base_file+variable+'_cor_matrix_degree_bins_'+str(self.degree_bins)+'.npz', self.cor_matrix)

for token in [(2,1),(4,1)]:
    degree,scale = token
    traj_class = argo_traj_data(degree_bins=degree)
    traj_class.get_direction_matrix()
    # traj_class.load_corr_matrix('o2')
    # traj_class.quiver_plot(traj_class.cor_matrix,arrows=False,degree_sep=6,scale_factor=scale)
    # plt.title('O2 correlation ellipses')
    # plt.savefig(traj_class.base_file+'/transition_plots/o2_correlation_degree_bins_'+str(degree)+'.png')
    # plt.close()
    traj_class.load_corr_matrix('pco2')
    traj_class.quiver_plot(traj_class.cor_matrix,arrows=False,degree_sep=2,scale_factor=10)
    plt.title('pCO2 correlation ellipses')
    plt.savefig(traj_class.base_file+'/transition_plots/pco2_correlation_degree_bins_'+str(degree)+'.png')
    plt.close()
    # for time in np.arange(20,260,40):
    #     print 'plotting for degree ',degree,' and time ',time
    #     traj_class = argo_traj_data(degree_bins=degree,date_span_limit=time)
    #     traj_class.get_direction_matrix()
    #     traj_class.trans_number_matrix_plot()
    #     traj_class.transition_matrix_plot(filename='base_transition_matrix')
    #     plt.figure(figsize=(10,10))
    #     traj_class.quiver_plot(traj_class.transition_matrix)
    #     plt.savefig(traj_class.base_file+'/transition_plots/quiver_diagnose_degree_bins_'+str(degree)+'_time_step_'+str(time)+'.png')
    #     plt.close()
    #     if time == 20:
    #         traj_class.argo_dense_plot()
    #         traj_class.dep_number_plot()
