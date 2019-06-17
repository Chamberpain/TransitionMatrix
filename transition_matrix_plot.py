from ../compute/transition_matrix_compute import Trajectory,Matrix
import plt as plt
from utils import basemap_setup

class TrajectoryPlot(Trajectory)
    pass
    def profile_density_plot(self): #plot the density of profiles
        ZZ = np.zeros([len(self.bins_lat),len(self.bins_lon)])
        series = self.df.groupby('bin_index').count()['Cruise']
        for item in series.iteritems():
            tup,n = item
            ii_index = self.bins_lon.index(tup[1])
            qq_index = self.bins_lat.index(tup[0])
            ZZ[qq_index,ii_index] = n           
        plt.figure(figsize=(10,10))
        XX,YY,m = basemap_setup(self.bins_lon,self.bins_lat,self.traj_file_type)    
        ZZ = np.ma.masked_equal(ZZ,0)
        m.pcolormesh(XX,YY,ZZ,vmin=0,cmap=plt.cm.magma)
        plt.title('Profile Density',size=30)
        plt.colorbar(label='Number of float profiles')
        print 'I am saving argo dense figure'
        plt.savefig('../plots/argo_dense_data/number_matrix_degree_bins_'+str(self.degree_bins)+'.png')
        plt.close()

    def deployment_density_plot(self): # plot the density of deployment locations
        ZZ = np.zeros([len(self.bins_lat),len(self.bins_lon)])
        series = self.df.drop_duplicates(subset=['Cruise'],keep='first').groupby('bin_index').count()['Cruise']
        for item in series.iteritems():
            tup,n = item
            ii_index = self.bins_lon.index(tup[1])
            qq_index = self.bins_lat.index(tup[0])
            ZZ[qq_index,ii_index] = n  
        plt.figure(figsize=(10,10))
        XX,YY,m = basemap_setup(self.bins_lon,self.bins_lat,self.traj_file_type)    
        ZZ = np.ma.masked_equal(ZZ,0)
        m.pcolormesh(XX,YY,ZZ,vmin=0,vmax=30,cmap=plt.cm.magma)
        plt.title('Deployment Density',size=30)
        plt.colorbar(label='Number of floats deployed')
        plt.savefig('../plots/deployment_number_data/number_matrix_degree_bins_'+str(self.degree_bins)+'.png')
        plt.close()

    def deployment_location_plot(self): # plot all deployment locations
        plt.figure(figsize=(10,10))
        XX,YY,m = basemap_setup(self.bins_lon,self.bins_lat,self.traj_file_type)    
        series = self.df.drop_duplicates(subset=['Cruise'],keep='first')
        m.scatter(series.Lon.values,series.Lat.values,s=0.2)
        plt.title('Deployment Locations',size=30)
        plt.savefig('../plots/deployment_number_data/deployment_locations.png')
        plt.close()

class TransMatrixPlot(Matrix):
    pass

    def get_SOCCOM_vector(self,age_return=False,future_projection=True):
        path = '../data/SOCCOM_trajectory/'
        files = []
        # r=root, d=directories, f = files
        for r, d, f in os.walk(path):
            for file in f:
                if '.txt' in file:
                    files.append(os.path.join(r, file))
        df_list = []
        for file in files:
            pd.read_csv(file)
            df_holder = pd.read_csv(file,skiprows=[1,2,3],delim_whitespace=True,usecols=['Float_ID','Cycle','Date','Time','Lat','Lon','POS_QC','#'])
            df_holder.columns = ['Float_ID','Cycle','Date','Time','Lat','Lon','POS_QC','Cruise'] 
            df_holder['Date'] = pd.to_datetime(df_holder['Date'],format='%Y%m%d')
            df_holder = df_holder[df_holder.POS_QC.isin([0,1])]
            if (datetime.date.today()-df_holder.Date.tail(1)).dt.days.values[0]>270:
                print 'Float is dead, rejecting'
                continue
            df = df_holder[['Lat','Lon']].tail(1)
            df['Age'] = ((df_holder.Date.tail(1)-df_holder.Date.head(1).values).dt.days/365).values[0]
            df_list.append(df)
        df_tot = pd.concat(df_list)
        df_tot['bins_lat'] = pd.cut(df_tot.Lat,bins = self.info.bins_lat,labels=self.info.bins_lat[:-1])
        df_tot['bins_lon'] = pd.cut(df_tot.Lon,bins = self.info.bins_lon,labels=self.info.bins_lon[:-1])
        df_tot['bin_index'] = zip(df_tot['bins_lat'].values,df_tot['bins_lon'].values)
        float_vector = np.zeros(self.matrix.transition_matrix.shape[0])
        for x,age in df_tot[['bin_index','Age']].values:
            try:
                idx = self.transition.list.index(list(x))
                if age_return:
                    percent = 1/np.ceil(age)
                    if percent < 1/6.:
                        percent = 0
                    float_vector[idx] = percent
                else:
                    float_vector[idx] = 1
            except ValueError:
                print str(x)+' is not found'
        if future_projection:
            float_vector = self.matrix.transition_matrix.dot(float_vector)
        assert (float_vector<=1).all()
        return float_vector

    def plot_latest_soccom_locations(self,m):
        y,x = zip(*np.array(self.list)[self.soccom_location>0])
        m.scatter(x,y,marker='*',color='b',s=34,latlon=True)
        return m

    def transition_vector_to_plottable(self,vector):
        plottable = np.zeros([len(self.bins_lat),len(self.bins_lon)])
        for n,tup in enumerate(self.list):
            ii_index = self.bins_lon.index(tup[1])
            qq_index = self.bins_lat.index(tup[0])
            plottable[qq_index,ii_index] = vector[n]
        return plottable
      
    def number_plot(self): 
        k = self.number_matrix.sum(axis=0)
        k = k.T
        print k
        number_matrix_plot = self.transition_vector_to_plottable(k)
        matplotlib.pyplot.figure('number matrix',figsize=(10,10))
        m,XX,YY = self.matrix_plot_setup()
        number_matrix_plot = np.ma.masked_equal(number_matrix_plot,0)   #this needs to be fixed in the plotting routine, because this is just showing the number of particles remaining
        m.pcolormesh(XX,YY,number_matrix_plot,cmap=matplotlib.pyplot.cm.magma)
        # matplotlib.pyplot.title('Transition Density',size=30)
        cbar = matplotlib.pyplot.colorbar()
        cbar.set_label(label='Transition Number',size=30)
        cbar.ax.tick_params(labelsize=30)
        matplotlib.pyplot.savefig(self.base_file+'/number_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
        matplotlib.pyplot.close()

    def standard_error_plot(self):

#Todo: the way this is calculated the diagonal goes to infinity if the number of floats remaining in a bin goes to zero. I suspect that I may not be calculating this properly
        row_list = self.transition_matrix.tocoo().row.tolist()
        column_list = self.transition_matrix.tocoo().col.tolist()
        data_list = self.transition_matrix.tocoo().data.tolist()        
        sparse_ones = scipy.sparse.csc_matrix(([1 for x in range(len(data_list))],(row_list,column_list)),shape=(len(self.transition.list),len(self.transition.list)))
        standard_error = np.sqrt(self.transition_matrix*(sparse_ones-self.transition_matrix)/self.number_matrix)
        standard_error = scipy.sparse.csc_matrix(standard_error)
        k = np.diagonal(standard_error.todense())
        standard_error_plot = self.transition_vector_to_plottable(k)
        matplotlib.pyplot.figure('standard error')
        # m.fillcontinents(color='coral',lake_color='aqua')
        # number_matrix_plot[number_matrix_plot>1000]=1000
        m,XX,YY = self.matrix_plot_setup()
        q = self.number_matrix.sum(axis=0)
        q = q.T
        q = np.ma.masked_equal(q,0)
        standard_error_plot = np.ma.array(standard_error_plot,mask=self.transition_vector_to_plottable(q)==0)
        m.pcolormesh(XX,YY,standard_error_plot,cmap=matplotlib.pyplot.cm.cividis)
        matplotlib.pyplot.title('Standard Error',size=30)
        matplotlib.pyplot.colorbar(label='Standard Error')
        matplotlib.pyplot.savefig(self.base_file+'/number_matrix/standard_error_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
        matplotlib.pyplot.close()

    def transition_matrix_plot(self,filename):
        matplotlib.pyplot.figure(figsize=(10,10))
        k = np.diagonal(self.transition_matrix.todense())
        transition_plot = self.transition_vector_to_plottable(k)
        m,XX,YY = self.matrix_plot_setup()
        k = self.number_matrix.sum(axis=0)
        k = k.T
        transition_plot = np.ma.array(100*(1-transition_plot),mask=self.transition_vector_to_plottable(k)==0)
        m.pcolormesh(XX,YY,transition_plot,vmin=0,vmax=100) # this is a plot for the tendancy of the residence time at a grid cell
        cbar = matplotlib.pyplot.colorbar()
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('% Particles Dispersed',size=30)
        # matplotlib.pyplot.title('1 - diagonal of transition matrix',size=30)
        matplotlib.pyplot.savefig(self.base_file+'transition_plots/'+filename+'_diag_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
        matplotlib.pyplot.close()


    def quiver_plot(self,trans_mat,arrows=True,degree_sep=4,scale_factor=20):
        """
        This plots the mean transition quiver as well as the variance ellipses
        """
# todo: need to check this over. I think there might be a bug.
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
        Basemap(projection='cea',llcrnrlon=-180.,llcrnrlat=-80.,urcrnrlon=180.,urcrnrlat=80,fix_aspect=False)
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
                    index = self.transition.list.index([lat,lon])
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

    def get_argo_vector():
        file_ = '../../argo_traj_box/ar_index_global_prof.txt'
        df_ = pd.read_csv(file_,skiprows=8)
        df_['Cruise'] = [dummy.split('/')[1] for dummy in df_['file'].values]
        df_ = df_[~df_.date.isna()]
        df_['Date'] = pd.to_datetime([int(_) for _ in df_.date.values.tolist()],format='%Y%m%d%H%M%S')
        df_['Lat'] = df_['latitude']
        df_['Lon'] = df_['longitude']
        df_ = df_[['Lat','Lon','Date','Cruise']]
        active_floats = df_[df_.Date>(df_.Date.max()-relativedelta(months=6))].Cruise.unique()
        df_ = df_[df_.Cruise.isin(active_floats)]
        df_ = df_.drop_duplicates(subset='Cruise',keep='first')
        df_['Age'] = np.ceil((df_.Date.max()-df_.Date).dt.days/360.)
        df_['bins_lat'] = pd.cut(df_.Lat,bins = self.info.bins_lat,labels=self.info.bins_lat[:-1])
        df_['bins_lon'] = pd.cut(df_.Lon,bins = self.info.bins_lon,labels=self.info.bins_lon[:-1])
        df_['bin_index'] = zip(df_['bins_lat'].values,df_['bins_lon'].values)
        float_vector = np.zeros((self.matrix.transition_matrix.shape[0],1))
        for x in df_['bin_index'].unique():
            try:
                idx = self.transition.list.index(list(x))
                df_dummy = df_[df_['bin_index']==x]
                percent = len(df_dummy)/df_dummy.Age.mean()
                float_vector[idx] = percent
            except ValueError:
                print str(x)+' is not found'
        return float_vector

    def plot_argo_vector():
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
        cmap = mcolors.LinearSegmentedColormap('my_colormap', cdict, 100) 
        plt.figure()
        plot_vector = self.matrix.transition_vector_to_plottable(float_vector)
        plot_vector = np.ma.masked_equal(plot_vector,0)
        print plot_vector.sum()-original_plot_vector.sum()
        XX,YY,m = basemap_setup(self.bins_lon,self.bins_lat,self.traj_file_type)    
        m.pcolor(XX,YY,plot_vector,cmap=cmap,vmax=2,vmin=0)
        plt.colorbar()
        plt.title('density/age for '+str(self.info.degree_bins)+' at '+self.info.date_span_limit+' days')            
        plt.savefig('../plots/argo_dendsity_projection')

    def future_agregation_plot(self):
        w = traj_class.transition_matrix
        vector = np.ones((w.shape[0],1))
        vector = vector/vector.sum()*100
        for k in range(20):
            months = 6*2**k
            w = w.dot(w)
            future_output = self.transition_vector_to_plottable((w.dot(scipy.sparse.csc_matrix(vector))).todense())
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

    def eig_vec_plot(self,eig_vec):
        eig_vec_token = self.transition_vector_to_plottable(eig_vec)
        eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
        if self.traj_file_type=='SOSE':
            m = Basemap(llcrnrlon=-180.,llcrnrlat=-80.,urcrnrlon=180.,urcrnrlat=-25,projection='cyl',fix_aspect=False)
        else:
            m = Basemap(projection='cyl',fix_aspect=False)
        m.drawcoastlines()
        XX,YY = m(self.X,self.Y)
        mag = abs(eig_vec_token).max()
        m.pcolormesh(XX,YY,eig_vec_token,norm=colors.SymLogNorm(linthresh=0.000003, linscale=0.000003,
                                              vmin=-mag/50, vmax=mag/50))

    def eig_total_plot(self,matrix,label=''):
        eig_vals,l_eig_vecs,r_eig_vecs = scipy.linalg.eig(matrix.todense(),left=True)
        idx = eig_vals.argsort()[::-1]
        eig_vals = eig_vals[idx]
        l_eig_vecs = l_eig_vecs[:,idx]
        r_eig_vecs = r_eig_vecs[:,idx]

        eig_vals = eig_vals[eig_vals>0.97]
        for k,eig_val in enumerate(eig_vals):
            print 'plotting eigenvector '+str(k)+' for '+str(self.degree_bins)+' eigenvalue is '+str(eig_val)
            l_eig_vec = l_eig_vecs[:,k]
            assert (l_eig_vec.T.dot(self.transition_matrix.todense())-eig_val*l_eig_vec).max()<10**-1
            r_eig_vec = r_eig_vecs[:,k]
            assert (self.transition_matrix.todense().dot(r_eig_vec)-eig_val*r_eig_vec).max()<10**-1

            plt.figure(figsize=(10,10))
            plt.subplot(3,2,1)
            self.eig_vec_plot(l_eig_vec.real)
            plt.title('left eig vec (real)')
            plt.subplot(3,2,2)
            self.eig_vec_plot(np.absolute(l_eig_vec))
            plt.title('left eig vec (absolute)')
            plt.subplot(3,2,3)
            self.eig_vec_plot(r_eig_vec.real)
            plt.title('right eig vec (real)')
            plt.subplot(3,2,4)
            self.eig_vec_plot(np.absolute(r_eig_vec))            
            plt.title('right eig vec (absolute)')
            plt.subplot(3,1,3)
            plt.plot(eig_vals)
            plt.title('Eigen Value Spectrum')
            plt.savefig(self.base_file+'plots/diag_degree_bins_'+str(self.degree_bins)+'/r_l_eig_vals_'+str(k)+'_time_step_'+str(self.date_span_limit)+label+'.png')
            plt.close()



def figure2():
    coord_list = [(2,2)]
    date_list = [100]
    for lat,lon in coord_list:
        for date in date_list:
            traj_class = TransMatrix(lat,lon,date)
            traj_class.get_direction_matrix()
            traj_class.quiver_plot(traj_class.transition_matrix,degree_sep=4,scale_factor=25)
            matplotlib.pyplot.savefig('test')





def SOCCOM_death_plot():
    traj_class = TransMatrixPlot(2,3,100,'Argo')
    traj_plot = argo_traj_plot(traj_class,time_multiplyer=2)
    float_vector = argo_traj_plot.get_SOCCOM_vector(age_return=True)
    plottable = traj_class.matrix.transition_vector_to_plottable(float_vector)
    traj_class.info.traj_file_type = 'SOSE'
    m,XX,YY = traj_class.matrix.matrix_plot_setup() 
    m.fillcontinents(color='coral',lake_color='aqua')
    m.pcolormesh(XX,YY,np.ma.masked_equal(plottable,0),cmap=plt.cm.magma,vmax = plottable.max()/2)
    plt.title('SOCCOM Sampling Survival PDF in 300 Days')
    plt.colorbar()
    plt.show()
    plt.savefig('survival')



def gps_argos_compare():
    argos_class = TransMatrix(degree_bin_lat=2,degree_bin_lon=3,date_span_limit=100,traj_file_type='Argo',float_type='ARGOS')
    gps_class = TransMatrix(degree_bin_lat=2,degree_bin_lon=3,date_span_limit=100,traj_file_type='Argo',float_type='GPS')
    gps_class, argos_class = matrix_size_match(gps_class,argos_class)
    matrix_difference_compare(gps_class.matrix.transition_matrix,argos_class.matrix.transition_matrix)


def season_compare():
    winter_class = TransMatrix(degree_bin_lat=2,degree_bin_lon=3,date_span_limit=100,traj_file_type='Argo',season = [11,12,1,2])
    summer_class = TransMatrix(degree_bin_lat=2,degree_bin_lon=3,date_span_limit=100,traj_file_type='Argo',season = [5,6,7,8])
    winter_class, summer_class = matrix_size_match(winter_class,summer_class)
    matrix_difference_compare(winter_class.matrix.transition_matrix,summer_class.matrix.transition_matrix)




    def __init__(self,traj_class,time_multiplyer=1):
        self = traj_class
        matrix = self.transition_matrix
        for _ in range(time_multiplyer-1):
            matrix = matrix.dot(self.matrix.transition_matrix)
        self.matrix.transition_matrix = matrix

        lon_list = np.load('lon_list.npy')
        lon_list[lon_list<-180]=lon_list[lon_list<-180]+360
        self.lon_list = lon_list

        self.translation_list = []
        for x in self.transition.list:
            mask = (self.lat_list==x[0])&(self.lon_list==x[1])      
            if not mask.any():
                for lat in np.arange(x[0]-self.info.degree_bins[0],x[0]+self.info.degree_bins[0],0.5):
                    for lon in np.arange(x[1]-self.info.degree_bins[1],x[1]+self.info.degree_bins[1],0.5):
                        mask = (self.lat_list==lat)&(self.lon_list==lon)
                        if mask.any():
                            t = np.where(mask)
                            break
                    if mask.any():
                        break
            else:
                t = np.where(mask)
            assert t[0]
            assert len(t[0])==1
            self.translation_list.append(t[0][0])


        self.corr_translation_list = []
        mask = (self.lon_list%1==0)&(self.lat_list%1==0)
        lats = self.lat_list[mask]
        lons = self.lon_list[mask]
        for x in self.transition.list:
            mask = (lats==x[0])&(lons==x[1])      
            if not mask.any():
                for lat in np.arange(x[0]-self.info.degree_bins[0],x[0]+self.info.degree_bins[0],0.5):
                    for lon in np.arange(x[1]-self.info.degree_bins[1],x[1]+self.info.degree_bins[1],0.5):
                        mask = (lats==lat)&(lons==lon)
                        if mask.any():
                            t = np.where(mask)
                            break
                    if mask.any():
                        break
            else:
                t = np.where(mask)
            assert t[0]
            assert len(t[0])==1
            self.corr_translation_list.append(t[0][0])

        self.o2 = np.load('subsampled_o2.npy')
        o2_corr = np.load('o2_corr.npy')
        self.o2_corr = self.get_sparse_correlation_matrix(o2_corr)


        self.pco2 = np.load('subsampled_pco2.npy')
        pco2_corr = np.load('pco2_corr.npy')
        self.pco2_corr = self.get_sparse_correlation_matrix(pco2_corr)
        self.soccom_location = self.get_SOCCOM_vector(age_return=False,future_projection=False)

    def get_sparse_correlation_matrix(self,correlation_list):
        total_set = Set([tuple(x) for x in self.transition.list])
        row_list = []
        col_list = []
        data_list = []
        mask = (self.lon_list%1==0)&(self.lat_list%1==0) # we only calculated correlations at whole degrees because of memory
        lats = self.lat_list[mask][self.corr_translation_list] #mask out non whole degrees and only grab at locations that match to the self.transition.index
        lons = self.lon_list[mask][self.corr_translation_list] 
        corr_list = correlation_list[self.corr_translation_list]

        # test_Y,test_X = zip(*self.transition.list)
        for k,(base_lat,base_lon,corr) in enumerate(zip(lats,lons,corr_list)):
            print k
            lat_index_list = np.arange(base_lat-12,base_lat+12.1,0.5)
            lon_index_list = np.arange(base_lon-12,base_lon+12.1,0.5)
            Y,X = np.meshgrid(lat_index_list,lon_index_list) #we construct in this way to match how the correlation matrix was made
            test_set = Set(zip(Y.flatten(),X.flatten()))
            intersection_set = total_set.intersection(test_set)
            location_idx = [self.transition.list.index(list(_)) for _ in intersection_set]
            data_idx = [zip(Y.flatten(),X.flatten()).index(_) for _ in intersection_set]

            data = corr.flatten()[data_idx]
            assert len(location_idx)==len(data)
            row_list += location_idx
            col_list += [k]*len(location_idx)
            data_list += data.tolist()
        assert (np.array(data_list)<=1).all()
        return scipy.sparse.csc_matrix((np.abs(data_list),(row_list,col_list)),shape=self.matrix.transition_matrix.shape)










    def landschutzer_co2_flux(self,var=False,plot=False):
        file_ = self.base_file+'spco2_MPI_SOM-FFN_v2018.nc'
        nc_fid = Dataset(file_)
        y = nc_fid['lat'][:]
        x = nc_fid['lon'][:]
        
        data = np.ma.masked_greater(nc_fid['fgco2_smoothed'][:],10**19) 
        if var:
            np.nanvar()
        else:
            XX,YY = np.gradient(np.nanmean(data,axis=0)) # take the time mean, then take the gradient of the 2d array
            data = np.abs(XX+YY)
        x[0] = -180
        x[-1] = 180
        co2_vector = np.zeros([len(self.total_list),1])
        for n,(lat,lon) in enumerate(self.total_list):
            lon_index = find_nearest(lon,x)
            lat_index = find_nearest(lat,y)
            co2_vector[n] = data[lat_index,lon_index]
        if plot:
            co2_plot = abs(self.transition_vector_to_plottable(co2_vector))
            plt.figure()
            m = Basemap(projection='cyl',fix_aspect=False)
            # m.fillcontinents(color='coral',lake_color='aqua')
            m.drawcoastlines()
            XX,YY = m(self.X,self.Y)
            m.pcolormesh(XX,YY,co2_plot,cmap=plt.cm.PRGn)
            plt.colorbar(label='CO2 Flux $gm C/m^2/yr$')
        return co2_vector


    def landschutzer_loc_plot(self,var=False):
        field_vector = landschutzer_co2_flux(var)
        field_plot = abs(self.matrix.transition_vector_to_plottable(field_vector))
        x,y,desired_vector,target_vector =  self.get_optimal_float_locations(field_vector,self.pco2_corr,desired_vector_in_plottable=(not cost_plot)) #if cost_plot, the vector will return desired vector in form for calculations
        m,XX,YY = traj_class.matrix.matrix_plot_setup() 
        m.fillcontinents(color='coral',lake_color='aqua')
        m.pcolormesh(XX,YY,np.ma.masked_equal(field_plot,0),cmap=plt.cm.Purples,norm=colors.LogNorm())
        plt.title('PCO2 Variance')
        plt.colorbar()
        m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Purples) # this is a plot for the tendancy of the residence time at a grid cell
        m.scatter(x,y,marker='*',color='g',s=34)
        m = self.plot_latest_soccom_locations(m)
        plt.show()


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

    def get_sparse_correlation_matrix(self,correlation_list):
        total_set = Set([tuple(x) for x in self.transition.list])
        row_list = []
        col_list = []
        data_list = []
        mask = (self.lon_list%1==0)&(self.lat_list%1==0) # we only calculated correlations at whole degrees because of memory
        lats = self.lat_list[mask][self.corr_translation_list] #mask out non whole degrees and only grab at locations that match to the self.transition.index
        lons = self.lon_list[mask][self.corr_translation_list] 
        corr_list = correlation_list[self.corr_translation_list]

        # test_Y,test_X = zip(*self.transition.list)
        for k,(base_lat,base_lon,corr) in enumerate(zip(lats,lons,corr_list)):
            print k
            lat_index_list = np.arange(base_lat-12,base_lat+12.1,0.5)
            lon_index_list = np.arange(base_lon-12,base_lon+12.1,0.5)
            Y,X = np.meshgrid(lat_index_list,lon_index_list) #we construct in this way to match how the correlation matrix was made
            test_set = Set(zip(Y.flatten(),X.flatten()))
            intersection_set = total_set.intersection(test_set)
            location_idx = [self.transition.list.index(list(_)) for _ in intersection_set]
            data_idx = [zip(Y.flatten(),X.flatten()).index(_) for _ in intersection_set]

            data = corr.flatten()[data_idx]
            assert len(location_idx)==len(data)
            row_list += location_idx
            col_list += [k]*len(location_idx)
            data_list += data.tolist()
        assert (np.array(data_list)<=1).all()
        return scipy.sparse.csc_matrix((np.abs(data_list),(row_list,col_list)),shape=self.matrix.transition_matrix.shape)

    def pco2_var_plot(self,cost_plot=False):
        field_vector = self.pco2.var(axis=0)[self.translation_list]
        field_plot = abs(self.matrix.transition_vector_to_plottable(field_vector))
        x,y,desired_vector,target_vector =  self.get_optimal_float_locations(field_vector,self.pco2_corr,desired_vector_in_plottable=(not cost_plot)) #if cost_plot, the vector will return desired vector in form for calculations
        m,XX,YY = traj_class.matrix.matrix_plot_setup() 
        m.fillcontinents(color='coral',lake_color='aqua')
        m.pcolormesh(XX,YY,np.ma.masked_equal(field_plot,0),cmap=plt.cm.Purples,norm=colors.LogNorm())
        plt.title('PCO2 Variance')
        plt.colorbar()
        m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Purples) # this is a plot for the tendancy of the residence time at a grid cell
        m.scatter(x,y,marker='*',color='g',s=34)
        m = self.plot_latest_soccom_locations(m)
        
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
        plt.figure()
        if line:
            plt.subplot(2,1,1)

        field_vector = self.o2.var(axis=0)[self.translation_list]
        field_plot = abs(self.matrix.transition_vector_to_plottable(field_vector))
        x,y,desired_vector =  self.get_optimal_float_locations(field_vector,self.o2_corr,level=0.45) #if cost_plot, the vector will return desired vector in form for calculations
        m,XX,YY = traj_class.matrix.matrix_plot_setup() 
        m.fillcontinents(color='coral',lake_color='aqua')
        m.pcolormesh(XX,YY,np.ma.masked_equal(field_plot,0),cmap=plt.cm.Reds,norm=colors.LogNorm())
        plt.title('O2 Variance')
        plt.colorbar()
        m.scatter(x,y,marker='*',color='g',s=30,latlon=True)
        m = self.plot_latest_soccom_locations(m)
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
        lat_list = np.arange(float(lat),float(lat)-12,(-1*self.degree_bins[0]))[::-1].tolist()+np.arange(float(lat),float(lat)+12,self.degree_bins[0])[1:].tolist()
        column_index_list = np.arange(lat-12,lat+12,0.5).tolist()
        lon_list = np.arange(float(lon),float(lon)-12,(-1*self.degree_bins[1]))[::-1].tolist()+np.arange(float(lon),float(lon)+12,self.degree_bins[1])[1:].tolist()
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
            self.cor_matrix = load_sparse_csc(self.info.base_file+variable+'_cor_matrix_degree_bins_'+str(self.info.degree_bins)+'.npz')
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
            self.cor_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(self.transition.list),len(self.transition.list)))
            save_sparse_csc(self.info.base_file+variable+'_cor_matrix_degree_bins_'+str(self.info.degree_bins)+'.npz', self.cor_matrix)

    def plot_cm26_cor_ellipse(self,matrix):
        """plots the correlation ellipse of the cm2.6 variable"""
        self.matrix.get_direction_matrix()
        self.matrix.quiver_plot(matrix,arrows=False,scale_factor = 0.1)
        plt.show()
