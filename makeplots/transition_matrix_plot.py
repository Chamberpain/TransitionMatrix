import matplotlib.pyplot as plt
import numpy as np
from transition_matrix.compute.trans_read import TransMat
import scipy
import matplotlib.colors as colors
import matplotlib.cm as cm
from transition_matrix.makeplots.plot_utils import basemap_setup,transition_vector_to_plottable
from transition_matrix.makeplots.argo_data import SOCCOM


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

        #         self.multiplyer = time_multiplyer
#         matrix = copy.deepcopy(self)
#         for _ in range(time_multiplyer-1):
#             matrix = matrix.dot(self)
#         self = matrix
# routine to propogate transition matrix into the future, can be used for forward or inverse problem


    def number_plot(self): 
        self.number_matrix = self.new_sparse_matrix(self.number_data)
        k = self.number_matrix.sum(axis=0)
        k = k.T
        print k
        bins_lat,bins_lon = self.bins_generator(self.degree_bins)
        number_matrix_plot = transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,k)
        plt.figure('number matrix',figsize=(10,10))
        XX,YY,m = basemap_setup(bins_lat,bins_lon,self.traj_file_type)  
        number_matrix_plot = np.ma.masked_equal(number_matrix_plot,0)   #this needs to be fixed in the plotting routine, because this is just showing the number of particles remaining
        m.pcolormesh(XX,YY,number_matrix_plot,cmap=plt.cm.magma,vmin=0,vmax=300)
        # plt.title('Transition Density',size=30)
        cbar = plt.colorbar()
        cbar.set_label(label='Transition Number',size=30)
        cbar.ax.tick_params(labelsize=30)
        plt.savefig('../plots/number_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.time_step)+'_location_'+str(self.traj_file_type)+'.png')
        plt.close()

    def standard_error_plot(self):

#Todo: the way this is calculated the diagonal goes to infinity if the number of floats remaining in a bin goes to zero. I suspect that I may not be calculating this properly
        self.number_matrix = self.new_sparse_matrix(self.number_data)
        standard_error = np.sqrt(self.data*(np.ones(len(self.data))-self.data)/self.number_matrix.data)
        standard_error = self.new_sparse_matrix(standard_error)
        k = np.diagonal(standard_error.todense())
        standard_error_plot = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,k)
        plt.figure('standard error')
        # m.fillcontinents(color='coral',lake_color='aqua')
        # number_matrix_plot[number_matrix_plot>1000]=1000
        XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type)  
        q = self.number_matrix.sum(axis=0)
        q = q.T
        q = np.ma.masked_equal(q,0)
        standard_error_plot = np.ma.array(standard_error_plot,mask=transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,q)==0)
        m.pcolormesh(XX,YY,standard_error_plot,cmap=plt.cm.cividis)
        plt.title('Standard Error',size=30)
        plt.colorbar(label='Standard Error')
        plt.savefig('../plots/number_matrix/standard_error_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.time_step)+'_location_'+str(self.traj_file_type)+'.png')
        plt.close()

    def transition_matrix_plot(self):
        plt.figure(figsize=(10,10))
        k = np.diagonal(self.todense())
        transition_plot = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,k)
        XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type)  
        k = self.number_matrix.sum(axis=0)
        k = k.T
        transition_plot = np.ma.array(100*(1-transition_plot),mask=transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,k)==0)
        m.pcolormesh(XX,YY,transition_plot,vmin=0,vmax=100) # this is a plot for the tendancy of the residence time at a grid cell
        cbar = plt.colorbar()
        cbar.ax.tick_params(labelsize=30)
        cbar.set_label('% Particles Dispersed',size=30)
        # plt.title('1 - diagonal of transition matrix',size=30)
        plt.savefig('../plots/transition_plots/trans_diag_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.time_step)+'_location_'+str(self.traj_file_type)+'.png')
        plt.close()


    def quiver_plot(self,m=False,arrows=True,degree_sep=4,scale_factor=.2,ellipses=True):
        """
        This plots the mean transition quiver as well as the variance ellipses
        """
# todo: need to check this over. I think there might be a bug.
        row_list, column_list, data_array = scipy.sparse.find(self)
        self.get_direction_matrix()
        east_west_data = self.east_west[row_list,column_list]*data_array
        north_south_data = self.north_south[row_list,column_list]*data_array
        east_west = self.new_sparse_matrix(east_west_data)
        north_south = self.new_sparse_matrix(north_south_data)

        print 'I have succesfully multiplied the transition_matrices'


        if not m:
            bins_lat,bins_lon = self.bins_generator(self.degree_bins)
            dummy,dummy,m = basemap_setup(bins_lon,bins_lat,self.traj_file_type,fill_color=False)  
        m.drawmapboundary()
        m.fillcontinents()
        Y_ = np.arange(-68.0,66.0,degree_sep)
        X_ = np.arange(-170.0,170.0,degree_sep)
        XX,YY = np.meshgrid(X_,Y_)
        n_s = np.zeros(XX.shape)
        e_w = np.zeros(XX.shape)
        XX,YY = m(XX,YY)
        if ellipses:
            for i,lat in enumerate(Y_):
                for k,lon in enumerate(X_):
                    print 'lat = ',lat
                    print 'lon = ',lon 
                    try:
                        index = self.total_list.tolist().index([lat,lon])
                    except ValueError:
                        print 'There was a value error in the total list'
                        continue
                    n_s[i,k] = north_south[:,index].mean()
                    e_w[i,k] = east_west[:,index].mean()
                    y = north_south[:,index].data
                    x = east_west[:,index].data
                    mask = (x!=0)|(y!=0) 
                    x = x[mask]
                    y = y[mask]
                    try:
                        w,v = np.linalg.eig(np.cov(x,y))
                    except:
                        continue
                    angle = np.degrees(np.arctan(v[1,np.argmax(w)]/v[0,np.argmax(w)]))+90
                    axis1 = np.log(2*max(w)*np.sqrt(5.991))

                    axis2 = np.log(2*min(w)*np.sqrt(5.991))

                    print 'angle = ',angle
                    print 'axis1 = ',axis1
                    print 'axis2 = ',axis2
                    try:
                        poly = m.ellipse(lon, lat, axis1*scale_factor,axis2*scale_factor, 80,phi=angle,line=False, facecolor='green', zorder=3,alpha=0.35)
                    except ValueError:
                        print ' there was a value error in the calculation of the transition_matrix'
                        continue
        if arrows:
            e_w = np.ma.array(e_w,mask=(e_w==0))
            n_s = np.ma.array(n_s,mask=(n_s==0))            
            m.quiver(XX,YY,e_w,n_s,scale=.01)
        return m

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
        print plot_vector.sum()-original_plot_vector.sum()
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

    def eig_vec_plot(self,eig_vec):
        eig_vec_token = transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.total_list,eig_vec)
        eig_vec_token = np.ma.masked_equal(eig_vec_token,0) 
        XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type) 
        mag = abs(eig_vec_token).max()
        m.pcolormesh(XX,YY,eig_vec_token,norm=colors.SymLogNorm(linthresh=0.000003, linscale=0.000003,
                                              vmin=-mag/50, vmax=mag/50))

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
            print 'plotting eigenvector '+str(k)+' for '+str(self.degree_bins)+' eigenvalue is '+str(eig_val)
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
    traj_class = TransitionPlot(date_span_limit=180,time_multiplyer=4)
    float_vector = SOCCOM(transition_plot=traj_class,age_return=True)
    plottable = transition_vector_to_plottable(traj_class.bins_lat,traj_class.bins_lon,traj_class.list,traj_class.transition_matrix.dot(float_vector.vector))
    traj_class.traj_file_type = 'SOSE'
    XX,YY,m = basemap_setup(traj_class.bins_lat,traj_class.bins_lon,traj_class.traj_file_type)  
    m.pcolormesh(XX,YY,np.ma.masked_less(plottable,5*10**-3),cmap=plt.cm.magma,vmax = plottable.max()/2)
    plt.title('SOCCOM Sampling Survival PDF in 720 Days')
    plt.colorbar(label='Probability Density/Age')
    plt.savefig('survival')
    plt.show()


def figure2():
    coord_list = [(2,2)]
    date_list = [100]
    for lat,lon in coord_list:
        for date in date_list:
            traj_class = TransMatrix(lat,lon,date)
            traj_class.get_direction_matrix()
            traj_class.quiver_plot(traj_class.transition_matrix,degree_sep=4,scale_factor=25)
            plt.savefig('test')

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