import matplotlib.pyplot as plt
import numpy as np
if __name__ == '__main__':
    import os, sys
    # get an absolute path to the directory that contains mypackage
    make_plot_dir = os.path.dirname(os.path.join(os.getcwd(),'dummy'))
    sys.path.append(os.path.normpath(os.path.join(make_plot_dir, '../compute')))
    from transition_matrix_compute import TransMatrix,Trajectory
else:
    from ..compute.transition_matrix_compute import TransMatrix,Trajectory


import pandas as pd
import scipy
import matplotlib.colors as colors
import os
import datetime
import matplotlib.cm as cm
from sets import Set
from dateutil.relativedelta import relativedelta
import matplotlib.colors as mcolors         
from plot_utils import basemap_setup


class TrajectoryPlot(Trajectory):
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

class TransitionPlot(TransMatrix):
    def __init__(self,time_multiplyer=1,**kwds):
        super(TransitionPlot,self).__init__(**kwds)

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
        matplotlib.pyplot.savefig('../plots/number_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
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
        matplotlib.pyplot.savefig('../plots/number_matrix/standard_error_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
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
        matplotlib.pyplot.savefig('../plots/transition_plots/'+filename+'_diag_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.png')
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

        m,XX,YY = basemap_setup(self.bins_lon,self.bins_lat,self.traj_file_type)  

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


    def get_SOCCOM_vector(self,age_return=False,future_projection=True):
        path = '/Users/pchamberlain/Projects/transition_matrix/SOCCOM_trajectory/'
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
        y,x = zip(*np.array(self.transition.list)[self.soccom_location>0])
        m.scatter(x,y,marker='*',color='b',s=34,latlon=True)
        return m

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

        cmap = mcolors.LinearSegmentedColormap('my_colormap', cdict, 100) 
        file_ = '../argo_traj_box/ar_index_global_prof.txt'
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
        plt.figure()
        plot_vector = self.matrix.transition_vector_to_plottable(float_vector)
        original_plot_vector = plot_vector
        plot_vector = np.ma.masked_equal(plot_vector,0)
        m,XX,YY = self.matrix.matrix_plot_setup()
        m.pcolor(XX,YY,plot_vector,cmap=cmap,vmax=2,vmin=0)
        plt.colorbar()
        plt.title('density/age for '+str(self.info.degree_bins)+' at 0 days')
        for _ in range(3):
            float_vector = self.matrix.transition_matrix.todense().dot(float_vector)
        plt.figure()
        plot_vector = self.matrix.transition_vector_to_plottable(float_vector)
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

def SOCCOM_death_plot():
    traj_class = transition_class_loader(2,3,100,'Argo')
    traj_plot = argo_traj_plot(traj_class,time_multiplyer=2)
    float_vector = argo_traj_plot.get_SOCCOM_vector(age_return=True)
    plottable = traj_class.matrix.transition_vector_to_plottable(float_vector)
    traj_class.info.traj_file_type = 'SOSE'
    m,XX,YY = basemap_setup(traj_class.bins_lon,traj_class.bins_lat,traj_class.traj_file_type)  
    m.fillcontinents(color='coral',lake_color='aqua')
    m.pcolormesh(XX,YY,np.ma.masked_equal(plottable,0),cmap=plt.cm.magma,vmax = plottable.max()/2)
    plt.title('SOCCOM Sampling Survival PDF in 300 Days')
    plt.colorbar()
    plt.show()
    plt.savefig('survival')


def figure2():
    coord_list = [(2,2)]
    date_list = [100]
    for lat,lon in coord_list:
        for date in date_list:
            traj_class = TransMatrix(lat,lon,date)
            traj_class.get_direction_matrix()
            traj_class.quiver_plot(traj_class.transition_matrix,degree_sep=4,scale_factor=25)
            matplotlib.pyplot.savefig('test')

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

# for token in [(2,2),(2,3),(3,3)]:
#     lat,lon = token
#     for date in [180]:
#         traj_class = argo_traj_plot(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=date)
#         traj_class.plot_cm26_cor_ellipse('co2')