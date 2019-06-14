from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from transition_matrix_compute import transition_class_loader
from scipy.interpolate import griddata
import pandas as pd
import pyproj
from matplotlib.patches import Polygon
from itertools import groupby
import pickle
import scipy
import matplotlib.colors as colors
import pickle
import os
import datetime
import matplotlib.cm as cm
from sets import Set
from dateutil.relativedelta import relativedelta
import matplotlib.colors as mcolors         
from netCDF4 import Dataset
"""compiles and compares transition matrix from trajectory data. """

def resolution_difference_plot():
    with open('transition_matrix_resolution_comparison.pickle','rb') as fp:
        datalist = pickle.load(fp)
    inner,outer,traj_file_type,actual_data = zip(*datalist)
    q,residual_mean,residual_std = zip(*actual_data)
    inner_lat,inner_lon = zip(*inner)
    outer_lat,outer_lon = zip(*outer)
    x_coord = np.array(outer_lon)/np.array(inner_lon)
    y_coord = np.array(outer_lat)/np.array(inner_lat)



    label_list = [str(x[0])+' to '+str(x[1]) for x in zip(inner,outer)]
    colors = cm.rainbow(np.linspace(0, 1, len(x_coord)))
    for n,plot_type in enumerate(np.unique(traj_file_type).tolist()):
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        mask = np.array(traj_file_type) == plot_type
        mean_token = np.array(residual_mean)[mask]
        


        for x_coord_token,y_coord_token,mean_token,std_token,label in zip(np.array(x_coord)[mask],np.array(y_coord)[mask],np.array(residual_mean)[mask],np.array(residual_std)[mask],np.array(label_list)[mask]):
            if n == 0:
                ax.errorbar(x_coord_token,y_coord_token,yerr=std_token*25,xerr=std_token*25,marker = 'o',markersize=mean_token*50000,zorder=1/mean_token,label=label)
                plt.title('Argo Resolution Difference Uncertainty')
            if n ==1:
                ax.errorbar(x_coord_token,y_coord_token,yerr=std_token*25,xerr=std_token*25,marker = 'o',markersize=mean_token*50000,zorder=1/mean_token,label=label)
                plt.title('SOSE Resolution Difference Uncertainty')
        plt.xlabel('Ratio of Longitude Resolution')
        plt.ylabel('Ratio of Latitude Resolution')
        plt.legend()

    ax.set_xlabel('Comparison Matrix Timestep')
    ax.set_ylabel('Mean L2 Norm Difference')
    plt.legend()
    plt.savefig('date_difference_l2')
    plt.close()

def date_difference_plot():
    with open('transition_matrix_datespace_data.pickle', 'rb') as fp:
        datalist = pickle.load(fp)
    date,traj_plot,actual_data = zip(*datalist)
    q,residual_mean,residual_std = zip(*actual_data)

    date = np.array(date)
    traj_plot = np.array(traj_plot)
    residual_mean = np.array(residual_mean)
    residual_std = np.array(residual_std)
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for n,plot_type in enumerate(np.unique(traj_plot).tolist()):
        mask = traj_plot == plot_type
        date_token = date[mask]
        mean_token = residual_mean[mask]
        std_token = residual_std[mask]
        if n == 0:
            ax.errorbar(date_token,mean_token, yerr=std_token, fmt='o',markersize=12,label=plot_type)
        if n ==1:
            ax.errorbar(date_token,mean_token, yerr=std_token, fmt='o',markersize=6,label=plot_type,zorder=10,alpha =.8)

    ax.set_xlabel('Comparison Matrix Timestep')
    ax.set_ylabel('Mean L2 Norm Difference')
    plt.legend()
    plt.savefig('date_difference_l2')
    plt.close()

    for plot_type in np.unique(traj_plot).tolist():   
        plt.figure()
        plt.title('Difference in Eigen Spectrum for '+plot_type+' Transition Matrices')
        plt.xlabel('Original Eigen Value')
        plt.ylabel('Difference')
        mask = traj_plot==plot_type
        date_token = date[mask]
        q_token = np.array(q)[mask]
        for n,d in enumerate(date_token):
            plot_label = str(d)+' days'
            eigen_spectrum,test_eigen_spectrum = zip(*q_token[n])
            eigen_spectrum = np.sort([np.absolute(x) for x in eigen_spectrum])
            test_eigen_spectrum = np.sort([np.absolute(x) for x in test_eigen_spectrum])
            mask = eigen_spectrum>0.8
            
            diff = eigen_spectrum[mask]-test_eigen_spectrum[mask]

            plt.plot(eigen_spectrum[mask],diff,label=plot_label)
        plt.legend()
        plt.savefig('date_difference_'+plot_type)
        plt.close()


def data_withholding_plot():

    with open('transition_matrix_withholding_data.pickle', 'rb') as fp:
        datalist = pickle.load(fp)
    token,percentage,actual_data = zip(*datalist)
    token = [str(x) for x in token]
    token = np.array(token)
    percentage = np.array(percentage)
    actual_data = np.array(actual_data)
    for t in np.unique(token):
        plt.figure()
        plt.title('Data Withholding for '+str(t))
        mask = token==t
        percentage_token = percentage[mask]
        actual_data_token = actual_data[mask]
        mean_list = []
        std_list = []
        for p in np.unique(percentage_token):
            mask = percentage_token ==p
            plot_label = str(round((1-p)*100))+'% withheld'
            data = actual_data_token[mask]
            eig_list = []
            mean_list_holder = []
            std_list_holder = []
            for n,d in enumerate(data):
                eigs,l2_mean,l2_std = d
                mean_list_holder.append(l2_mean)
                std_list_holder.append(l2_std)
                eigen_spectrum,test_eigen_spectrum = zip(*eigs)
                eigen_spectrum = np.sort([np.absolute(x) for x in eigen_spectrum])
                test_eigen_spectrum = np.sort([np.absolute(x) for x in test_eigen_spectrum])
                if n == 0:
                    eig_x_coord = eigen_spectrum[eigen_spectrum>0.8]
                    eig_len = len(eig_x_coord)
                diff = eigen_spectrum[-eig_len:]-test_eigen_spectrum[-eig_len:]
                eig_list.append(diff.tolist())
            mean_list.append(np.mean(mean_list_holder))
            std_list.append(np.mean(std_list_holder))            
            plt.plot(eig_x_coord,np.array(eig_list).mean(axis=0),label=plot_label)
        plt.legend()
        plt.xlabel('Eigen Value')
        plt.ylabel('Mean Eigen Value Difference')
        plt.savefig('data_withholding_'+str(t))
        plt.close()
        plt.figure('l2 error')
        plt.errorbar(np.round((1-np.unique(percentage_token))*100),mean_list,yerr=std_list,fmt='o',label=str(t))
    plt.figure('l2 error')
    plt.xlabel('% withheld')
    plt.ylabel('Mean L2 Difference')
    plt.legend()
    plt.savefig('combined_withholding_L2')





def resolution_difference_plot():
    with open('transition_matrix_resolution_comparison.pickle', 'rb') as fp:
        datalist = pickle.load(fp)
    inner,outer,traj_file_type, actual_data = zip(*datalist)
    inner = np.array([str(x) for x in inner])
    outer = np.array([str(x) for x in outer])
    traj_file_type = np.array(traj_file_type)
    actual_data = np.array(actual_data)
    for plot_type in np.unique(traj_file_type):
        plt.figure()
        plt.title(plot_type+' Eigen Spectrum')
        mean_list = []
        std_list = []
        mask = traj_file_type == plot_type
        inner_holder = inner[mask]
        outer_holder = outer[mask]
        data_holder = actual_data[mask]
        for n,(inn,out,dat) in enumerate(zip(inner_holder, outer_holder, data_holder)):
            plot_label = str(inn)+','+str(out)
            eigs,l2_mean,l2_std = dat
            mean_list.append(l2_mean)
            std_list.append(std_list)
            eigen_spectrum,test_eigen_spectrum = zip(*eigs)
            eigen_spectrum = np.sort([np.absolute(x) for x in eigen_spectrum])
            test_eigen_spectrum = np.sort([np.absolute(x) for x in test_eigen_spectrum])
            eig_x_coord = eigen_spectrum[eigen_spectrum>0.8]
            eig_len = len(eig_x_coord)
            diff = eigen_spectrum[-eig_len:]-test_eigen_spectrum[-eig_len:]
            plt.plot(eig_x_coord,diff,label=plot_label)

def SOCCOM_death_plot():
    traj_class = transition_class_loader(2,3,100,'Argo')
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

class argo_traj_plot():
    def __init__(self,traj_class,time_multiplyer=1):
        self.info = traj_class.info
        self.transition = traj_class.trans
        self.matrix = traj_class.matrix
        self.lat_list = np.load('lat_list.npy')

        matrix = self.matrix.transition_matrix
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

    def z_test(self,p_1,p_2,n_1,n_2):
        p_1 = np.ma.array(p_1,mask = (n_1==0))
        n_1 = np.ma.array(n_1,mask = (n_1==0))
        p_2 = np.ma.array(p_2,mask = (n_2==0))
        n_2 = np.ma.array(n_2,mask = (n_2==0))      
        z_stat = (p_1-p_2)/np.sqrt(self.transition_matrix.todense()*(1-self.transition_matrix.todense())*(1/n_1+1/n_2))
        assert (np.abs(z_stat)<1.96).data.all()


    def plot_latest_soccom_locations(self,m):
        y,x = zip(*np.array(self.transition.list)[self.soccom_location>0])
        m.scatter(x,y,marker='*',color='b',s=34,latlon=True)
        return m

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

# for token in [(2,2),(2,3),(3,3)]:
#     lat,lon = token
#     for date in [180]:
#         traj_class = argo_traj_plot(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=date)
#         traj_class.plot_cm26_cor_ellipse('co2')