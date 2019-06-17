from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
import numpy as np
from transition_matrix_compute import transition_class_loader,find_nearest
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

class base_inversion_plot(object):
    def __init__(self,traj_class,time_multiplyer=1):
        self.info = traj_class.info
        self.transition = traj_class.trans
        self.matrix = traj_class.matrix
        matrix = self.matrix.transition_matrix
        for _ in range(time_multiplyer-1):
            matrix = matrix.dot(self.matrix.transition_matrix)
        self.matrix.transition_matrix = matrix
        self.soccom_location = self.get_SOCCOM_vector(age_return=False,future_projection=False)        

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
            if (datetime.datetime.today()-df_holder.Date.tail(1)).dt.days.values[0]>270:
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

    def get_optimal_float_locations(self,field_vector,correlation_matrix,float_number=1000):
        """ accepts a target vecotr in sparse matrix form, returns the ideal float locations and weights to sample that"""

        # self.w[self.w<0.007]=0
        target_vector = field_vector/field_vector.max()+1
        target_vector = np.log(abs(target_vector))
        target_vector = target_vector/target_vector.max() # normalize the target_vector
        soccom_float_result = self.soccom_location
        print 'I have completed the setup'

        soccom_float_result_new = (correlation_matrix.dot(self.matrix.transition_matrix)).dot(soccom_float_result)
        float_result = soccom_float_result_new/soccom_float_result_new.max()
        target_vector = target_vector.flatten()-float_result
        target_vector = np.array(target_vector)
        target_vector[target_vector<0] = 0
        print 'I have subtracted off the SOCCOM vector'

        #desired_vector, residual = scipy.optimize.nnls(np.matrix(self.w.todense()),np.squeeze(target_vector))
        print 'I am starting the optimization'
        optimize_fun = scipy.optimize.lsq_linear(correlation_matrix.dot(self.matrix.transition_matrix),np.squeeze(target_vector),bounds=(0,1.2),verbose=2,max_iter=40)
        desired_vector = optimize_fun.x
        print 'It is optimized'
        lats,lons = zip(*[x for _,x in sorted(zip(desired_vector.tolist(),self.transition.list))[::-1][:float_number]])
        return (lons,lats,self.matrix.transition_vector_to_plottable(desired_vector))

    def plot_latest_soccom_locations(self,m):
        y,x = zip(*np.array(self.transition.list)[self.soccom_location>0])
        m.scatter(x,y,marker='*',color='b',s=34,latlon=True)
        return m

class landschutzer_plot(base_inversion_plot):
    def __init__(self,**kwds):
            super(landschutzer_plot,self).__init__(**kwds)
            self.lat_list = np.load('lat_list.npy')

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

            self.pco2 = np.load('subsampled_pco2.npy')
            pco2_corr = np.load('pco2_corr.npy')
            self.pco2_corr = self.get_sparse_correlation_matrix(pco2_corr)

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
        file_ = self.info.base_file+'spco2_MPI_SOM-FFN_v2018.nc'
        nc_fid = Dataset(file_)
        y = nc_fid['lat'][:]
        x = nc_fid['lon'][:]
        
        data = np.ma.masked_greater(nc_fid['fgco2_smoothed'][:],10**19) 
        if var:
            data = np.nanvar(data,axis=0)
        else:
            XX,YY = np.gradient(np.nanmean(data,axis=0)) # take the time mean, then take the gradient of the 2d array
            data = np.abs(XX+YY)
        x[0] = -180
        x[-1] = 180
        co2_vector = np.zeros([len(self.transition.list),1])
        for n,(lat,lon) in enumerate(self.transition.list):
            lon_index = x.tolist().index(find_nearest(x,lon))
            lat_index = y.tolist().index(find_nearest(y,lat))
            co2_vector[n] = data[lat_index,lon_index]
        if plot:
            co2_plot = abs(self.matrix.transition_vector_to_plottable(co2_vector))
            plt.figure()
            m = Basemap(projection='cyl',fix_aspect=False)
            # m.fillcontinents(color='coral',lake_color='aqua')
            m.drawcoastlines()
            XX,YY = m(self.info.X,self.info.Y)
            m.pcolormesh(XX,YY,np.ma.masked_equal(co2_plot,0),cmap=plt.cm.PRGn)
            plt.colorbar(label='CO2 Flux $gm C/m^2/yr$')
        return co2_vector

    def landschutzer_loc_plot(self,filename,variance=False,floats=500):
        field_vector = self.landschutzer_co2_flux(var=variance,plot=False)
        field_plot = abs(self.matrix.transition_vector_to_plottable(field_vector))
        x,y,desired_vector =  self.get_optimal_float_locations(field_vector,self.pco2_corr,float_number=500)
        k = Basemap(projection='cea',llcrnrlon=-180.,llcrnrlat=-80.,urcrnrlon=180.,urcrnrlat=80,fix_aspect=False)
        k.drawcoastlines()
        XX,YY = k(self.info.X,self.info.Y)
        k.fillcontinents(color='coral',lake_color='aqua')
        k.pcolormesh(XX,YY,np.ma.masked_equal(field_plot,0),cmap=plt.cm.PRGn)
        plt.colorbar()
        if variance:
            plt.title('Variance CO2 Flux')
        else:
            plt.title('Gradient CO2 Flux')
        k.scatter(x,y,marker='*',color='g',s=34,latlon=True)

        k = self.plot_latest_soccom_locations(k)
        plt.savefig(filename)
        plt.close()
