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

class TransitionPlot(TransMatrix):
    def __init__(self,traj_class,time_multiplyer=1):
        
        
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


    def plot_latest_soccom_locations(self,m):
        y,x = zip(*np.array(self.transition.list)[self.soccom_location>0])
        m.scatter(x,y,marker='*',color='b',s=34,latlon=True)
        return m

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
    m,XX,YY = traj_class.matrix.matrix_plot_setup() 
    m.fillcontinents(color='coral',lake_color='aqua')
    m.pcolormesh(XX,YY,np.ma.masked_equal(plottable,0),cmap=plt.cm.magma,vmax = plottable.max()/2)
    plt.title('SOCCOM Sampling Survival PDF in 300 Days')
    plt.colorbar()
    plt.show()
    plt.savefig('survival')
# for token in [(2,2),(2,3),(3,3)]:
#     lat,lon = token
#     for date in [180]:
#         traj_class = argo_traj_plot(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=date)
#         traj_class.plot_cm26_cor_ellipse('co2')