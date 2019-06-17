from plot_utils import TBasemap
from inverse_plot import InversePlot
import matplotlib.pyplot as plt
import numpy as np
from ../compute/compute_utils import find_nearest
import matplotlib.cm as cm
from netCDF4 import Dataset

class LandschutzerPlot(InversePlot):

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
