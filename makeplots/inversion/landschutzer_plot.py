import os, sys
# get an absolute path to the directory that contains mypackage
try:
    inverse_plot_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    inverse_plot_dir = os.path.dirname(os.path.join(os.getcwd(),'dummy'))
sys.path.append(os.path.normpath(os.path.join(inverse_plot_dir, '../')))
sys.path.append(os.path.normpath(os.path.join(inverse_plot_dir, '../../compute')))
from plot_utils import TBasemap
from inverse_plot import InversionPlot
import matplotlib.pyplot as plt
import numpy as np
from compute_utils import find_nearest
import matplotlib.cm as cm
from netCDF4 import Dataset

class LandschutzerPlot(InversionPlot):
    def __init__(self,**kwds):
        super(LandschutzerPlot,self).__init__(**kwds)

    def landschutzer_co2_flux(self,var=False,plot=False):
        file_ = self.base_file+'../spco2_MPI_SOM-FFN_v2018.nc'
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

    def loc_plot(self,filename,variance=False,floats=500):
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
