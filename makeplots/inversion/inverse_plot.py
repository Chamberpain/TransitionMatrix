import os, sys
# get an absolute path to the directory that contains mypackage
try:
    inverse_plot_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    inverse_plot_dir = os.path.dirname(os.path.join(os.getcwd(),'dummy'))
sys.path.append(os.path.normpath(os.path.join(inverse_plot_dir, '../')))
sys.path.append(os.path.normpath(os.path.join(inverse_plot_dir, '../../compute')))

from transition_matrix_plot import TransitionPlot
from plot_utils import basemap_setup
from argo_data import SOCCOM
import numpy as np
from field_correlation import CM2p6Correlation,GlodapCorrelation
from field_vector import CM2p6Vector,GlodapVector

class InverseBase(object):
    def __init__(self,transition_plot,**kwds):
        self.list = transition_plot.list
        self.base_file=transition_plot.base_file
        self.bins_lat = transition_plot.bins_lat
        self.bins_lon = transition_plot.bins_lon
        self.degree_bins = transition_plot.degree_bins
        self.quiver = transition_plot.quiver_plot
        try:
            self.east_west = transition_plot.east_west
            self.north_south = transition_plot.north_south
        except AttributeError:
            pass

class InversionPlot():
    def __init__(self,transition_plot,correlation_matrix_class,target_vector_class,**kwds):
        self = transition_plot
        self.target = target_vector_class
        self.correlation = correlation_matrix_class

    def get_optimal_float_locations(self,float_number=1000):
        """ accepts a target vecotr in sparse matrix form, returns the ideal float locations and weights to sample that"""
        vector = self.normalize_vector(vector)
        vector = self.remove_soccom_observations(vector)
        print 'I am starting the optimization'
        optimize_fun = scipy.optimize.lsq_linear(correlation_matrix.dot(self.matrix.transition_matrix),np.squeeze(target_vector),bounds=(0,1.2),verbose=2,max_iter=40)
        desired_vector = optimize_fun.x
        print 'It is optimized'
        lats,lons = zip(*[x for _,x in sorted(zip(desired_vector.tolist(),self.transition.list))[::-1][:float_number]])
        return (lons,lats,self.matrix.transition_vector_to_plottable(desired_vector))

        def normalize_vector(self,vector):
    #this needs some extensive unit testing, the normalization is very ad hock
            target_vector = field_vector/field_vector.max()+1
            target_vector = np.log(abs(target_vector))
            target_vector = target_vector/target_vector.max()
            print 'I have normalized the target vector'
            return target_vector

        def instrument_to_observation(self,vector):
            return (self.correlation.matrix.dot(self.transition_matrix)).dot(vector)

        def remove_soccom_observations(self,vector):
            try:
                self.soccom
            except AttributeError:
                self.soccom = SOCCOM(degree_bin_lat=self.bins_lat,degree_bin_lon=self.bins_lon)
            soccom_result = self.instrument_to_observation(self.soccom)
            soccom_result = soccom_result/soccom_result.max()
            vector = vector.flatten()-soccom_result
            vector = np.array(target_vector)
            vector[target_vector<0] = 0
            print 'I have subtracted off the SOCCOM vector'        
            return vector

    def loc_plot(self,filename,variance=False,floats=500):
        field_plot = abs(transition_vector_to_plottable(self.bins_lat,self.bins_lon,self.list,self.target.vector))
        x,y,desired_vector = self.target.get_optimal_float_locations(field_vector,self.correlation.matrix,float_number=500)
        XX,YY,m = basemap_setup(self.bins_lon,self.bins_lat,self.traj_file_type)    
        m = self.target.plot(XX,YY,m)
        m.scatter(x,y,marker='*',color='g',s=34,latlon=True)
        m = self.plot_latest_soccom_locations(m)
        plt.savefig(filename)
        plt.close()

    @classmethod 
    def landschutzer(cls,transition_plot,var):
        return cls(transition_plot,LandschutzerCO2Flux(var),LandschutzerCorr())

    @classmethod 
    def modis(cls,var):
        return cls(MODISVector(var),MOIDSCorr())

    @classmethod 
    def glodap(cls,transition_plot,flux):
        transition_plot.get_direction_matrix()
        corr_matrix_class = GlodapCorrelation(transition_plot=transition_plot)
        target_vector_class = GlodapVector(transition_plot=transition_plot,flux=flux)
        return cls(transition_plot=transition_plot,correlation_matrix_class=corr_matrix_class,target_vector_class=target_vector_class)

    @classmethod 
    def cm2p6(cls,transition_plot,variable,variance):
        corr_matrix_class = CM2p6Correlation(transition_plot=transition_plot,variable=variable)
        target_vector_class = CM2p6Vector(transition_plot=transition_plot,variable=variable,variance=variance)
        return cls(transition_plot=transition_plot,correlation_matrix_class=corr_matrix_class,target_vector_class=target_vector_class)