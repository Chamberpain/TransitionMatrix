import os, sys
# get an absolute path to the directory that contains mypackage
try:
    inverse_plot_dir = os.path.dirname(os.path.realpath(__file__))
except NameError:
    inverse_plot_dir = os.path.dirname(os.path.join(os.getcwd(),'dummy'))
sys.path.append(os.path.normpath(os.path.join(inverse_plot_dir, '../')))
sys.path.append(os.path.normpath(os.path.join(inverse_plot_dir, '../../compute')))

from transition_matrix_plot import TransitionPlot
from plot_utils import basemap_setup,transition_vector_to_plottable
from argo_data import SOCCOM
import numpy as np
from target_load import CM2p6Correlation,CM2p6VectorSpatialGradient,CM2p6VectorTemporalVariance,CM2p6VectorMean,GlodapCorrelation,GlodapVector
import matplotlib.pyplot as plt
import scipy


class InversionPlot(TransitionPlot):
    def __init__(self,correlation_matrix_class,target_vector_class,float_class,**kwds):
        super(InversionPlot,self).__init__(**kwds)
        self.target = target_vector_class
        self.correlation = correlation_matrix_class
        self.float_class = float_class

    def get_optimal_float_locations(self,vector,float_number=1000,vector_exploration_factor=0,corr_exploration_factor=0):
        """ accepts a target vecotr in sparse matrix form, returns the ideal float locations and weights to sample that"""
        vector = self.normalize_vector(vector,vector_exploration_factor)
        vector = self.remove_float_observations(vector)
        self.normalize_correlation(corr_exploration_factor)
        print 'I am starting the optimization'
        optimize_fun = scipy.optimize.lsq_linear(self.correlation.matrix.dot(self.transition_matrix),\
            np.squeeze(vector),bounds=(0,1.2),verbose=2,max_iter=20)
        desired_vector = optimize_fun.x
        print 'It is optimized'
        lats,lons = zip(*[x for _,x in sorted(zip(desired_vector.tolist(),self.list))[::-1][:float_number]])
        return (lons,lats,desired_vector)

    def normalize_vector(self,vector,exploration_factor):
#this needs some extensive unit testing, the normalization is very ad hock
        target_vector = vector-vector.min()
        target_vector = target_vector/target_vector.max()+1
        # target_vector = np.log(abs(target_vector))
        target_vector = target_vector+exploration_factor*target_vector.mean()*np.random.random(target_vector.shape)
        target_vector = target_vector/target_vector.max()
        print 'I have normalized the target vector'
        return target_vector

    def normalize_correlation(self,exploration_factor):
        scale = np.exp(-(self.correlation.east_west**2)/(exploration_factor/2)\
            -(self.correlation.north_south**2)/(exploration_factor/4))
        scale = scale*np.random.random(scale.shape)
        self.correlation.matrix+=scale
        self.correlation.matrix = scipy.sparse.csc_matrix(self.correlation.matrix)
        self.correlation.matrix = self.rescale_matrix(self.correlation.matrix)

    def instrument_to_observation(self,vector):
        return (self.correlation.matrix.dot(self.transition_matrix)).dot(vector)

    def remove_float_observations(self,vector):
        float_result = self.instrument_to_observation(self.float_class.vector)
        float_result = float_result/float_result.max()
        vector = vector.flatten()-float_result
        vector = np.array(vector)
        vector[vector<0] = 0
        print 'I have subtracted off the SOCCOM vector'        
        return vector

    def loc_plot(self,variance=False,floats=500,corr_exploration_factor=0,vector_exploration_factor=0):
        x,y,desired_vector = self.get_optimal_float_locations(self.target.vector,float_number=floats,corr_exploration_factor=corr_exploration_factor,vector_exploration_factor=vector_exploration_factor)
        XX,YY,m = basemap_setup(self.bins_lat,self.bins_lon,self.traj_file_type)    
        dummy,dummy,m = self.target.plot(XX=XX,YY=YY,m=m)
        m.scatter(x,y,marker='*',color='g',s=34,latlon=True)
        m = self.float_class.plot(m=m)
        return (m,desired_vector)

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
        float_class = SOCCOM(transition_plot=transition_plot)
        return cls(correlation_matrix_class=corr_matrix_class,target_vector_class=target_vector_class,float_class=float_class)

    @classmethod 
    def cm2p6(cls,transition_plot,variable,variance):
        variance_class_dict = {'spatial':CM2p6VectorSpatialGradient,'time':CM2p6VectorTemporalVariance,'mean':CM2p6VectorMean}
        corr_matrix_class = CM2p6Correlation(transition_plot=transition_plot,variable=variable)
        target_vector_class = variance_class_dict[variance](transition_plot=transition_plot,variable=variable)
        float_class = SOCCOM(transition_plot=transition_plot)
        return cls(correlation_matrix_class=corr_matrix_class,target_vector_class=target_vector_class,float_class=float_class)


trans_plot = TransitionPlot()
trans_plot.get_direction_matrix()
factor_list = [0,2,5,10]
for corr_factor in factor_list:
    for vector_factor in factor_list:
        for variance in ['spatial','time','mean']:
            for variable in ['surf_o2','surf_dic','surf_pco2','100m_dic','100m_o2']:
                ip = InversionPlot.cm2p6(transition_plot=trans_plot,variable=variable,variance=variance)
                filename = '../../plots/cm2p6_'+variable+'_'+variance+'_explorationcorr_'+str(corr_factor)+'_vectorcorr_'+str(vector_factor)
                ip.loc_plot(corr_exploration_factor=corr_factor,vector_exploration_factor=vector_factor)
                plt.title(variance+' '+variable)
                plt.savefig(filename)
                plt.close()