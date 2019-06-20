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
import numpy as np

class InversionPlot(TransitionPlot):
    def __init__(self,**kwds):
        super(InversionPlot,self).__init__(**kwds)


    def get_optimal_float_locations(self,field_vector,correlation_matrix,float_number=1000):
        """ accepts a target vecotr in sparse matrix form, returns the ideal float locations and weights to sample that"""

        # self.w[self.w<0.007]=0
        self.get_SOCCOM_vector()
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