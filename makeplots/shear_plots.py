from transition_matrix.makeplots.transition_matrix_plot import TransPlot
from transition_matrix.compute.compute_utils import matrix_size_match
import scipy.sparse
from sets import Set
import numpy as np
import matplotlib.pyplot as plt

for time in [60,90,120]:
	for spacing in [[1,1],[2,2]]:
		outer_class = TransMat.load_from_type(traj_type='argo',lat_spacing=spacing[0],lon_spacing=spacing[1],time_step=time)
		inner_class = TransPlot.load_from_type(traj_type='aoml',lat_spacing=spacing[0],lon_spacing=spacing[1],time_step=time)
		outer_class,inner_class = matrix_size_match(outer_class,inner_class)
		row_idx,column_idx,data = scipy.sparse.find(outer_class)
		out_mat = scipy.sparse.csc_matrix((data,(row_idx,column_idx)),shape=outer_class.shape)
		row_idx,column_idx,data = scipy.sparse.find(inner_class)
		inn_mat = scipy.sparse.csc_matrix((data,(row_idx,column_idx)),shape=inner_class.shape)
		result = inn_mat-out_mat
		result = result + scipy.sparse.diags([0.01]*result.shape[0])
		row_idx,column_idx,data = scipy.sparse.find(result)
		plot_mat = TransPlot((data.tolist(),(row_idx.tolist(),column_idx.tolist())),shape=result.shape,total_list=np.array(outer_class.total_list),lat_spacing = outer_class.degree_bins[1],lon_spacing=outer_class.degree_bins[0]
                ,time_step = outer_class.time_step,number_data=outer_class.number_data,traj_file_type=outer_class.traj_file_type)
		plot_mat.quiver_plot()
		plt.show()