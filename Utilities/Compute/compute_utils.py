import numpy as np 
import scipy.sparse
import geopy
from TransitionMatrix.Utilities.Compute.trans_read import TransMat

def matrix_difference_compare(matrix_1,matrix_2):
    # eig_vals,l_eig_vecs,r_eig_vecs = scipy.linalg.eig(matrix_1.todense(),left=True)
    # test_eig_vals,test_l_eig_vecs,test_r_eig_vecs = scipy.linalg.eig(matrix_2.todense(),left=True)
    # q = zip(eig_vals,test_eig_vals)

    l2_residual = (matrix_1-matrix_2).data*(matrix_1-matrix_2).data
    traj_1 = matrix_1.trans_geo
    traj_2 = matrix_2.trans_geo
    return ((traj_1.lat_sep,traj_1.lon_sep,traj_1.time_step,traj_1.file_type),
        (traj_2.lat_sep,traj_2.lon_sep,traj_2.time_step,traj_2.file_type),l2_residual.mean(),l2_residual.std())

def matrix_size_match(matrix_1,matrix_2):
    matrix_1_set = set(matrix_1.trans_geo.total_list.tuple_total_list())
    matrix_2_set = set(matrix_2.trans_geo.total_list.tuple_total_list())
    new_index_list = [geopy.Point(x[0],x[1]) for x in list(matrix_1_set.intersection(matrix_2_set))]

    new_matrix_1 = matrix_1.new_coordinate_list(new_index_list)
    new_matrix_2 = matrix_2.new_coordinate_list(new_index_list)

    return(new_matrix_1, new_matrix_2)

def z_test(self,p_1,p_2,n_1,n_2):
    p_1 = np.ma.array(p_1,mask = (n_1==0))
    n_1 = np.ma.array(n_1,mask = (n_1==0))
    p_2 = np.ma.array(p_2,mask = (n_2==0))
    n_2 = np.ma.array(n_2,mask = (n_2==0))      
    z_stat = (p_1-p_2)/np.sqrt(self.transition_matrix.todense()*(1-self.transition_matrix.todense())*(1/n_1+1/n_2))
    assert (np.abs(z_stat)<1.96).data.all()