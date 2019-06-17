import numpy as np 
import scipy.sparse


def find_nearest(items, pivot):
    return min(items, key=lambda x: abs(x - pivot))

def save_sparse_csc(filename, array):
    np.savez(filename, data=array.data, indices=array.indices,
             indptr=array.indptr, shape=array.shape)

def load_sparse_csc(filename):
    loader = np.load(filename)
    return scipy.sparse.csc_matrix((loader['data'], loader['indices'], loader['indptr']),
                      shape=loader['shape'])

def matrix_difference_compare(matrix_1,matrix_2):
    eig_vals,l_eig_vecs,r_eig_vecs = scipy.linalg.eig(matrix_1.todense(),left=True)
    test_eig_vals,test_l_eig_vecs,test_r_eig_vecs = scipy.linalg.eig(matrix_2.todense(),left=True)
    q = zip(eig_vals,test_eig_vals)

    l2_residual = (matrix_1-matrix_2)**2
    return (q,l2_residual.mean(),l2_residual.data.std())

def z_test(self,p_1,p_2,n_1,n_2,transition_matrix):
    p_1 = np.ma.array(p_1,mask = (n_1==0))
    n_1 = np.ma.array(n_1,mask = (n_1==0))
    p_2 = np.ma.array(p_2,mask = (n_2==0))
    n_2 = np.ma.array(n_2,mask = (n_2==0))      
    z_stat = (p_1-p_2)/np.sqrt(self.transition_matrix.todense()*(1-transition_matrix.todense())*(1/n_1+1/n_2))
    assert (np.abs(z_stat)<1.96).data.all()

def matrix_size_match(outer_class,inner_class):
    outer_class_set = Set([tuple(x) for x in outer_class.trans.list])
    inner_class_set = Set([tuple(x) for x in inner_class.trans.list])
    new_index_list = [list(x) for x in list(outer_class_set&inner_class_set)]

    old_outer_class_list = copy.deepcopy(outer_class.trans.list)
    old_inner_class_list = copy.deepcopy(inner_class.trans.list)

    outer_class.list = new_index_list
    inner_class.list = new_index_list

    outer_class.matrix.transition_matrix = outer_class.matrix_recalc([],old_outer_class_list,outer_class.transition_matrix,noise=False)
    inner_class.matrix.transition_matrix = inner_class.matrix_recalc([],old_inner_class_list,inner_class.transition_matrix,noise=False)
    return(outer_class, inner_class)

def z_test(self,p_1,p_2,n_1,n_2):
    p_1 = np.ma.array(p_1,mask = (n_1==0))
    n_1 = np.ma.array(n_1,mask = (n_1==0))
    p_2 = np.ma.array(p_2,mask = (n_2==0))
    n_2 = np.ma.array(n_2,mask = (n_2==0))      
    z_stat = (p_1-p_2)/np.sqrt(self.transition_matrix.todense()*(1-self.transition_matrix.todense())*(1/n_1+1/n_2))
    assert (np.abs(z_stat)<1.96).data.all()