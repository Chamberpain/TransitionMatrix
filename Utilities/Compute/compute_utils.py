import numpy as np 
import scipy.sparse
from sets import Set
from transition_matrix.compute.trans_read import TransMat

def matrix_difference_compare(matrix_1,matrix_2):
    eig_vals,l_eig_vecs,r_eig_vecs = scipy.linalg.eig(matrix_1.todense(),left=True)
    test_eig_vals,test_l_eig_vecs,test_r_eig_vecs = scipy.linalg.eig(matrix_2.todense(),left=True)
    q = zip(eig_vals,test_eig_vals)

    l2_residual = (matrix_1-matrix_2)**2
    return (q,l2_residual.mean(),l2_residual.data.std())

def matrix_size_match(outer_class,inner_class):
    outer_class_set = Set([tuple(x) for x in outer_class.total_list])
    inner_class_set = Set([tuple(x) for x in inner_class.total_list])
    new_index_list = [list(x) for x in list(outer_class_set&inner_class_set)]

    def make_new_matrix(_class,new_index_list):
        _list = []    
        for k,cell in enumerate(_class.total_list):
            print(k)
            try:
                if type(cell)==tuple:
                    _list.append((k,new_index_list.index(list(cell))))
                else:
                    _list.append((k,new_index_list.index(cell.tolist())))
            except ValueError:
                continue
        translation_dict = dict(_list)
        row_idx,column_idx,data = scipy.sparse.find(_class)

        def idx_loop(translation_dict,idx_list):
            new_list = []
            for dummy_idx in idx_list:
                try: 
                    new_list.append(translation_dict[dummy_idx])
                except KeyError:
                    new_list.append(np.nan)
            return new_list

        new_row_idx = idx_loop(translation_dict,row_idx)
        new_column_idx = idx_loop(translation_dict,column_idx)

        mask = ~((np.isnan(new_row_idx)|np.isnan(new_column_idx)))
        data = data[mask].tolist()
        new_row_idx = np.array(new_row_idx)[mask].tolist()
        new_column_idx = np.array(new_column_idx)[mask].tolist()
        try: 
            number_data = _class.number_data[mask].tolist()
        except TypeError:
            number_data = []
        dummy_mat = scipy.sparse.csc_matrix((data,(new_row_idx,new_column_idx)),shape=(len(new_index_list),len(new_index_list)))
        aug_diag = np.where(np.abs(dummy_mat.sum(axis=0))==0)[1].tolist()
        new_row_idx+=aug_diag
        new_column_idx+=aug_diag
        data+=[1]*len(aug_diag)
        number_data+=[1]*len(aug_diag)
        return TransMat((data,(new_row_idx,new_column_idx)),shape=(len(new_index_list)
                ,len(new_index_list)),total_list=new_index_list
                ,lat_spacing = _class.degree_bins[1],lon_spacing=_class.degree_bins[0]
                ,time_step = _class.time_step,number_data=number_data,traj_file_type=_class.traj_file_type)



    new_outer_class_translation_matrix = make_new_matrix(outer_class,new_index_list)
    new_inner_class_translation_matrix = make_new_matrix(inner_class,new_index_list)

    return(new_outer_class_translation_matrix, new_inner_class_translation_matrix)

def z_test(self,p_1,p_2,n_1,n_2):
    p_1 = np.ma.array(p_1,mask = (n_1==0))
    n_1 = np.ma.array(n_1,mask = (n_1==0))
    p_2 = np.ma.array(p_2,mask = (n_2==0))
    n_2 = np.ma.array(n_2,mask = (n_2==0))      
    z_stat = (p_1-p_2)/np.sqrt(self.transition_matrix.todense()*(1-self.transition_matrix.todense())*(1/n_1+1/n_2))
    assert (np.abs(z_stat)<1.96).data.all()