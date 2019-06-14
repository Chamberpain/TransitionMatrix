from transition_matrix_compute import transition_class_loader
import numpy as np
import pandas as pd
import random
import scipy
from matplotlib import pyplot as plt
import copy
from itertools import combinations

def matrix_resolution_intercomparison():
    datalist = []
    date_span_limit = 60
    coord_list = [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]
    for outer in coord_list:
        for inner in coord_list:
            for traj_file_type in ['Argo','SOSE']:
                if outer==inner:
                    continue
                lat_outer,lon_outer = outer
                lat_inner,lon_inner = inner
                if (lat_outer%lat_inner==0)&(lon_outer%lon_inner==0): #outer lat will always be greater than inner lat
                    max_len = lat_outer/lat_inner*lon_outer/lon_inner                
                    print('they are divisable')
                    outer_class = transition_class_loader(lat_outer,lon_outer,date_span_limit,traj_file_type)
                    inner_class = transition_class_loader(lat_inner,lon_inner,date_span_limit,traj_file_type)
                    inner_class.trans.df[inner_class.info.end_bin_string]
                    inner_lats,inner_lons = zip(*inner_class.trans.df[inner_class.info.end_bin_string].unique().tolist())
                    outer_lat_relation = pd.cut(inner_lats, outer_class.info.bins_lat,right = False, include_lowest=True,labels=outer_class.info.bins_lat[:-1]).tolist()
                    outer_lon_relation = pd.cut(inner_lons, outer_class.info.bins_lon,right = False, include_lowest=True,labels=outer_class.info.bins_lon[:-1]).tolist()
                    inner_coord_list = zip(inner_lats,inner_lons)
                    outer_coord_relation = zip(outer_lat_relation,outer_lon_relation)
                    coord_dictionary_list = []
                    for token in zip(inner_coord_list,outer_coord_relation):
                        try:
                            coord_dictionary_list.append((inner_class.trans.list.index(list(token[0])),outer_class.trans.list.index(list(token[1]))))
                        except ValueError:
                            continue
                    inner_coord_list,outer_coord_list = zip(*coord_dictionary_list)
                    inner_coord_list = np.array(inner_coord_list)
                    outer_coord_list = np.array(outer_coord_list)
                    test_list = [inner_coord_list[outer_coord_list==x].tolist() for x in np.unique(outer_coord_list)]
                    assert (np.array([len(x) for x in test_list])<=max_len).all()
                    matrix_to_change = inner_class.matrix.transition_matrix

                    new_row_list = []
                    new_col_list = []
                    new_data_list = []
                    for k,col_list in enumerate(test_list):
                        print 'I am working on '+str(k)+' column of outer'
                        token = inner_class.matrix.transition_matrix[:,col_list].mean(axis=1)
                        assert abs(token.sum()-1)<10**-3
                        row_dummy = np.where(token!=0)[0].tolist()
                        col_dummy = [k]*len(row_dummy)
                        data_dummy = token[row_dummy].T.tolist()[0]
                        new_row_list += row_dummy
                        new_col_list += col_dummy
                        new_data_list += data_dummy
                    matrix_holder = scipy.sparse.csc_matrix((new_data_list,(new_row_list,new_col_list)),shape=(inner_class.matrix.transition_matrix.shape[0],outer_class.matrix.transition_matrix.shape[1]))
                    new_row_list = []
                    new_col_list = []
                    new_data_list = []
                    for k,row_list in enumerate(test_list):
                        print 'I am working on '+str(k)+' column of outer'
                        token = np.array(matrix_holder[row_list,:].mean(axis=0))[0]
                        col_dummy = np.where(token!=0)[0].tolist()
                        row_dummy = [k]*len(col_dummy)
                        data_dummy = token[col_dummy].tolist()
                        print data_dummy
                        new_row_list += row_dummy
                        new_col_list += col_dummy
                        new_data_list += data_dummy
                    matrix_holder = scipy.sparse.csc_matrix((new_data_list,(new_row_list,new_col_list)),shape=(outer_class.matrix.transition_matrix.shape))
                    datalist.append(matrix_difference_compare(matrix_holder,outer_class.matrix.transition_matrix))
                else:
                    print('they are not divisable')
                    continue
    return datalist


def matrix_difference_compare(matrix_1,matrix_2):
    eig_vals,l_eig_vecs,r_eig_vecs = scipy.linalg.eig(matrix_1.todense(),left=True)
    test_eig_vals,test_l_eig_vecs,test_r_eig_vecs = scipy.linalg.eig(matrix_2.todense(),left=True)
    q = zip(eig_vals,test_eig_vals)

    l2_residual = (matrix_1-matrix_2)**2
    return (q,l2_residual.mean(),l2_residual.std())


def data_withholding_calc():
datalist = []
date = 60
coord_list = [(4,4),(2,3),(3,6),(1,1),(2,2),(3,3)]
for token in coord_list:
    lat,lon = token
    traj_class = transition_class_loader(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=date,traj_file_type='SOSE')
    df_transition_len = len(traj_class.trans.df)
    for percentage in np.arange(0.95,0.65,-0.05):
        repeat = 10
        while repeat >0:
            print 'repeat is ',repeat 
            repeat-=1
            test_traj_class = copy.deepcopy(traj_class)
            test_traj_class.matrix.trajectory_data_withholding_setup(percentage)
            datalist.append((token,percentage,matrix_difference_compare(traj_class.matrix.transition_matrix,test_traj_class.matrix.transition_matrix)))
return datalist