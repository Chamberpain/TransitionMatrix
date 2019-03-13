from transition_matrix_compute import argo_traj_data,load_sparse_csc,save_sparse_csc,find_nearest
import numpy as np
import pandas as pd
import random
import scipy
from matplotlib import pyplot as plt


class argo_withholding(argo_traj_data):
    def matrix_difference_compare(self,test_matrix):
        eig_vals,l_eig_vecs,r_eig_vecs = scipy.linalg.eig(self.transition_matrix.todense(),left=True)
        test_eig_vals,test_l_eig_vecs,test_r_eig_vecs = scipy.linalg.eig(test_matrix.todense(),left=True)
        q = zip(eig_vals,test_eig_vals,range(len(eig_vals)))
        eig_val_list = sorted(q,key = lambda t: t[0].real)
        mask = [eig_val_list[i][0].real>0.95 for i in range(len(eig_val_list))]
        return_list = []
        num = 0
        for eig_val,test_eig_val,k in np.array(eig_val_list)[mask]:
            k = int(k.real) #this is because this number gets saved as compex
            l_eig_vec = l_eig_vecs[:,k]
            r_eig_vec = r_eig_vecs[:,k]
            test_l_eig_vec = test_l_eig_vecs[:,k]
            test_r_eig_vec = test_r_eig_vecs[:,k]
            l_vec_return = np.sqrt((l_eig_vec-test_l_eig_vec)**2).sum()
            r_vec_return = np.sqrt((r_eig_vec-test_r_eig_vec)**2).sum()
            token = (num,eig_val,test_eig_val,l_vec_return,r_vec_return)
            return_list.append(token)
            num +=1
        return return_list

    def transition_matrix_grid_transform(self):
        return_list = []
        for lat_multiplyer in range(1,7):
            for lon_multiplyer in range(1,7):
                if lat_multiplyer*lon_multiplyer==1:
                    continue 
                lat,lon = self.degree_bins
                lat = lat*lat_multiplyer #starting at lower grid cell resolution and going to higher
                lon = lon*lon_multiplyer
                try:
                    open(self.base_file+'transition_matrix/transition_matrix_degree_bins_'+str((lat,lon))+'_time_step_'+str(self.date_span_limit)+'.npz','r')
                except IOError: #this is triggered when the transition matrix doesnt exist
                    continue
                test_traj_class = argo_traj_data(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=self.date_span_limit,traj_file_type='SOSE') # if the file exists, load it
                lat_list,lon_list = zip(*test_traj_class.total_list)    # get all the lat and lon values
                lower_res_lat_bins = np.sort(np.unique(lat_list))
                lower_res_lon_bins = np.sort(np.unique(lon_list))
                lat_list,lon_list = zip(*self.total_list)
                lat_list = [lower_res_lat_bins[m-1] for m in np.digitize(lat_list,lower_res_lat_bins)]
                lon_list = [lower_res_lon_bins[m-1] for m in np.digitize(lon_list,lower_res_lon_bins)]
                lower_res_total_list = [list(m) for m in zip(lat_list,lon_list)]
                row_list = []
                col_list = []
                data_list = []
                index_list = []
                master_index_list = []
                for n,x in enumerate(test_traj_class.total_list):
                    index_list = [i for i,val in enumerate(lower_res_total_list) if val==x]
                    if not index_list: # this tests if the index list is empty and continues in that case
                        continue 
                    row_list += index_list 
                    col_list += [n]*len(index_list) # this is the column number of the reduced resolution matrix
                    data_list += [1./len(index_list)]*len(index_list)   # this scales the data by the number of grid cells represented
                    assert len(index_list)<=lat_multiplyer*lon_multiplyer
                    master_index_list.append((index_list,n))
                translation_matrix = scipy.sparse.csc_matrix((data_list,(row_list,col_list)),shape=(len(self.total_list),len(test_traj_class.total_list))) # this matrix can translate a higher demensional state vector into a lower dimensional one
                
                data_list = []
                col_list = []
                row_list = []
                for index_list,column_index in master_index_list:
                    averaging_list = []
                    for index in index_list:
                        averaging_list.append(translation_matrix.T.dot(self.transition_matrix[:,index]))
                    result = np.sum(averaging_list)
                    data_list += (result.data/result.sum()).tolist()
                    row_list += result.tocoo().row.tolist()
                    col_list += [column_index]*len(result.data)
                low_res_transition_matrix = scipy.sparse.csc_matrix((data_list,(row_list,col_list)),shape=(len(test_traj_class.total_list),len(test_traj_class.total_list)))
                return_list.append((self.degree_bins,test_traj_class.degree_bins,self.matrix_difference_compare(low_res_transition_matrix,test_traj_class.transition_matrix)))
        return return_list 

    def time_offset_bias_calc(self):
        return_list = []
        for n in range(2,14): # transition matrices are calculated from 20 to 280 days, so this is the maximum
            try:
                test_matrix = load_sparse_csc(self.base_file+'transition_matrix/transition_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(n*self.date_span_limit)+'.npz')
            except NameError:
                continue
            self.load_w.number(n)
            l2 = np.sqrt(((self.w-test_matrix).data**2).sum())
            return_list.append((self.degree_bins,self.date_span_limit,l2))
        return return_list

    def trajectory_data_withholding_setup(self,percentage):
        # number_matrix = load_sparse_csc(self.base_file+'transition_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
        # standard_error_matrix = load_sparse_csc(self.base_file+'transition_matrix/standard_error_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
        df_tape = self.df_transition.copy() # this is a hack, so we dont have to reload every time
        mylist = self.df_transition.Cruise.unique().tolist() # total list of floats
        not_mylist = [mylist[i] for i in sorted(random.sample(xrange(len(mylist)), int(round(len(mylist)*(1-percentage)))))] #randomly select floats to exclude
        df_holder = self.df_transition[self.df_transition.Cruise.isin(not_mylist)]
        self.df_transition = self.df_transition[~self.df_transition.Cruise.isin(not_mylist)]
        row_list = self.transition_matrix.tocoo().row
        col_list = self.transition_matrix.tocoo().col
        data_list = self.transition_matrix.tocoo().data


#needs to be updated due to new startbin procedure... 
        total_index_list = [self.total_list.index(list(ii)) for ii in df_holder['start bin'].unique()]
        # this is the index list for the withheld dataset
        nomatch_startbin = df_holder[~df_holder['start bin'].isin(self.df_transition['start bin'].unique())]['start bin'].unique()
        #these are the starting bins with no match in the transition df dataset and have to be removed because there is no way to compare
        nomatch_index_list = [self.total_list.index(list(ii)) for ii in nomatch_startbin]
        df_holder = df_holder[df_holder['start bin'].isin(self.df_transition['start bin'].unique())]
        #only keep the start bins that have a pair in the left over df transition 

        truth_list1 = ~np.isin(col_list,total_index_list)
        truth_list2 = ~np.isin(row_list,nomatch_index_list)
        test_matrix = scipy.sparse.csc_matrix((data_list[truth_list2],(row_list[truth_list2],col_list[truth_list2])),shape=(len(self.total_list),len(self.total_list)))

        #we include this because it must be taken off the rows because it is impossible to compare rows with no data
        col_list = col_list[truth_list1&truth_list2].tolist() # subtract off the the columns that have changed
        row_list = row_list[truth_list1&truth_list2].tolist()
        data_list = data_list[truth_list1&truth_list2].tolist()

        k = len(df_holder['start bin'].unique())
        for n,index in enumerate(df_holder['start bin'].unique()):
            if n%10==0:
                print 'made it through ',n,' bins. ',(k-n),' remaining'
            dummy_num_list, dummy_data_list,dummy_row_list,dummy_column_list = self.column_compute(index)
            data_list += dummy_data_list
            row_list += dummy_row_list
            col_list += dummy_column_list

        withheld_matrix = scipy.sparse.csc_matrix((data_list,(row_list,col_list)),shape=(len(self.total_list),len(self.total_list)))
        self.df_transition = df_tape
        return self.matrix_difference_compare(withheld_matrix)
        # self.transition_matrix = self.add_noise(self.transition_matrix)

# master_list = []
# for token in [(2,3),(3,6),(3,3)]:
#     lat,lon = token
#     date = [20,40,60,80]
#     traj_class = argo_withholding(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=date,traj_file_type='SOSE')
#     master_list.append(traj_class.transition_matrix_grid_transform())

master_list = []
for token in [(2,3)]:
    lat,lon = token
    date = 20
    traj_class = argo_withholding(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=date,traj_file_type='SOSE')
    df_transition_len = len(traj_class.df_transition)
    for percentage in np.arange(0.95,0.65,-0.05):
        repeat = 
        while repeat >0:
            print 'repeat is ',repeat 
            repeat-=1
            l2 = traj_class.trajectory_data_withholding_setup(percentage)
            assert df_transition_len == len(traj_class.df_transition)
            print 'the length of the df is', len(traj_class.df_transition)
            master_list.append((token,percentage,l2))
token,percentage,eigenspectrum = zip(*master_list)
for percent in np.unique(percentage):
    avg_list = []
    for ii in np.where(np.array(percentage)==percent)[0]:
        test_eig_val,l_vec_return,r_vec_return = zip(*eigenspectrum[ii])
        avg_list.append(list(test_eig_val))
    x = range(len(test_eig_val))
    y = np.array(avg_list).mean(axis=0)
    std = np.array(avg_list).std(axis=0)
    plt.plot(x,y,label='percent = '+str(percent))
    plt.fill_between(x,y-std,y+std)
plt.legend()
plt.show()