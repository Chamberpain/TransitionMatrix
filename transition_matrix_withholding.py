from transition_matrix_compute import argo_traj_data,load_sparse_csc,save_sparse_csc,find_nearest
import numpy as np
import pandas as pd
import random
import scipy

class argo_withholding(argo_traj_data):

    def trajectory_data_withholding_setup(self,percentage):

        number_matrix = load_sparse_csc(self.base_file+'transition_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
        standard_error_matrix = load_sparse_csc(self.base_file+'transition_matrix/standard_error_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
        df_tape = self.df_transition # this is a hack, so we dont have to reload every time
        mylist = self.df_transition.Cruise.unique().tolist() # total list of floats
        not_mylist = [mylist[i] for i in sorted(random.sample(xrange(len(mylist)), int(round(len(mylist)*(1-percentage)))))] #randomly select floats to exclude
        df_holder = self.df_transition[self.df_transition.Cruise.isin(not_mylist)]
        self.df_transition = self.df_transition[~self.df_transition.Cruise.isin(not_mylist)]
        row_list = self.transition_matrix.tocoo().row
        col_list = self.transition_matrix.tocoo().col
        data_list = self.transition_matrix.tocoo().data
        num_list = number_matrix.tocoo().data

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
        num_list = num_list[truth_list1&truth_list2].tolist()

        k = len(df_holder['start bin'].unique())
        for n,index in enumerate(df_holder['start bin'].unique()):
            print 'made it through ',n,' bins. ',(k-n),' remaining'
            dummy_num_list, dummy_data_list,dummy_row_list,dummy_column_list = self.column_compute(index)
            num_list += dummy_num_list
            data_list += dummy_data_list
            row_list += dummy_row_list
            col_list += dummy_column_list


        withheld_matrix = scipy.sparse.csc_matrix((data_list,(row_list,col_list)),shape=(len(self.total_list),len(self.total_list)))

        return np.sqrt(((test_matrix-withheld_matrix).data**2).sum())
        # self.transition_matrix = self.add_noise(self.transition_matrix)


master_list = []
for token in [(2,3),(3,6),(1,1),(2,2),(3,3),(4,4)]:
    lat,lon = token
    date = 20
    traj_class = argo_withholding(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=date,traj_file_type='SOSE')
    for percentage in np.arange(0.95,0.65,-0.05):
        repeat = 100
        while repeat >0:
            print 'repeat is ',repeat 
            repeat-=1
            l2 = traj_class.trajectory_data_withholding_setup(percentage)
            master_list.append((token,percentage,l2))