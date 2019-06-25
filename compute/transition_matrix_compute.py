import pandas as pd
import numpy as np
import os
import pickle
import datetime
import scipy.optimize
from itertools import groupby  
import random
import copy
from sets import Set
from compute_utils import find_nearest,save_sparse_csc,load_sparse_csc

class BaseInfo(object):
    def __init__(self,degree_bin_lat=2,degree_bin_lon=2,date_span_limit=60,traj_file_type='Argo',float_type=None,season=None):
        self.season = season
        self.float_type=float_type
        self.traj_file_type = traj_file_type
        self.project_base = '/Users/pchamberlain/Projects/transition_matrix/'
        if self.traj_file_type=='Argo':
            self.base_file = self.project_base+'data/global_argo/'  #this is a total hack to easily change the code to run sose particles
            # self.base_file = os.getenv("HOME")+'/transition_matrix/'
            print 'I am loading Argo'
        elif self.traj_file_type=='SOSE':
            self.base_file = self.project_base+'data/sose/'
            # self.base_file = os.getenv("HOME")+'/transition_matrix/sose/'
            print 'I am loading SOSE data'
        if self.traj_file_type=='Crete':
            self.base_file = self.project_base+'data/crete/'  #this is a total hack to easily change the code to run sose particles
            # self.base_file = os.getenv("HOME")+'/transition_matrix/'
            print 'I am loading Crete'
        print 'I have started BaseInfo'
        self.degree_bins = (degree_bin_lat,degree_bin_lon)
        self.date_span_limit = date_span_limit
        self.bins_lat = np.arange(-90,90.1,self.degree_bins[0]).tolist()
        self.bins_lon = np.arange(-180,180.1,self.degree_bins[1]).tolist()
        self.end_bin_string = 'end bin '+str(self.date_span_limit)+' day' # we compute the transition df for many different date spans, here is where we find that column
        if 180.0 not in self.bins_lon:
            print '180 is not divisable by the degree bins chosen'      #need to add this logic for the degree bin choices that do not end at 180.
            raise
        self.name = self.base_file +'matrix_class_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)

class Trajectory(BaseInfo):
    def __init__(self,**kwds):
        super(Trajectory,self).__init__(**kwds)
        self.load_dataframe(pd.read_pickle(self.base_file+'all_argo_traj.pickle'))

    def load_dataframe(self,dataframe):
        if self.traj_file_type=='SOSE':
            dataframe=dataframe[dataframe.Lat<-36] # this cuts off floats that wander north of 35
        dataframe = dataframe.sort_values(by=['Cruise','Date'])
        dataframe['bins_lat'] = pd.cut(dataframe.Lat,bins = self.bins_lat,labels=self.bins_lat[:-1])
        dataframe['bins_lon'] = pd.cut(dataframe.Lon,bins = self.bins_lon,labels=self.bins_lon[:-1])
        dataframe['bin_index'] = zip(dataframe['bins_lat'].values,dataframe['bins_lon'].values)
        dataframe = dataframe.reset_index(drop=True)
        #implement tests of the dataset so that all values are within the known domain
        assert dataframe.Lon.min() >= -180
        assert dataframe.Lon.max() <= 180
        assert dataframe.Lat.max() <= 90
        assert dataframe.Lat.min() >=-90
        print 'Trajectory dataframe passed necessary tests'
        self.df = dataframe

class Transition(Trajectory):
    def __init__(self,**kwds):
        super(Transition,self).__init__(**kwds)
        try:
            self.load_df_and_list(pd.read_pickle(\
                self.base_file+'transition_df_degree_bins_'+str(self.degree_bins[0])\
                +'_'+str(self.degree_bins[1])+'.pickle') \
                ,float_type = self.float_type,season = self.season)
        except IOError: #this is the case that the file could not load
            print 'i could not load the transition df, I am recompiling with degree step size', self.degree_bins,''
            print 'file was '+self.base_file+'transition_df_degree_bins_'\
                +str(self.degree_bins[0])+'_'+str(self.degree_bins[1])+'.pickle'
            self.recompile_df(traj.df)

    def load_df_and_list(self,dataframe,float_type=None,season=None):
        if float_type:
            dataframe = dataframe[dataframe['position type']==float_type]
        if season:
            dataframe[dataframe.date.dt.month.isin(season)] # where season is a list of months
        dataframe = dataframe.dropna(subset=[self.end_bin_string]) 
        dataframe = dataframe.dropna(subset=['start bin']) 
        assert ~dataframe['start bin'].isnull().any()
        assert ~dataframe[self.end_bin_string].isnull().any()
        dataframe = dataframe[dataframe['start bin'].isin(dataframe[self.end_bin_string].unique())]
        token = dataframe[dataframe[self.end_bin_string]!=dataframe['start bin']].drop_duplicates().groupby(self.end_bin_string).count() 
        total_list = [list(x) for x in token[token['start bin']>0].index.unique().values.tolist()] # this will make a unique "total list" for every transition matrix, but is necessary so that we dont get "holes" in the transition matrix
        dataframe = dataframe[dataframe[self.end_bin_string].isin([tuple(x) for x in total_list])]
        print 'Transition dataframe passed all necessary tests'
        self.df = dataframe
        self.list = total_list

    def recompile_df(self,dataframe,dump=True):
        """
        from self.df, calculate the dataframe that is used to create the transition matrix

        input: dump - logical variable that is used to determine whether to save the dataframe
        """
        cruise_list = []
        start_bin_list = []
        final_bin_list = []
        end_bin_list = []
        date_span_list = []
        start_date_list = []
        position_type_list = []
        time_delta_list = np.arange(20,300,20)
        k = len(dataframe.Cruise.unique())
        for n,cruise in enumerate(dataframe.Cruise.unique()):
            print 'cruise is ',cruise,'there are ',k-n,' cruises remaining'
            print 'start bin list is ',len(start_bin_list),' long'
            mask = dataframe.Cruise==cruise         #pick out the cruise data from the df
            df_holder = dataframe[mask]
            time_lag = 30   #we assume a decorrelation timescale of 30 days
            time_bins = np.arange(-0.001,(df_holder.Date.max()-df_holder.Date.min()).days,time_lag).tolist()
            df_holder['Time From Deployment'] = [(dummy-df_holder.Date.min()).days + (dummy-df_holder.Date.min()).seconds/float(3600*24) for dummy in df_holder.Date]
            #time from deployment is calculated like this to have the fractional day component
            assert (df_holder['Time From Deployment'].diff().tail(len(df_holder)-1)>0).all() # test that these are all positive and non zero

            max_date = df_holder.Date.max()
            df_holder['time_bins'] = pd.cut(df_holder['Time From Deployment'],bins = time_bins,labels=time_bins[:-1])
            #cut df_holder into a series of time bins, then drop duplicate time bins and only keep the first, this enforces the decorrelation criteria
            for row in df_holder.dropna(subset=['time_bins']).drop_duplicates(subset=['time_bins'],keep='first').iterrows():
                dummy, row = row
                cruise_list.append(row['Cruise']) # record cruise information
                start_date_list.append(row['Date']) # record date information
                start_bin_list.append(row['bin_index']) # record which bin was started in  
                position_type_list.append(row['position type']) # record positioning information
                location_tuple = []
                for time_addition in [datetime.timedelta(days=x) for x in time_delta_list]: # for all the time additions that we need to calculate
                    final_date = row['Date']+time_addition # final date is initial date plus the addition
                    if final_date>max_date:
                        location_tuple.append(np.nan) #if the summed date is greater than the record, do not include in any transition matrix calculations
                    else:
                        final_dataframe_date = find_nearest(df_holder.Date,final_date) #find the nearest record to the final date
                        if abs((final_dataframe_date-final_date).days)>30: 
                            location_tuple.append(np.nan) #if this nearest record is greater than 30 days away from positioning, exclude
                        else:
                            location_tuple.append(df_holder[df_holder.Date==final_dataframe_date].bin_index.values[0]) # find the nearest date and record the bin index
                final_bin_list.append(tuple(location_tuple))
                if any(len(lst) != len(position_type_list) for lst in [start_date_list, start_bin_list, final_bin_list]):
                    print 'position list length ,',len(position_type_list)
                    print 'start date list ',len(start_date_list)
                    print 'start bin list ',len(start_bin_list)
                    raise
        df_dict = {}
        df_dict['Cruise']=cruise_list
        df_dict['position type']=position_type_list
        df_dict['date'] = start_date_list
        df_dict['start bin'] = start_bin_list
        for time_delta,bin_list in zip(time_delta_list,zip(*final_bin_list)):
            bins_string = 'end bin '+str(time_delta)+' day'
            df_dict[bins_string]=bin_list
        df_dict['end bin'] = final_bin_list
        dataframe = pd.DataFrame(df_dict)
        if dump:
            dataframe.to_pickle(self.base_file+'transition_df_degree_bins_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])+'.pickle')
        self.load_df_and_list(dataframe)

class TransMatrix(Transition):
    def __init__(self,save=True,recompile=False,**kwds):
        super(TransMatrix,self).__init__(**kwds)
        self.save=save
        self.transition_matrix_file_path = self.base_file + 'transition_matrix/transition_matrix_degree_bins_'+str(self.degree_bins)\
        +'_time_step_'+str(self.date_span_limit)+'.npz'
        self.number_matrix_file_path = self.base_file + 'transition_matrix/number_matrix_degree_bins_'+str(self.degree_bins)\
        +'_time_step_'+str(self.date_span_limit)+'.npz'
        self.index_list_file_path = self.base_file + 'transition_matrix/index_list_degree_bins_'+str(self.degree_bins)\
        +'_time_step_'+str(self.date_span_limit)+'.npy'
        if recompile:
            self.recompile_transition_and_number_matrix()
        else:
            try:
                print 'I am attempting to load transition matrix data'


                transition_matrix = load_sparse_csc(self.transition_matrix_file_path)
                number_matrix = load_sparse_csc(self.number_matrix_file_path)
                index_list = np.load(self.index_list_file_path)
                self.index_list_remove(index_list)
                self.load_transition_and_number_matrix(transition_matrix,number_matrix)
            except IOError: #this is the case that the file could not load
                print 'i could not load the transition matrix, I am recompiling with degree step size', self.degree_bins,''
                print 'file was '+self.transition_matrix_file_path
                print 'file was '+self.number_matrix_file_path
                print 'file was '+self.index_list_file_path

                self.recompile_transition_and_number_matrix()


    def save_function(self,transition_matrix,number_matrix,index_list):
        save_sparse_csc(self.transition_matrix_file_path,transition_matrix)
        save_sparse_csc(self.number_matrix_file_path,number_matrix)
        np.save(self.index_list_file_path,index_list)


    def load_transition_and_number_matrix(self,transition_matrix,number_matrix):
        assert (np.abs(transition_matrix.sum(axis=0)-1)<10**-10).all()
        assert (transition_matrix>0).data.all() 
        assert (number_matrix>0).data.all()
        assert len(self.list)==transition_matrix.shape[0]
        self.transition_matrix = transition_matrix
        self.number_matrix = number_matrix
        print 'Transition matrix passed all necessary tests. Initial load complete'


    def index_list_remove(self,index_list):
        transition_df_holder = self.df.copy()
        transition_df_holder.loc[self.df[self.end_bin_string].isin([tuple(x) for x in np.array(self.list)[index_list]])\
            ,self.end_bin_string]=np.nan # make all of the end string values nan in these locations
        transition_df_holder.loc[self.df['start bin'].isin([tuple(x) for x in np.array(self.list)[index_list]])\
            ,'start bin']=np.nan # make all of the start bin values nan in these locations   
        self.load_df_and_list(transition_df_holder)

    def column_compute(self,ii,total_list):
#this algorithm has problems with low data density because the frame does not drop na values for the self.end_bin_string
        token_row_list = []
        token_column_list = []
        token_num_list = []
        token_num_test_list = []
        token_data_list = []

        ii_index = total_list.index(list(ii))
        frame = self.df[self.df['start bin']==ii] # data from of floats that start in looped grid cell
        frame_cut = frame[frame[self.end_bin_string]!=ii].dropna(subset=[self.end_bin_string]) #this is all floats that go out of the looped grid cell
        if frame_cut.empty:
            print 'the frame cut was empty'
            return (token_num_list, token_data_list, token_row_list,token_column_list)
        token_row_list = []
        token_column_list = []
        token_num_list = []
        token_num_test_list = []
        token_data_list = []
        token_row_list.append(ii_index) #compute the on diagonal elements
        token_column_list.append(ii_index)  #compute the on diagonal elemnts
        data = (len(frame)-len(frame_cut))/float(len(frame)) 
        # this is the percentage of floats that stay in the looped grid cell
        token_num_list.append(len(frame)-len(frame_cut)) 
        # this is where we save the data density of every cell
        token_data_list.append(data)
        for qq in frame_cut[self.end_bin_string].unique():
            qq_index = total_list.index(list(qq))
            token_row_list.append(qq_index) #these will be the off diagonal elements
            token_column_list.append(ii_index)
            data = (len(frame_cut[frame_cut[self.end_bin_string]==qq]))/float(len(frame))
            token_data_list.append(data)
            token_num_list.append(len(frame_cut[frame_cut[self.end_bin_string]==qq])) 
            # this is where we save the data density of every cell
        assert abs(sum(token_data_list)-1)<0.01 #ensure that all columns scale to 1
        assert ~np.isnan(token_data_list).any()
        assert (np.array(token_data_list)<=1).all()
        assert (np.array(token_data_list)>=0).all()
        assert sum(token_num_list)==len(frame)
        return (token_num_list, token_data_list, token_row_list,token_column_list)

    def matrix_coordinate_change(self,matrix_,old_coord_list,new_coord_list):
        row_list = matrix_.tocoo().row.tolist()
        column_list = matrix_.tocoo().col.tolist()
        data_list = matrix_.tocoo().data.tolist()
        new_row_col_list = []
        new_data_list = []
        for col,row,data in zip(column_list,row_list,data_list):
            try: 
                token = (new_coord_list.index(old_coord_list[row]),new_coord_list.index(old_coord_list[col]))
                new_row_col_list.append(token)
                new_data_list.append(data)
                assert len(new_data_list)==len(new_row_col_list)
            except ValueError:
                continue
        new_row_list,new_col_list = zip(*new_row_col_list)
        return scipy.sparse.csc_matrix((new_data_list,(new_row_list,new_col_list)),shape=(len(new_coord_list),len(new_coord_list)))

    def matrix_recalc(self,index_list,old_total_list,transition_matrix,number_matrix,noise=True):
        row_list = transition_matrix.tocoo().row.tolist()
        column_list = transition_matrix.tocoo().col.tolist()
        data_list = transition_matrix.tocoo().data.tolist()

        full_matrix_list = range(transition_matrix.shape[0])
        not_index_list = np.array(full_matrix_list)[np.isin(full_matrix_list,[old_total_list.index(x) for x in self.list])] # all of the matrix that is not in the new transition list

        index_mask = np.isin(column_list,not_index_list) # this eliminates all columns that have bad data
        print 'the column list is '+str(len(column_list))+' long and we are rejecting '+str((len(index_mask)-sum(index_mask)))

        dummy, column_redo = np.where(transition_matrix[index_list,:].todense()!=0) #find all of the row locations where the transition matrix is non zero
        column_redo = np.unique(column_redo)
        # column_redo = column_redo[~np.isin(column_redo,index_list)] # no need to redo columns that we are removing from the matrix 
        column_redo_mask = ~np.isin(column_list,column_redo)    # delete all parts of the matrix that need to be redone
        print 'the column list is '+str(len(column_list))+' long and we are recalculating '+str((len(column_redo_mask)-sum(column_redo_mask)))

        mask = index_mask&column_redo_mask
        row_list = np.array(row_list)[mask].tolist()
        column_list = np.array(column_list)[mask].tolist()
        data_list = np.array(data_list)[mask].tolist()
        k = len(column_redo)
        for n,index in enumerate(column_redo):
            if n%10==0:
                print 'made it through ',n,' bins. ',(k-n),' remaining'
            dummy_num_list, dummy_data_list,dummy_row_list,dummy_column_list = self.column_compute(tuple(old_total_list[index]),total_list=old_total_list)
            data_list += dummy_data_list
            row_list += dummy_row_list
            column_list += dummy_column_list

            assert len(column_list)==len(row_list)
            assert len(column_list)==len(data_list)

        transition_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(old_total_list),len(old_total_list)))
        transition_matrix = self.matrix_coordinate_change(transition_matrix,old_total_list,self.list)
        number_matrix = self.matrix_coordinate_change(number_matrix,old_total_list,self.list)
        if noise:
            transition_matrix += self.add_noise()
        transition_matrix = self.rescale_matrix(transition_matrix)
        return (transition_matrix,number_matrix)


    def recompile_transition_and_number_matrix(self,plot=False):
        """
        Recompiles transition matrix from __transition_df based on set timestep
        """
        org_number_list = []
        org_data_list = []
        org_row_list = []
        org_column_list = []

        k = len(self.list)   # sets up the total number of loops
        for n,index in enumerate(self.list): #loop through all values of total_list
            print 'made it through ',n,' bins. ',(k-n),' remaining' 
            dummy_num_list, dummy_data_list,dummy_row_list,dummy_column_list = self.column_compute(tuple(index),self.list)
            org_number_list += dummy_num_list
            org_data_list += dummy_data_list
            org_row_list += dummy_row_list
            org_column_list += dummy_column_list
            assert len(org_column_list)==len(org_row_list)
            assert len(org_column_list)==len(org_data_list)
            assert len(org_column_list)==len(org_number_list)
        assert sum(org_number_list)==len(self.df[self.df['start bin'].isin([tuple(self.list[x]) for x in np.unique(org_column_list)])])

        transition_matrix = scipy.sparse.csc_matrix((org_data_list,(org_row_list,org_column_list)),shape=(len(self.list),len(self.list)))
        number_matrix = scipy.sparse.csc_matrix((org_number_list,(org_row_list,org_column_list)),shape=(len(self.list),len(self.list)))

        # self.transition_matrix = self.add_noise(self.transition_matrix)

        eig_vals,eig_vecs = scipy.linalg.eig(transition_matrix.todense())
        orig_len = len(transition_matrix.todense())
        idx = np.where((eig_vecs>0).sum(axis=0)<=2) #find the indexes of all eigenvectors with less than 2 entries
        index_list = []
        for index in idx[0]:
            eig_vec = eig_vecs[:,index]     
            index_list+=np.where(eig_vec>0)[0].tolist() # identify the specific grid boxes where the eigen values are 1 
        dummy,column_index = np.where(np.abs(transition_matrix.sum(axis=0)-1)>10**-10)

        index_list = np.unique(index_list+column_index.tolist()) #these may be duplicated so take unique values

        if index_list.tolist(): #if there is anything in the index list 
            old_total_list = copy.deepcopy(self.list)
            self.index_list_remove(index_list)

            assert ~np.isin(index_list,[old_total_list.index(list(x)) for x in self.df['start bin'].unique()]).any()
            assert ~np.isin(index_list,[old_total_list.index(list(x)) for x in self.df[self.end_bin_string].unique()]).any()
            transition_matrix,number_matrix = self.matrix_recalc(index_list,old_total_list,transition_matrix,number_matrix)
        assert transition_matrix.shape==number_matrix.shape
        self.load_transition_and_number_matrix(transition_matrix,number_matrix)
        if self.save:
            self.save_function(transition_matrix,number_matrix,index_list)

    def get_direction_matrix(self):
        """
        This creates a matrix that shows the number of grid cells north and south, east and west.
        """
        lat_list, lon_list = zip(*self.list)
        pos_max = 180/self.degree_bins[1] #this is the maximum number of bins possible
        output_list = []
        for n,position_list in enumerate([lat_list,lon_list]):
            token_array = np.zeros([len(position_list),len(position_list)])
            for token in np.unique(position_list):
                index_list = np.where(np.array(position_list)==token)[0]
                token_list = (np.array(position_list)-token)/self.degree_bins[n] #the relative number of degree bins
                token_list[token_list>pos_max]=token_list[token_list>pos_max]-2*pos_max #the equivalent of saying -360 degrees
                token_list[token_list<-pos_max]=token_list[token_list<-pos_max]+2*pos_max #the equivalent of saying +360 degrees
                token_array[:,index_list]=np.array([token_list.tolist(),]*len(index_list)).transpose() 
            output_list.append(token_array)
        north_south,east_west = output_list
        self.east_west = east_west
        self.north_south = north_south  #this is because np.array makes lists of lists go the wrong way
        assert (self.east_west<=180/self.degree_bins[1]).all()
        assert (self.east_west>=-180/self.degree_bins[1]).all()
        assert (self.north_south>=-180/self.degree_bins[0]).all()
        assert (self.north_south<=180/self.degree_bins[0]).all()

    def add_noise(self,noise=0.05):
        """
        Adds guassian noise to the transition matrix
        The appropriate level of noise has not been worked out and is kind of ad hock
        """
        print 'adding matrix noise'
        self.get_direction_matrix()
        direction_mat = (-abs(self.east_west)**2-abs(self.north_south)**2)
        noise_mat = abs(np.random.normal(0,noise,direction_mat.shape[0]*direction_mat.shape[1])).reshape(self.east_west.shape)
        direction_mat = noise*np.exp(direction_mat)
        direction_mat[direction_mat<noise/200]=0
        return scipy.sparse.csc_matrix(direction_mat)

    def rescale_matrix(self,matrix_):
        print 'rescaling the matrix'
        mat_sum = matrix_.todense().sum(axis=0)
        scalefactor,dummy = np.meshgrid(1/mat_sum,1/mat_sum)
        matrix_ = scipy.sparse.csc_matrix(np.array(matrix_.todense())*np.array(scalefactor)) #must make these arrays so that the multiplication is element wise
        assert (np.abs(matrix_.sum(axis=0)-1)<10**-2).all()
        return matrix_

    # def delete_w(self):
    #   """
    #   deletes all w matrices in data directory
    #   """
    #   for w in glob.glob(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_*.npz'):
    #       os.remove(w)

    # def load_w(self,number,dump=True):
    #   """
    #   recursively multiplies the transition matrix by itself 
    #   """
    #   print 'in w matrix, current number is ',number
    #   try:
    #       self.w = load_sparse_csc(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz')
    #       assert self.transition_matrix.shape==self.w.shape
 #          assert ~np.isnan(self.w).any()
    #       assert (np.array(self.w)<=1).all()
    #       assert (np.array(self.w)>=0).all()
    #       assert (self.w.todense().sum(axis=0)-1<0.01).all()
    #       print 'w matrix successfully loaded'
    #   except IOError:
    #       print 'w matrix could not be loaded and will be recompiled'
    #       if number == 0:
    #           self.w = self.transition_matrix 
    #       else:
    #           self.load_w(number-1,dump=True)     # recursion to get to the first saved transition matrix 
    #           self.w = self.w.dot(self.transition_matrix)     # advance the transition matrix once
    #       assert self.transition_matrix.shape==self.w.shape
 #          assert ~np.isnan(self.w).any()
    #       assert (np.array(self.w)<=1).all()
    #       assert (np.array(self.w)>=0).all()
    #       assert (self.w.todense().sum(axis=0)-1<0.01).all()
    #       if dump:
    #           save_sparse_csc(self.base_file+'/w_matrix_data/w_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'_number_'+str(number)+'.npz',self.w)   #and save