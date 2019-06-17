from transition_matrix_compute import TransMatrix

class TransMatrixWithholding(TransMatrix):
	pass

    def trajectory_data_withholding_setup(self,percentage):
        # number_matrix = load_sparse_csc(self.base_file+'transition_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
        # standard_error_matrix = load_sparse_csc(self.base_file+'transition_matrix/standard_error_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
        old_coord_list = copy.deepcopy(self.list)
        assert_len = len(old_coord_list)
        mylist = self.transition.df.Cruise.unique().tolist() # total list of floats
        remaining_float_list = [mylist[i] for i in sorted(random.sample(xrange(len(mylist)), int(round(len(mylist)*(1-percentage)))))] #randomly select floats to exclude
        subtracted_float_list = np.array(mylist)[~np.isin(mylist,remaining_float_list)]

        df_holder = self.transition.df[self.transition.df.Cruise.isin(subtracted_float_list)]
        
        index_list = []
        for x in (df_holder[self.end_bin_string].unique().tolist()+df_holder['start bin'].unique().tolist()):
            try:
                index_list.append(old_coord_list.index(list(x)))
            except ValueError:
                print 'there was a value error during withholding'
                continue
        index_list = np.unique(index_list)

        new_df = self.transition.df[self.transition.df.Cruise.isin(remaining_float_list)]        
        self.transition.load_df_and_list(new_df)
        assert assert_len == len(old_coord_list)
        transition_matrix = self.matrix_recalc(index_list,old_coord_list,self.transition_matrix)
        self.load_transition_and_number_matrix(transition_matrix,self.number_matrix)

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
                    max_lat = abs(lat_outer-lat_inner)
                    max_lon = abs(lon_outer-lon_inner)              
                    print('they are divisable')

                    outer_class = transition_class_loader(lat_outer,lon_outer,date_span_limit,traj_file_type)
                    inner_class = transition_class_loader(lat_inner,lon_inner,date_span_limit,traj_file_type)
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
                    inner_coord_lat, inner_coord_lon = zip(*[inner_class.trans.list[x] for x in inner_coord_list])
                    outer_coord_lat, outer_coord_lon = zip(*[outer_class.trans.list[x] for x in outer_coord_list])
                    assert (abs(np.array(inner_coord_lat)-np.array(outer_coord_lat))<=max_lat).all()
                    assert (abs(np.array(inner_coord_lon)-np.array(outer_coord_lon))<=max_lon).all()

                    inner_coord_list = np.array(inner_coord_list)
                    outer_coord_list = np.array(outer_coord_list)
                    test_list = [(inner_coord_list[outer_coord_list==x].tolist(),x) for x in np.sort(np.unique(outer_coord_list))]
                    assert (np.array([len(x[0]) for x in test_list])<=max_len).all()

                    new_row_list = []
                    new_col_list = []
                    new_data_list = []
                    for k,(col_list,col_index) in enumerate(test_list):
                        print 'I am working on '+str(k)+' column of outer'
                        col_num = len(col_list)
                        for n,(row_list,row_index) in enumerate(test_list):
                            row_num = len(row_list)
                            col_holder = col_list*row_num
                            row_list = [val for val in row_list for _ in range(col_num)]
                            
                            token = inner_class.matrix.transition_matrix[row_list,col_holder]
                            if token.mean()!=0:
                                new_data_list.append(token.sum())
                                new_row_list.append(row_index)
                                new_col_list.append(col_index)

                            row_dummy = np.where(token!=0)[0].tolist()
                            col_dummy = [k]*len(row_dummy)
                            data_dummy = token[row_dummy].T.tolist()[0]
                            new_row_list += row_dummy
                            new_col_list += col_dummy
                            new_data_list += data_dummy

                    matrix_holder = scipy.sparse.csc_matrix((new_data_list,(new_row_list,new_col_list)),shape=(outer_class.matrix.transition_matrix.shape[0],outer_class.matrix.transition_matrix.shape[1]))
                    matrix_holder = inner_class.matrix.rescale_matrix(matrix_holder)

                    datalist.append((inner,outer,traj_file_type,matrix_difference_compare(matrix_holder,outer_class.matrix.transition_matrix)))
                else:
                    print('they are not divisable')
                    continue
    with open('transition_matrix_resolution_comparison.pickle', 'wb') as fp:
        pickle.dump(datalist, fp)

def data_withholding_calc():
    datalist = []
    date = 60
    coord_list = [(4,4),(2,3),(3,6),(1,1),(2,2),(3,3)]
    for token in coord_list:
        lat,lon = token
        traj_class = transition_class_loader(degree_bin_lat=lat,degree_bin_lon=lon,date_span_limit=date,traj_file_type='SOSE')
        for percentage in np.arange(0.95,0.65,-0.05):
            repeat = 10
            while repeat >0:
                print 'repeat is ',repeat 
                repeat-=1
                test_traj_class = copy.deepcopy(traj_class)
                test_traj_class.matrix.trajectory_data_withholding_setup(percentage)
                token_1,test_traj_class = matrix_size_match(copy.deepcopy(traj_class),test_traj_class)
                datalist.append((token,percentage,matrix_difference_compare(token_1.matrix.transition_matrix,test_traj_class.matrix.transition_matrix)))
    with open('transition_matrix_withholding_data.pickle', 'wb') as fp:
        pickle.dump(datalist, fp)



def matrix_datespace_intercomparison():
    datalist = []
    datelist = range(40,300,20)
    coord_list = [(2,3)]
    for lat,lon in coord_list:
        for date2 in datelist:
            for traj_file_type in ['Argo','SOSE']:
                outer_class = transition_class_loader(lat,lon,date2,traj_file_type)
                inner_class = transition_class_loader(lat,lon,20,traj_file_type)

                outer_class,inner_class = matrix_size_match(outer_class,inner_class)
                matrix_token = copy.deepcopy(inner_class.matrix.transition_matrix)
                for dummy in range(int(date2/20)-1):
                    inner_class.matrix.transition_matrix = inner_class.matrix.transition_matrix.dot(matrix_token)
                    inner_class.matrix.transition_matrix = inner_class.matrix.rescale_matrix(inner_class.matrix.transition_matrix)
                outer_class, inner_class = matrix_size_match(outer_class,inner_class)
                datalist.append((date2,traj_file_type,matrix_difference_compare(inner_class.matrix.transition_matrix,outer_class.matrix.transition_matrix)))
    with open('transition_matrix_datespace_data.pickle', 'wb') as fp:
        pickle.dump(datalist, fp)
