from transition_matrix_plot import TransitionPlot

class InversionPlot(TransitionPlot)
    def __init__(self,**kwds):
        super(InversionPlot,self).__init__(**kwds)
        self.lat_list = np.load('lat_list.npy')
        matrix = self.matrix.transition_matrix
        for _ in range(time_multiplyer-1):
            matrix = matrix.dot(self.matrix.transition_matrix)
        self.matrix.transition_matrix = matrix

        lon_list = np.load('lon_list.npy')
        lon_list[lon_list<-180]=lon_list[lon_list<-180]+360
        self.lon_list = lon_list

        self.translation_list = []
        for x in self.transition.list:
            mask = (self.lat_list==x[0])&(self.lon_list==x[1])      
            if not mask.any():
                for lat in np.arange(x[0]-self.info.degree_bins[0],x[0]+self.info.degree_bins[0],0.5):
                    for lon in np.arange(x[1]-self.info.degree_bins[1],x[1]+self.info.degree_bins[1],0.5):
                        mask = (self.lat_list==lat)&(self.lon_list==lon)
                        if mask.any():
                            t = np.where(mask)
                            break
                    if mask.any():
                        break
            else:
                t = np.where(mask)
            assert t[0]
            assert len(t[0])==1
            self.translation_list.append(t[0][0])


        self.corr_translation_list = []
        mask = (self.lon_list%1==0)&(self.lat_list%1==0)
        lats = self.lat_list[mask]
        lons = self.lon_list[mask]
        for x in self.transition.list:
            mask = (lats==x[0])&(lons==x[1])      
            if not mask.any():
                for lat in np.arange(x[0]-self.info.degree_bins[0],x[0]+self.info.degree_bins[0],0.5):
                    for lon in np.arange(x[1]-self.info.degree_bins[1],x[1]+self.info.degree_bins[1],0.5):
                        mask = (lats==lat)&(lons==lon)
                        if mask.any():
                            t = np.where(mask)
                            break
                    if mask.any():
                        break
            else:
                t = np.where(mask)
            assert t[0]
            assert len(t[0])==1
            self.corr_translation_list.append(t[0][0])

        self.o2 = np.load('subsampled_o2.npy')
        o2_corr = np.load('o2_corr.npy')
        self.o2_corr = self.get_sparse_correlation_matrix(o2_corr)

        self.pco2 = np.load('subsampled_pco2.npy')
        pco2_corr = np.load('pco2_corr.npy')
        self.pco2_corr = self.get_sparse_correlation_matrix(pco2_corr)
        self.soccom_location = self.get_SOCCOM_vector(age_return=False,future_projection=False)

    def get_sparse_correlation_matrix(self,correlation_list):
        total_set = Set([tuple(x) for x in self.transition.list])
        row_list = []
        col_list = []
        data_list = []
        mask = (self.lon_list%1==0)&(self.lat_list%1==0) # we only calculated correlations at whole degrees because of memory
        lats = self.lat_list[mask][self.corr_translation_list] #mask out non whole degrees and only grab at locations that match to the self.transition.index
        lons = self.lon_list[mask][self.corr_translation_list] 
        corr_list = correlation_list[self.corr_translation_list]

        # test_Y,test_X = zip(*self.transition.list)
        for k,(base_lat,base_lon,corr) in enumerate(zip(lats,lons,corr_list)):
            print k
            lat_index_list = np.arange(base_lat-12,base_lat+12.1,0.5)
            lon_index_list = np.arange(base_lon-12,base_lon+12.1,0.5)
            Y,X = np.meshgrid(lat_index_list,lon_index_list) #we construct in this way to match how the correlation matrix was made
            test_set = Set(zip(Y.flatten(),X.flatten()))
            intersection_set = total_set.intersection(test_set)
            location_idx = [self.transition.list.index(list(_)) for _ in intersection_set]
            data_idx = [zip(Y.flatten(),X.flatten()).index(_) for _ in intersection_set]

            data = corr.flatten()[data_idx]
            assert len(location_idx)==len(data)
            row_list += location_idx
            col_list += [k]*len(location_idx)
            data_list += data.tolist()
        assert (np.array(data_list)<=1).all()
        return scipy.sparse.csc_matrix((np.abs(data_list),(row_list,col_list)),shape=self.matrix.transition_matrix.shape)

    def load_corr_array(self,variable,lat,lon):
        cor_folder = './'+str(variable)+'/'
        lon_ = lon
        if lon_>80:
            lon_ -= 360
        if lon_<-260:
            lon_ += 360
        file_ = cor_folder+str(lat)+'_'+str(lon_)+'.npy'
        try:
            array_ = np.load(file_)
        except IOError:
            print 'could not load file, lon is ',lon_,' lat is ',lat
            index_list = [self.total_list.index([lat,lon])]
            data_list = [1]
            return (data_list,index_list)
        lat_list = np.arange(float(lat),float(lat)-12,(-1*self.degree_bins[0]))[::-1].tolist()+np.arange(float(lat),float(lat)+12,self.degree_bins[0])[1:].tolist()
        column_index_list = np.arange(lat-12,lat+12,0.5).tolist()
        lon_list = np.arange(float(lon),float(lon)-12,(-1*self.degree_bins[1]))[::-1].tolist()+np.arange(float(lon),float(lon)+12,self.degree_bins[1])[1:].tolist()
        row_index_list = np.arange(lon-12,lon+12,0.5).tolist()
        data_list = []
        index_list = []
        Y,X = np.meshgrid(lat_list,lon_list)
        for token in zip(Y.flatten(),X.flatten()):
            token = list(token)
            # try:
            row = row_index_list.index(token[1])
            column = column_index_list.index(token[0])
            if token[1]>180:
                token[1]-=360
            if token[1]<-180:
                token[1]+=360
            try:
                index_list.append(self.total_list.index(token))
                data_list.append(array_[row,column])
            except ValueError:
                print 'there was a mistake in the index_list'
                print token
                continue
        assert len(data_list)>0
        assert len(index_list)>0
        return (data_list,index_list)

    def load_corr_matrix(self,variable):
        try:
            self.cor_matrix = load_sparse_csc(self.info.base_file+variable+'_cor_matrix_degree_bins_'+str(self.info.degree_bins)+'.npz')
        except IOError:
            np.load(variable+'_corr.npy')
            data_list = []
            row_list = []
            column_list = []
            for n,token in enumerate(self.total_list):
                print n
                base_lat,base_lon = token
                data_list_token,row_list_token = self.load_corr_array(variable,base_lat,base_lon)
                print row_list_token
                row_list += row_list_token
                column_list += [n]*len(row_list_token)
                data_list += data_list_token            
            self.cor_matrix = scipy.sparse.csc_matrix((data_list,(row_list,column_list)),shape=(len(self.transition.list),len(self.transition.list)))
            save_sparse_csc(self.info.base_file+variable+'_cor_matrix_degree_bins_'+str(self.info.degree_bins)+'.npz', self.cor_matrix)


    def get_optimal_float_locations(self,field_vector,correlation_matrix,float_number=1000):
        """ accepts a target vecotr in sparse matrix form, returns the ideal float locations and weights to sample that"""

        # self.w[self.w<0.007]=0
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