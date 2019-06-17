from transition_matrix_compute import TransMatrix

class TransInversion(TransMatrix)











    def __init__(self,traj_class,time_multiplyer=1):
        self = traj_class
        matrix = self.transition_matrix
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










    def landschutzer_co2_flux(self,var=False,plot=False):
        file_ = self.base_file+'spco2_MPI_SOM-FFN_v2018.nc'
        nc_fid = Dataset(file_)
        y = nc_fid['lat'][:]
        x = nc_fid['lon'][:]
        
        data = np.ma.masked_greater(nc_fid['fgco2_smoothed'][:],10**19) 
        if var:
            np.nanvar()
        else:
            XX,YY = np.gradient(np.nanmean(data,axis=0)) # take the time mean, then take the gradient of the 2d array
            data = np.abs(XX+YY)
        x[0] = -180
        x[-1] = 180
        co2_vector = np.zeros([len(self.total_list),1])
        for n,(lat,lon) in enumerate(self.total_list):
            lon_index = find_nearest(lon,x)
            lat_index = find_nearest(lat,y)
            co2_vector[n] = data[lat_index,lon_index]
        if plot:
            co2_plot = abs(self.transition_vector_to_plottable(co2_vector))
            plt.figure()
            m = Basemap(projection='cyl',fix_aspect=False)
            # m.fillcontinents(color='coral',lake_color='aqua')
            m.drawcoastlines()
            XX,YY = m(self.X,self.Y)
            m.pcolormesh(XX,YY,co2_plot,cmap=plt.cm.PRGn)
            plt.colorbar(label='CO2 Flux $gm C/m^2/yr$')
        return co2_vector


    def landschutzer_loc_plot(self,var=False):
        field_vector = landschutzer_co2_flux(var)
        field_plot = abs(self.matrix.transition_vector_to_plottable(field_vector))
        x,y,desired_vector,target_vector =  self.get_optimal_float_locations(field_vector,self.pco2_corr,desired_vector_in_plottable=(not cost_plot)) #if cost_plot, the vector will return desired vector in form for calculations
        m,XX,YY = traj_class.matrix.matrix_plot_setup() 
        m.fillcontinents(color='coral',lake_color='aqua')
        m.pcolormesh(XX,YY,np.ma.masked_equal(field_plot,0),cmap=plt.cm.Purples,norm=colors.LogNorm())
        plt.title('PCO2 Variance')
        plt.colorbar()
        m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Purples) # this is a plot for the tendancy of the residence time at a grid cell
        m.scatter(x,y,marker='*',color='g',s=34)
        m = self.plot_latest_soccom_locations(m)
        plt.show()


    def pco2_mag_plot(self):
        df = pd.read_csv('../eulerian_plot/basemap/data/sumflux_2006c.txt', skiprows = 51,sep=r"\s*")
        y = np.sort(df.LAT.unique())
        x = np.sort(df.LON.unique())
        XC,YC = np.meshgrid(x,y)
        CO2 = np.zeros([len(y),len(x)])
        di = df.iterrows()
        for i in range(len(df)):
            row = next(di)[1]
            CO2[(row['LON']==XC)&(row['LAT']==YC)] = row['DELTA_PCO2']
        CO2, x = shiftgrid(180, CO2, x, start=False)
        x = x+5
        x[0] = -180
        x[-1] = 180
        co2_vector = np.zeros([len(self.total_list),1])
        for n,(lat,lon) in enumerate(self.total_list):
            lon_index = find_nearest(lon,x)
            lat_index = find_nearest(lat,y)
            co2_vector[n] = CO2[lat_index,lon_index]
        co2_plot = abs(self.transition_vector_to_plottable(co2_vector))
        plt.figure()
        m = Basemap(projection='cyl',fix_aspect=False)
        # m.fillcontinents(color='coral',lake_color='aqua')
        m.drawcoastlines()
        XX,YY = m(self.X,self.Y)
        m.pcolormesh(XX,YY,co2_plot,cmap=plt.cm.PRGn) # this is a plot for the tendancy of the residence time at a grid cell
        plt.colorbar(label='CO2 Flux $gm C/m^2/yr$')

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

    def pco2_var_plot(self,cost_plot=False):
        field_vector = self.pco2.var(axis=0)[self.translation_list]
        field_plot = abs(self.matrix.transition_vector_to_plottable(field_vector))
        x,y,desired_vector,target_vector =  self.get_optimal_float_locations(field_vector,self.pco2_corr,desired_vector_in_plottable=(not cost_plot)) #if cost_plot, the vector will return desired vector in form for calculations
        m,XX,YY = traj_class.matrix.matrix_plot_setup() 
        m.fillcontinents(color='coral',lake_color='aqua')
        m.pcolormesh(XX,YY,np.ma.masked_equal(field_plot,0),cmap=plt.cm.Purples,norm=colors.LogNorm())
        plt.title('PCO2 Variance')
        plt.colorbar()
        m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Purples) # this is a plot for the tendancy of the residence time at a grid cell
        m.scatter(x,y,marker='*',color='g',s=34)
        m = self.plot_latest_soccom_locations(m)
        
        if cost_plot:
            float_number = []
            variance_misfit = []
            total_var = np.sum(target_vector.dot(target_vector.T))
            for limit in np.arange(1 ,0,-0.01):
                print 'limit is ',limit
                
                token_vector = np.zeros(len(desired_vector))
                token_vector[desired_vector>limit] = [math.ceil(k) for k in desired_vector[desired_vector>limit]]
                float_number.append(token_vector.sum())
                token_result = target_vector - self.w.todense().dot(np.matrix(token_vector).T)
                token_result[token_result<0] = 0
                variance_misfit.append(100-np.sum(token_result.dot(token_result.T))/total_var*100)
                print 'variance misfit is ',variance_misfit[-1]
            plt.subplot(2,1,2)
            plt.plot(float_number,variance_misfit)
            plt.ylabel('pCO$^2$ Variance Constrained (%)')
            plt.xlabel('Number of Additional Floats Deployed')
        plt.show()

    def o2_var_plot(self,line=([20, -55], [-30, -15])):
        plt.figure()
        if line:
            plt.subplot(2,1,1)

        field_vector = self.o2.var(axis=0)[self.translation_list]
        field_plot = abs(self.matrix.transition_vector_to_plottable(field_vector))
        x,y,desired_vector =  self.get_optimal_float_locations(field_vector,self.o2_corr,level=0.45) #if cost_plot, the vector will return desired vector in form for calculations
        m,XX,YY = traj_class.matrix.matrix_plot_setup() 
        m.fillcontinents(color='coral',lake_color='aqua')
        m.pcolormesh(XX,YY,np.ma.masked_equal(field_plot,0),cmap=plt.cm.Reds,norm=colors.LogNorm())
        plt.title('O2 Variance')
        plt.colorbar()
        m.scatter(x,y,marker='*',color='g',s=30,latlon=True)
        m = self.plot_latest_soccom_locations(m)
        if line:
            lat,lon = line
            x,y = m(lon,lat)
            m.plot(x,y,'o-')
            plt.subplot(2,1,2)
            lat = np.linspace(lat[0],lat[1],50)
            lon = np.linspace(lon[0],lon[1],50)
            points = np.array(zip(lat,lon))
            grid_z0 = griddata((self.X.flatten(),self.Y.flatten()),desired_vector.flatten(),points,method='linear')
            plt.plot((lat-lat.min())*111,grid_z0)
            plt.xlabel('Distance Along Cruise Track (km)')
            plt.ylabel('Variance Constrained')
        plt.show()

    def hybrid_var_plot(self):
        plt.figure()
        field_vector_pco2 = self.cm2p6(self.base_file+'mean_pco2.dat')
        field_vector_o2 = self.cm2p6(self.base_file+'mean_o2.dat')
        field_vector_pco2[field_vector_pco2>(10*field_vector_pco2.std()+field_vector_pco2.mean())]=0
        field_vector = field_vector_pco2/(2*field_vector_pco2.max())+field_vector_o2/(2*field_vector_o2.max())
        field_plot = abs(self.transition_vector_to_plottable(field_vector))

        m,x,y,desired_vector = self.get_optimal_float_locations(field_vector)
        XX,YY = m(self.X,self.Y)
        m.pcolormesh(XX,YY,np.log(field_plot),cmap=plt.cm.Greens) # this is a plot for the tendancy of the residence time at a grid cell
        m.scatter(x,y,marker='*',color='r',s=30)        
        plt.show()

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

    def plot_cm26_cor_ellipse(self,matrix):
        """plots the correlation ellipse of the cm2.6 variable"""
        self.matrix.get_direction_matrix()
        self.matrix.quiver_plot(matrix,arrows=False,scale_factor = 0.1)
        plt.show()