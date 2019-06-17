from inverse_plot import InversionPlot

class CM2p6Plot(InversionPlot)
    pass

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