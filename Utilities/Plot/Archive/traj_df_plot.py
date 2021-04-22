import matplotlib.pyplot as plt
from utils import basemap_setup
from ../datasave/traj_df_save import AllTraj

class TrajPlot(AllTraj):
	pass

	def plot_mean_speed():
		plt.figure(figsize=(10,10))
		XX,YY,m = basemap_setup(self.lat_grid,self.lon_grid)
		m.pcolor(XX,YY,np.log(np.ma.masked_equal(self.speed_mean_matrix,0)))
		plt.title('Log Raw Mean Speed')
		plt.colorbar(label='log(km/hr)')
		plt.savefig('../plots/trajectory_statistics/map_of_raw_mean_speed.png')
		plt.close()

	def plot_speed_variance():
		plt.figure(figsize=(10,10))
		m.pcolor(X,Y,np.ma.masked_equal(speed_variance_matrix,0))
		plt.title('Raw Speed Variance')
		plt.colorbar(label='$km^2/hr^2$')
		plt.savefig('../plots/trajectory_statistics/map_of_raw_speed_variance.png')
		plt.close()

	def float_type_rejected_breakdown():
		fig1, ax1 = plt.subplots()
		argos_number = df_rejected.drop_duplicates(subset=['Cruise']).groupby('position type').count()['Cruise']['ARGOS']
		iridium_number = df_rejected.drop_duplicates(subset=['Cruise']).groupby('position type').count()['Cruise']['GPS']
		argos_number_low = df_rejected[df_rejected.percentage<percent_reject].drop_duplicates(subset=['Cruise']).groupby('position type').count()['Cruise']['ARGOS']
		iridium_number_low = df_rejected[df_rejected.percentage<percent_reject].drop_duplicates(subset=['Cruise']).groupby('position type').count()['Cruise']['GPS']

		ind = (1,2)
		ax1.bar(ind,(argos_number,iridium_number),label='All Floats')
		ax1.bar(ind,(argos_number_low,iridium_number_low),color='r',alpha=.5,label='5% Rejected')
		ax1.set_ylabel('Number')
		ax1.set_title('Number of rejected floats by positioning type')
		ax1.set_xticks(ind)
		ax1.set_xticklabels(('ARGOS', 'GPS'))
		plt.legend()
		plt.savefig('../plots/trajectory_statistics/number_of_floats_by_positioning_type.png')
		plt.close()

	def total_rejected_breakdown():
		fig2, ax2 = plt.subplots()
		df_rejected.drop_duplicates(subset=['Cruise']).percentage.hist(bins=50,label='All floats')
		df_rejected[df_rejected.percentage<percent_reject].drop_duplicates(subset=['Cruise']).percentage.hist(bins=5,color='r',alpha=0.5,label='5% Rejected')
		plt.xlabel('Percentage of bad displacements')
		plt.ylabel('Number of floats')
		plt.title('Histogram of percentage of bad dispacements by float')
		plt.legend()
		plt.savefig('../plots/trajectory_statistics/percentage_of_bad_displacements.png')
		plt.close()

	def velocity_rejected()
		fig, ax = plt.subplots()
		df_rejected.Speed.hist(ax=ax, bins=100, bottom=0.1,label='All Floats')
		df_rejected[df_rejected.percentage<percent_reject].Speed.hist(ax=ax, color='r', bins=100, bottom=0.1,alpha=.5,label='5% Rejected')
		ax.set_yscale('log')
		ax.set_xscale('log')
		plt.legend()
		plt.xlabel('km/hr')
		plt.ylabel('number of displacements')
		plt.title('Histogram of rejected velocities')
		plt.savefig('../plots/trajectory_statistics/histogram_of_rejected_velocities.png')
		plt.close()

	def displacements_per_grid
		self.redemption()
		fig, ax = plt.subplots()
		for number in [(1,1),(2,2),(2,3),(3,3),(4,4)]:
			self.lat_grid = np.arange(-90,90.1,number[0]).tolist()
			self.lon_grid = np.arange(-180,180.1,number[1]).tolist()
			df['bins_lat'] = pd.cut(df.Lat,bins = self.lat_grid,labels=self.lat_grid[:-1])
			df['bins_lon'] = pd.cut(df.Lon,bins = self.lon_grid,labels=self.lon_grid[:-1])
			assert (~df['bins_lat'].isnull()).all()
			assert (~df['bins_lon'].isnull()).all()
			df['bin_index'] = zip(df['bins_lat'].values,df['bins_lon'].values)
			df.groupby('bin_index').count()['Cruise'].hist(label='lat '+str(number[0])+', lon '+str(number[1])+' degree bins',alpha=0.5,bins=500)
		plt.xlim([0,2000])
		ax.set_yscale('log')
		plt.ylabel('Number of bins')
		plt.xlabel('Number of displacements')
		plt.legend()
		plt.savefig('../plots/trajectory_statistics/number_displacements_degree_bin.png')
		plt.close()
