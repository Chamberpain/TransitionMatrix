import matplotlib.pyplot as plt
from utils import basemap_setup
from ../datasave/traj_df_save import AllTraj

class TrajPlot(AllTraj):
	pass

	def plot_mean_speed()
		plt.figure(figsize=(10,10))
		XX,YY,m = basemap_setup(self.lat_grid,self.lon_grid)
		m.pcolor(XX,YY,np.log(np.ma.masked_equal(self.speed_mean_matrix,0)))
		plt.title('Log Raw Mean Speed')
		plt.colorbar(label='log(km/hr)')
		plt.savefig('../plots/map_of_raw_mean_speed.png')
		plt.close()