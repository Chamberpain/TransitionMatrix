import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys,os
from netCDF4 import Dataset
from mpl_toolkits.basemap import Basemap,shiftgrid

def plot_the_data(data,cm,title,cblabel,file_name,alphanum=1):

	plt.figure()
	m = Basemap(llcrnrlon=0.,llcrnrlat=-80.,urcrnrlon=360.,urcrnrlat=-25,projection='cyl',fix_aspect=False)
	m.drawcoastlines()
	XX,YY = m(X,Y)
	m.pcolormesh(XX,YY,data,cmap=cm,alpha=alphanum)
	plt.title(title,size=30)
	plt.colorbar(label=cblabel)
	plt.savefig(base_file+file_name)	
	plt.close()


base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/sose/energy/'	#this is a total hack to easily change the code to run sose particles
mean_energy = np.load(base_file+'mean_SOSE_energy.npy')/2
mean_eke = np.load(base_file+'mean_SOSE_eke.npy')/2
XG = np.load(base_file+'XG.npy')
YC = np.load(base_file+'YC.npy')
grid = Dataset(base_file+'grid.nc')
X,Y = np.meshgrid(XG[::6],YC[::6])

mean_energy = np.ma.masked_equal(mean_energy,0)
plot_the_data(np.log(mean_energy),plt.cm.gist_heat,'Mean Energy','log(J/kg)','mean_energy')
mean_eke = np.ma.masked_equal(mean_eke,0)
plot_the_data(np.log(mean_eke),plt.cm.copper,'Mean EKE','log(J/kg)','mean_eke')


dx,dy = np.gradient(mean_energy,YC[::6],XG[::6])
plot_the_data(np.log(np.sqrt(dx*dx+dy*dy)),plt.cm.gist_heat,'Gradient Mean Energy','log(J/m*kg)','gradient_mean_energy')
dx,dy = np.gradient(mean_eke,YC[::6],XG[::6])
plot_the_data(np.log(np.sqrt(dx*dx+dy*dy)),plt.cm.copper,'Gradient Mean EKE','log(J/m*kg)','gradient_mean_eke')


depth = grid['Depth'][:]/1000.
depth = np.ma.masked_equal(depth,0)   
X = grid['XC'][:]
Y = grid['YC'][:]
plot_the_data(depth,plt.cm.binary,'Depth','km','depth',alphanum=0.5)