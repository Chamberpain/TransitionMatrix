import pickle
import numpy as np
import os
import LatLon
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import griddata


base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/'
save_folder = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/'
for variable in ['o2','pco2']
    with open(base_file+variable+"_array_list", "rb") as fp:   
        array_list = pickle.load(fp)

    with open(self.base_file+variable+"_position_list", "rb") as fp:
        position_list = pickle.load(fp)

    distance_list = []
    dx_list = []
    dy_list = []
    for token in position_list:
        base, alt = token
        distance = (LatLon.LatLon(base[0],base[1])-LatLon.LatLon(alt[0],alt[1]))
        dx_list.append(distance.dx)
        dy_list.append(distance.dy)
        distance_list.append(distance.magnitude)
    sns.jointplot(x=distance_list,y=array_list,kind='hex')
    plt.xlabel('Distance (km)')
    plt.ylabel('Correlation')
    plt.savefig(save_folder+variable+'total_distance_correlation')
    plt.close()

    sns.jointplot(x=dx_list,y=array_list,kind='hex')
    plt.xlabel('X Distance (km)')
    plt.ylabel('Correlation')
    plt.savefig(save_folder+variable+'X_distance_correlation')
    plt.close()

    sns.jointplot(x=dy_list,y=array_list,kind='hex')
    plt.xlabel('Y Distance (km)')
    plt.ylabel('Correlation')
    plt.savefig(save_folder+variable+'Y_distance_correlation')
    plt.close()


    n = 11
    xg = np.linspace(min(dx_list),max(dx_list),n)
    yg = np.linspace(min(dy_list),max(dy_list),n)

    cor_mat = np.zeros([len(xg)-1,len(yg)-1])
    for k in range(len(xg)-1):
        x_mask = (xg[k]<np.array(dx_list))&(xg[k+1]>np.array(dx_list))
        for i in range(len(yg)-1):
            y_mask = (yg[i]<np.array(dy_list))&(yg[i+1]>np.array(dy_list))
            val = array_list[(x_mask)&(y_mask)].mean()
            if np.isnan(val):
                print xg[k]
                print yg[i]
                print array_list[(x_mask)&(y_mask)]
            cor_mat[k,i] = val
    cor_mat = np.ma.masked_where(np.isnan(cor_mat),cor_mat)
    X,Y = np.meshgrid(xg,yg)

    # interpolate Z values on defined grid
    Z = griddata(np.vstack((dx_list,dy_list)).T, \
      np.vstack(array_list.flatten()),(X,Y),method='linear').reshape(X.shape)
    # mask nan values, so they will not appear on plot
    Zm = np.ma.masked_where(np.isnan(Z),Z)

    # plot
    plt.figure()
    plt.pcolormesh(X,Y,cor_mat)
    plt.title('Correlation')
    plt.xlabel('X displacement (km)')
    plt.ylabel('Y displacement (km)')
    plt.colorbar()
    plt.savefig(save_folder+variable+'_correlation_map')
    plt.close()    