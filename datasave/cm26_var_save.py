import numpy as np
from netCDF4 import Dataset
import fnmatch
import sys,os

nc_fid = Dataset('/data/SO12/CM2p6/ocean_scalar.static.nc')
mask = nc_fid['ht'][:]<2000
data_directory_list = [('Surface','/data/SO12/CM2p6/ocean_minibling_surf_flux/'),('100m','/data/SO12/CM2p6/ocean_minibling_100m/')]
pco2_list = []
o2_list = []
matches = []

    for root, dirnames, filenames in os.walk(data_directory):
        for filename in fnmatch.filter(filenames, '*.nc'):
            matches.append(os.path.join(root, filename))
    for n, match in enumerate(matches):
        print 'file is ',match,', there are ',len(matches[:])-n,'files left'
        nc_fid = Dataset(match, 'r')
        file_name = match.split('/')[-1].split('.')[0]

        pco2_variance = np.var(nc_fid['pco2'][:],axis=0)
        pco2_variance.dump('pco2_'+file_name)
        pco2_list.append(pco2_variance)

        o2_variance = np.var(nc_fid['o2_saturation'][:],axis=0)
        o2_variance.dump('o2_'+file_name)
        o2_list.append(o2_variance)

    mean_pco2 = np.ma.mean(np.ma.dstack(pco2_list),axis=2)
    mean_pco2 = np.ma.array(mean_pco2,mask=mask)
    mean_pco2.dump('mean_pco2.dat')
    mean_o2 = np.ma.mean(np.ma.dstack(o2_list),axis=2)
    mean_o2 = np.ma.array(mean_o2,mask=mask)
    mean_o2.dump('mean_o2.dat')