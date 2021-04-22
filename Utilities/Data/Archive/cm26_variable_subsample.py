import numpy as np
from netCDF4 import Dataset
import fnmatch
import sys,os

base_filepath = '/home/pchamber/sub_sampled_cm2p6/'
# nc_fid = Dataset('/data/SO12/CM2p6/ocean_scalar.static.nc')
# mask = nc_fid['ht'][:]<2000
surf_dir = '/data/SO12/CM2p6/ocean_minibling_surf_flux/'
hundred_m_dir = '/data/SO12/CM2p6/ocean_minibling_100m/'

data_directory_list = [('pco2',surf_dir),('o2',hundred_m_dir),('dic',hundred_m_dir),('po4',hundred_m_dir),('biomass_p',hundred_m_dir)]
for variable,directory in data_directory_list:
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, '*.nc'):
            matches.append(os.path.join(root, filename))
    for n, match in enumerate(matches):
        print 'file is ',match,', there are ',len(matches[:])-n,'files left'
        nc_fid = Dataset(match, 'r')
        file_name = match.split('/')[-1].split('.')[0]
        save_name = base_filepath+variable+file_name
        variable_holder = nc_fid[variable][::10,::10,::10]
        np.save(save_name,variable_holder.data)