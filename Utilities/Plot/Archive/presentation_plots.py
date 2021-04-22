from compute_utilities.list_utilities import find_nearest  
from transition_matrix.datasave.decimated_cm26_arrays import transition_vector_to_plottable,basemap_setup
import scipy.sparse
import numpy as np
import matplotlib.pyplot as plt
import random 




lats,lons,truth_list,lat_grid,lon_grid = lats_lons()
lats = [round(x) for x in lats]
lons = [round(x) for x in lons]


variable_list = ['dic','o2','salt','temp']
holder_list = []
for variable in variable_list:
	holder = np.load('subsampled_'+variable+'.npy')
	holder_list.append(holder[:,truth_list])
data_array = np.hstack(holder_list)
cov = np.cov(data_array.T)
break_unit = holder[:,truth_list].shape[1]


def get_cov(row_var,col_var):
	def get_breaks(variable):
		idx = variable_list.index(variable)
		start = idx*break_unit
		end = start + break_unit
		return (start,end)
	row_start,row_end = get_breaks(row_var)
	col_start,col_end = get_breaks(col_var)
	return cov[row_start:row_end,col_start:col_end]


scaling = calculate_scaling()

cov_dict = {}
for col_var in variable_list:
	variable_dict = {}
	for row_var in variable_list:
		cov_holder = abs(get_cov(row_var,col_var))
		variable_dict[row_var] = scipy.sparse.csc_matrix(cov_holder*scaling)
	cov_dict[col_var]=variable_dict

idxs = random.sample(range(len(lats)),3800)


dic_cov = cov_dict['temp']['dic'].sum(axis=1) + cov_dict['salt']['dic'].sum(axis=1)
dic_cov = dic_cov.ravel().tolist()[0]

dic_cor = dic_cov/cov_dict['dic']['dic'].diagonal()
dic_cor[dic_cor>1]=1

o2_cov = cov_dict['temp']['o2'].sum(axis=1) + cov_dict['salt']['o2'].sum(axis=1) 
o2_cov = o2_cov.ravel().tolist()[0]

o2_cor = o2_cov/cov_dict['o2']['o2'].diagonal()
o2_cor[o2_cor>1]=1



index_list = zip(lats,lons)
o2_plottable = transition_vector_to_plottable(lat_grid.tolist(),lon_grid.tolist(),index_list,o2_cov)
dic_plottable = transition_vector_to_plottable(lat_grid.tolist(),lon_grid.tolist(),index_list,dic_cov)

XX,YY,m = basemap_setup(lat_grid,lon_grid)
m.scatter(np.array(lons)[idxs],np.array(lats)[idxs],s=2,marker='*',color='y',latlon=True)
plt.savefig('array')
plt.close()

XX,YY,m = basemap_setup(lat_grid,lon_grid)
m.pcolormesh(XX,YY,o2_plottable)
plt.colorbar(label='Variance $(mol\ m^{-2})^2$')
plt.savefig('o2_cov_map')
plt.close()

XX,YY,m = basemap_setup(lat_grid,lon_grid)
m.pcolormesh(XX,YY,dic_plottable)
plt.colorbar(label='Variance $(mol\ m^{-2})^2$')
plt.savefig('dic_cov_map')
plt.close()

o2_plottable = transition_vector_to_plottable(lat_grid.tolist(),lon_grid.tolist(),index_list,o2_cor)
dic_plottable = transition_vector_to_plottable(lat_grid.tolist(),lon_grid.tolist(),index_list,dic_cor)

XX,YY,m = basemap_setup(lat_grid,lon_grid)
m.pcolormesh(XX,YY,o2_plottable)
plt.colorbar(label='Correlation')
plt.savefig('o2_cor_map')
plt.close()

XX,YY,m = basemap_setup(lat_grid,lon_grid)
m.pcolormesh(XX,YY,dic_plottable)
plt.colorbar(label='Correlation')
plt.savefig('dic_cor_map')
plt.close()
