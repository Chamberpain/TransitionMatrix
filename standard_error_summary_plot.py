import numpy as np
import sys,os
import matplotlib.pyplot as plt
import scipy.sparse


base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/'
time_list = [20,40,60,80,100,120,140,160,180]


try:
    datalist = np.load(base_file+'transition_matrix/data_summary_list')
except IOError:
    total_cell_num_list = []
    error_std_list = []
    error_mean_list = []
    displacements_per_cell_std_list = []
    displacements_per_cell_mean_list = []
    degree_bin_list = [(2,3),(3,6),(1,1),(2,2),(3,3),(4,4)]
    for degree_bin in degree_bin_list:
            print degree_bin 
            lat_bin,lon_bin = degree_bin
            total_cell_num_list_holder = []
            error_mean_list_holder = []
            error_std_list_holder = []
            displacements_per_cell_std_list_holder = []
            displacements_per_cell_mean_list_holder = []
            for date_span_limit in time_list:
                    print date_span_limit
                    filename = base_file+'transition_matrix/standard_error_matrix_degree_bins_'+str(degree_bin)+'_time_step_'+str(date_span_limit)+'.npz'
                    loader = np.load(filename)
                    data = loader['data']
                    data[np.isinf(data)]=np.nan
                    matrix = scipy.sparse.csr_matrix((data, loader['indices'], loader['indptr']),
                          shape=loader['shape']).T
                    mean = np.nanmean(data)
                    error_mean_list_holder.append(mean)
                    std = np.nanstd(data)
                    error_std_list_holder.append(std)
                    num = len(np.diag(matrix.todense())[~np.isnan(np.diag(matrix.todense()))])
                    total_cell_num_list_holder.append(num)

                    filename = base_file+'transition_matrix/number_matrix_degree_bins_'+str(degree_bin)+'_time_step_'+str(date_span_limit)+'.npz'
                    loader = np.load(filename)
                    data = loader['data']
                    mean = np.mean(data)
                    displacements_per_cell_mean_list_holder.append(mean)
                    displacements_per_cell_std_list_holder.append(np.std(data))

            total_cell_num_list.append(total_cell_num_list_holder)
            error_std_list.append(error_std_list_holder)
            error_mean_list.append(error_mean_list_holder)
            displacements_per_cell_std_list.append(displacements_per_cell_std_list_holder)
            displacements_per_cell_mean_list.append(displacements_per_cell_mean_list_holder)
    datalist = zip(total_cell_num_list,error_std_list,error_mean_list,displacements_per_cell_std_list,displacements_per_cell_mean_list,degree_bin_list)
    np.save(base_file+'transition_matrix/data_summary_list',datalist)

for token in datalist:
        total_cell_num_list_holder,error_std_list_holder,error_mean_list_holder,displacements_per_cell_std_list_holder,displacements_per_cell_mean_list_holder,name = token
        plt.figure('number plot')
        plt.plot(time_list,total_cell_num_list_holder,'o-',label=str(name))
        plt.figure('standard error')
        plt.errorbar(time_list,error_mean_list_holder,yerr=np.array(error_std_list_holder),fmt='o-',alpha = 0.4,label=str(name))
        plt.figure('displacement number plot')
        plt.errorbar(time_list,displacements_per_cell_mean_list_holder,yerr=np.array(displacements_per_cell_std_list_holder),fmt='o-',alpha = 0.4,label=str(name))


plt.figure('number plot')
plt.title('Number of non-zero grid cells for each grid resolution and datespan')
plt.xlabel('Time Step (days)')
plt.legend()
plt.ylabel('Number of total grid cells')
plt.yscale('Log')
plt.figure('standard error')
plt.title('Mean and standard deviation of standard error \n for each grid resolution and datespan')
plt.legend()
plt.ylabel('Mean standard error')
plt.xlabel('Time Step (days)')
plt.figure('displacement number plot')
plt.title('Mean and standard deviation of displacements per \n cell for each grid resolution and datespan')
plt.legend()
plt.ylabel('Displacements per cell')
plt.xlabel('Time Step (days)')
plt.show()
