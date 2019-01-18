import numpy as np
import sys,os
import matplotlib.pyplot as plt
import scipy.sparse


base_file = os.getenv("HOME")+'/iCloud/Data/Processed/transition_matrix/'
degree_bin_list = [(2,3),(3,6),(1,1),(2,2),(3,3),(4,4)]
time_list = [20,40,60,80,100,120,140,160,180]
num_list = []
std_list = []
mean_list = []


for degree_bin in degree_bin_list:
        print degree_bin 
        lat_bin,lon_bin = degree_bin
        num_list_holder = []
        mean_list_holder = []
        std_list_holder = []
        for date_span_limit in time_list:
                print date_span_limit
                filename = base_file+'transition_matrix/standard_error_matrix_degree_bins_'+str(degree_bin)+'_time_step_'+str(date_span_limit)+'.npz'
                loader = np.load(filename)
                data = loader['data']

                data[np.isinf(data)]=np.nan

                matrix = scipy.sparse.csr_matrix((data, loader['indices'], loader['indptr']),
                      shape=loader['shape']).T
                mean = np.nanmean(data)
                mean_list_holder.append(mean)
                std = np.nanstd(data)
                std_list_holder.append(std)
                num = len(np.diag(matrix.todense())[~np.isnan(np.diag(matrix.todense()))])
                num_list_holder.append(num)
        num_list.append(num_list_holder)
        std_list.append(std_list_holder)
        mean_list.append(mean_list_holder)

for token in zip(num_list,std_list,mean_list,degree_bin_list):
        num_list_holder,std_list_holder,mean_list_holder,name = token
        plt.figure('number plot')
        plt.plot(num_list_holder,'o-',label=str(name))
        plt.figure('standard error')
        plt.errorbar(time_list,mean_list_holder,yerr=np.array(std_list_holder),fmt='o-',label=str(name))
plt.figure('number plot')
plt.title('Number of non-zero grid cells for each grid resolution and datespan')
plt.xlabel('Time Step (days)')
plt.legend()
plt.ylabel('Number of total grid cells')
plt.figure('standard error')
plt.title('Mean and standard deviation of standard error \n for each grid resolution and datespan')
plt.legend()
plt.ylabel('Mean standard error')
plt.xlabel('Time Step (days)')
plt.show()
