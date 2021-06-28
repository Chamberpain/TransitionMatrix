import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Compute.__init__ import ROOT_DIR as data_root
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import pickle

data_handler = FilePathHandler(data_root,'transmat_withholding')
plot_handler = FilePathHandler(ROOT_DIR,'transition_matrix_withholding_plot')

def argos_gps_plot():
	with open(data_handler.tmp_file('argos_gps_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	plot_style_dict = {'argos':'--','gps':':'}
	plot_color_dict = {(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}
	base_mat,subsampled_list,l2_mean,l2_std = zip(*datalist)
	lats,lons,time_step,pos_type = zip(*subsampled_list)
	pos_type = ['argos','gps']*(int(len(pos_type)/2))
	tuple_list = list(zip(lats,lons))
	for positioning in np.unique(pos_type):
		for grid in list(set(tuple_list)):
			lat_mask = np.array([grid[0]==x for x in lats])
			lon_mask = np.array([grid[1]==x for x in lons])
			season_mask = np.array([positioning==x for x in pos_type])
			mask = lat_mask&lon_mask&season_mask
			out = np.array(l2_mean)[mask]
			print('I am plotting ',positioning)
			plt.plot(np.unique(time_step),out,linestyle=plot_style_dict[positioning],color=plot_color_dict[grid])
	plt.xlim(30,180)
	plt.xlabel('Timestep')
	plt.ylabel('Mean Difference')

def seasonal_plot():
	with open(data_handler.tmp_file('seasonal_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	plot_style_dict = {'summer':'--','winter':':'}
	plot_color_dict = {(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}
	base_mat,subsampled_list,l2_mean,l2_std = zip(*datalist)
	lats,lons,time_step,pos_type = zip(*subsampled_list)
	pos_type = ['summer','winter']*(int(len(pos_type)/2))
	tuple_list = list(zip(lats,lons))
	for season in np.unique(pos_type):
		for grid in list(set(tuple_list)):
			lat_mask = np.array([grid[0]==x for x in lats])
			lon_mask = np.array([grid[1]==x for x in lons])
			season_mask = np.array([season==x for x in pos_type])
			mask = lat_mask&lon_mask&season_mask
			out = np.array(l2_mean)[mask]
			print('I am plotting ',season)
			plt.plot(np.unique(time_step),out,linestyle=plot_style_dict[season],color=plot_color_dict[grid])
	plt.xlim(30,180)
	plt.xlabel('Timestep')
	plt.ylabel('Mean Difference')

def resolution_difference_plot():
	with open(data_handler.tmp_file('resolution_difference_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	plot_style_dict = {'argo':'--','SOSE':':'}
	plot_color_dict = {(1,1):'teal',(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}
	ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,lat,lon,time,pos_type = zip(*datalist)
	for system in ['argo']:
		for grid in list(plot_color_dict.keys()):
			lat_mask = np.array([grid[0]==x for x in lat])
			lon_mask = np.array([grid[1]==x for x in lon])
			system_mask = np.array([system==x for x in pos_type])
			mask = lat_mask&lon_mask&system_mask
			ew_mean_diff_holder = np.array(ew_mean_diff)[mask]
			ns_mean_diff_holder = np.array(ns_mean_diff)[mask]
			ew_std_diff_holder = np.array(ew_std_diff)[mask]
			ns_std_diff_holder = np.array(ns_std_diff)[mask]

			out = np.sqrt(ew_std_diff_holder**2+ns_std_diff_holder**2)
			print('I am plotting ',system)
			plt.plot(np.unique(time),out,linestyle=plot_style_dict[system],color=plot_color_dict[grid])
	plt.xlim(30,120)
	plt.xlabel('Timestep')
	plt.ylabel('Mean Difference')

def date_difference_plot():
	with open(data_handler.tmp_file('datespace_data'), 'rb') as fp:
		datalist = pickle.load(fp)
	plot_style_dict = {'argo':'--','SOSE':':'}
	plot_color_dict = {(1,1):'teal',(1,2):'purple',(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}
	inner,outer,l2_mean,l2_std = zip(*datalist)
	lats,lons,time_step,pos_type = zip(*inner)
	tuple_list = list(zip(lats,lons))
	for system in np.unique(pos_type):
		for grid in list(set(tuple_list)):
			lat_mask = np.array([grid[0]==x for x in lats])
			lon_mask = np.array([grid[1]==x for x in lons])
			system_mask = np.array([system==x for x in pos_type])
			mask = lat_mask&lon_mask&system_mask
			out = np.array(l2_mean)[mask]
			print('I am plotting ',system)
			plt.plot(np.unique(time_step),out,linestyle=plot_style_dict[system],color=plot_color_dict[grid])
	plt.xlim(30,180)
	plt.xlabel('Timestep')
	plt.ylabel('Mean Difference')


def data_withholding_plot():

	with open('transition_matrix_withholding_data.pickle', 'rb') as fp:
		datalist = pickle.load(fp)
	percentage,actual_data = zip(*datalist)
	token = [(2,3)]*len(percentage)
	token = [str(x) for x in token]
	token = np.array(token)
	percentage = np.array(percentage)
	actual_data = np.array(actual_data)
	for t in np.unique(token):
		plt.figure()
		plt.title('Data Withholding for '+str(t))
		mask = token==t
		percentage_token = percentage[mask]
		actual_data_token = actual_data[mask]
		mean_list = []
		std_list = []
		for p in np.unique(percentage_token):
			mask = percentage_token ==p
			plot_label = str(round((1-p)*100))+'% withheld'
			data = actual_data_token[mask]
			eig_list = []
			mean_list_holder = []
			std_list_holder = []
			for n,d in enumerate(data):
				eigs,l2_mean,l2_std = d
				mean_list_holder.append(l2_mean)
				std_list_holder.append(l2_std)
				eigen_spectrum,test_eigen_spectrum = zip(*eigs)
				eigen_spectrum = np.sort([np.absolute(x) for x in eigen_spectrum])
				test_eigen_spectrum = np.sort([np.absolute(x) for x in test_eigen_spectrum])
				if n == 0:
					eig_x_coord = eigen_spectrum[eigen_spectrum>0.8]
					eig_len = len(eig_x_coord)
				diff = eigen_spectrum[-eig_len:]-test_eigen_spectrum[-eig_len:]
				eig_list.append(diff.tolist())
			mean_list.append(np.mean(mean_list_holder))
			std_list.append(np.mean(std_list_holder))            
			plt.plot(eig_x_coord,np.array(eig_list).mean(axis=0),label=plot_label)
		plt.legend()
		plt.xlabel('Eigen Value')
		plt.ylabel('Mean Eigen Value Difference')
		plt.savefig(plot_handler.out_file('data_withholding_'+str(t)))
		plt.close()
		plt.figure('l2 error')
		plt.errorbar(np.round((1-np.unique(percentage_token))*100),mean_list,yerr=std_list,fmt='o',label=str(t))
	plt.figure('l2 error')
	plt.xlabel('% withheld')
	plt.ylabel('Mean L2 Difference')
	plt.legend()
	plt.savefig(plot_handler.out_file('combined_withholding_L2'))


