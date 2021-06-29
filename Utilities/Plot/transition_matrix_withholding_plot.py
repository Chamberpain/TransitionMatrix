import matplotlib.pyplot as plt
from TransitionMatrix.Utilities.Compute.__init__ import ROOT_DIR as data_root
from TransitionMatrix.Utilities.Plot.__init__ import ROOT_DIR
from GeneralUtilities.Filepath.instance import FilePathHandler
import pickle
import numpy as np

data_handler = FilePathHandler(data_root,'transmat_withholding')
plot_handler = FilePathHandler(ROOT_DIR,'transition_matrix_withholding_plot')

plot_style_dict = {'argo':'--','SOSE':':'}
plot_color_dict = {(1,1):'teal',(2,2):'red',(2,3):'blue',(3,3):'yellow',(4,4):'orange',(4,6):'green'}


def argos_gps_plot():
	with open(data_handler.tmp_file('argos_gps_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
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
	ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,lat,lon,time,season = zip(*datalist)
	fig = plt.figure(figsize=(9,8))
	ax1 = fig.add_subplot(2,1,1)
	ax2 = fig.add_subplot(2,1,2)
	for k,pos in enumerate(np.unique(season)):
		for grid in list(plot_color_dict.keys()):
			lat_mask = np.array([grid[0]==x for x in lat])
			lon_mask = np.array([grid[1]==x for x in lon])
			season_mask = np.array([pos==x for x in season])
			mask = lat_mask&lon_mask&season_mask
			ew_mean_diff_holder = np.array(ew_mean_diff)[mask]
			ns_mean_diff_holder = np.array(ns_mean_diff)[mask]
			ew_std_diff_holder = np.array(ew_std_diff)[mask]
			ns_std_diff_holder = np.array(ns_std_diff)[mask]

			out = np.sqrt(ew_mean_diff_holder**2+ns_mean_diff_holder**2)
			ax1.plot(np.unique(time),out,linestyle=plot_style_dict[pos],color=plot_color_dict[grid],label=str(grid))
			ax1.scatter(np.unique(time),out,color=plot_color_dict[grid])
			out = np.sqrt(ew_std_diff_holder**2+ns_std_diff_holder**2)
			ax2.plot(np.unique(time),out,linestyle=plot_style_dict[pos],color=plot_color_dict[grid],label=str(grid))
			ax2.scatter(np.unique(time),out,color=plot_color_dict[grid])
		if k ==0:
			ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
      		ncol=3, fancybox=True, shadow=True)

	ax1.set_xlim(28,182)
	ax2.set_xlim(28,182)
	ticks = [30,60,90,120,180]
	ax1.set_xticks(ticks)
	ax2.set_xticks(ticks)
	ax2.set_xlabel('Timestep')
	ax1.set_ylabel('Mean Difference')
	ax2.set_ylabel('Mean Difference')

	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(plot_handler.out_file('seasonal_plot'))
	plt.close()


def resolution_difference_plot():
	plt.rcParams['font.size'] = '16'
	with open(data_handler.tmp_file('resolution_difference_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,lat,lon,time,pos_type = zip(*datalist)
	fig = plt.figure(figsize=(9,8))
	ax1 = fig.add_subplot(2,1,1)
	ax2 = fig.add_subplot(2,1,2)
	for system in ['argo']:
		for grid in list(plot_color_dict.keys()):

			lat_mask = np.array([grid[0]==x for x in lat])
			lon_mask = np.array([grid[1]==x for x in lon])
			system_mask = np.array([system==x for x in pos_type])
			mask = lat_mask&lon_mask&system_mask&percent_mask
			ew_mean_diff_holder = np.array(ew_mean_diff)[mask]
			ns_mean_diff_holder = np.array(ns_mean_diff)[mask]
			ew_std_diff_holder = np.array(ew_std_diff)[mask]
			ns_std_diff_holder = np.array(ns_std_diff)[mask]

			out = np.sqrt(ew_mean_diff_holder**2+ns_mean_diff_holder**2)
			print('I am plotting ',system)
			ax1.plot(np.unique(time),out,linestyle=plot_style_dict[system],color=plot_color_dict[grid],label=str(grid))
			ax1.scatter(np.unique(time),out,color=plot_color_dict[grid])
			out = np.sqrt(ew_std_diff_holder**2+ns_std_diff_holder**2)
			ax2.plot(np.unique(time),out,linestyle=plot_style_dict[system],color=plot_color_dict[grid],label=str(grid))
			ax2.scatter(np.unique(time),out,color=plot_color_dict[grid])
	ax1.set_xlim(28,122)
	ax2.set_xlim(28,122)
	ticks = [30,60,90,120]
	ax1.set_xticks(ticks)
	ax2.set_xticks(ticks)
	ax2.set_xlabel('Timestep')
	ax1.set_ylabel('Mean Difference')
	ax2.set_ylabel('Mean Difference')
	ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3, fancybox=True, shadow=True)
	ax1.annotate('a', xy = (0.9,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.9,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(plot_handler.out_file('data_withholding_plot'))
	plt.close()


def data_withholding_plot():
	plot_color_dict = {0.95:'teal',0.9:'red',0.85:'blue',0.8:'yellow',0.75:'orange',0.7:'green'}
	plt.rcParams['font.size'] = '16'
	with open(data_handler.tmp_file('transition_matrix_withholding_data'), 'rb') as fp:
		datalist = pickle.load(fp)   
	fp.close()
	ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,lat,lon,time,descriptor = zip(*datalist)
	pos_type,percentage = zip(*descriptor)
	fig = plt.figure(figsize=(9,8))
	ax1 = fig.add_subplot(2,1,1)
	ax2 = fig.add_subplot(2,1,2)
	for system in ['SOSE']:
		for grid in [(2,2)]:
			for percent in np.unique(percentage):
				time_list = []
				mean_mean_list = []
				mean_std_list = []
				std_mean_list = []
				std_std_list = []
				for t in np.sort(np.unique(time)):
					time_list.append(t)
					percent_mask = np.array([percent==x for x in percentage])
					lat_mask = np.array([grid[0]==x for x in lat])
					lon_mask = np.array([grid[1]==x for x in lon])
					system_mask = np.array([system==x for x in pos_type])
					time_mask = np.array([t==x for x in time])
					mask = lat_mask&lon_mask&system_mask&percent_mask&time_mask

					ew_mean_diff_holder = np.array(ew_mean_diff)[mask]
					ns_mean_diff_holder = np.array(ns_mean_diff)[mask]

					out = np.sqrt(ew_mean_diff_holder**2+ns_mean_diff_holder**2)
					mean_mean_list.append(out.mean())
					mean_std_list.append(out.std())

					ew_std_diff_holder = np.array(ew_std_diff)[mask]
					ns_std_diff_holder = np.array(ns_std_diff)[mask]
					out = np.sqrt(ew_mean_diff_holder**2+ns_mean_diff_holder**2)

					std_mean_list.append(out.mean())
					std_std_list.append(out.std())


				ax1.errorbar(time_list,mean_mean_list,yerr=mean_std_list,linestyle=plot_style_dict[system],color=plot_color_dict[percent],label=percent)
				ax1.scatter(time_list,mean_mean_list,color=plot_color_dict[percent])

				ax2.errorbar(time_list,std_mean_list,yerr=std_std_list,linestyle=plot_style_dict[system],color=plot_color_dict[percent],label=percent)
				ax2.scatter(time_list,std_mean_list,color=plot_color_dict[percent])
	ax1.set_xlim(28,92)
	ax2.set_xlim(28,92)
	ticks = [30,60,90]
	ax1.set_xticks(ticks)
	ax2.set_xticks(ticks)
	ax2.set_xlabel('Timestep')
	ax1.set_ylabel('Mean Difference')
	ax2.set_ylabel('Mean Difference')
	ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.3),
          ncol=3, fancybox=True, shadow=True)
	ax1.annotate('a', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	ax2.annotate('b', xy = (0.1,0.9),xycoords='axes fraction',zorder=10,size=22,bbox=dict(boxstyle="round", fc="0.8"),)
	plt.savefig(plot_handler.out_file('data_withholding_plot'))
	plt.close()


