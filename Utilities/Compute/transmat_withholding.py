from transition_matrix.compute.trans_read import TransMat
import numpy as np
import copy
import random
from transition_matrix.compute.compute_utils import matrix_size_match
from GeneralUtilities.Compute.list import flat_list,find_nearest

def matrix_resolution_intercomparison():

	datalist = []
	date_span_limit = 60
	coord_list = [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]
	for outer in coord_list:
		for inner in coord_list:
			for traj_file_type in ['argo','SOSE']:
				if outer==inner:
					continue
				lat_outer,lon_outer = outer
				lat_inner,lon_inner = inner



				if (lat_outer%lat_inner==0)&(lon_outer%lon_inner==0): #outer lat will always be greater than inner lat
					max_len = lat_outer/lat_inner*lon_outer/lon_inner
					max_lat = abs(lat_outer-lat_inner)
					max_lon = abs(lon_outer-lon_inner)              
					print('they are divisable')

					outer_class = TransMat.load_from_type(lat_spacing = lat_outer,lon_spacing = lon_outer,time_step = date_span_limit,traj_type = traj_file_type)
					outer_class_bins_lon = outer_class.trans_geo.get_lon_bins()
					outer_class_bins_lat = outer_class.trans_geo.get_lat_bins()

					inner_class = TransMat.load_from_type(lat_spacing = lat_inner,lon_spacing = lon_inner,time_step = date_span_limit,traj_type = traj_file_type)
					inner_class_bins_lon = inner_class.trans_geo.get_lon_bins()
					inner_class_bins_lat = inner_class.trans_geo.get_lat_bins()


					lon_bins_translate = [find_nearest(outer_class_bins_lon,x) for x in inner_class_bins_lon]
					lat_bins_translate = [find_nearest(outer_class_bins_lat,x) for x in inner_class_bins_lat]



					lon_dict = dict(zip(inner_class_bins_lon,lon_bins_translate))
					lat_dict = dict(zip(inner_class_bins_lat,lat_bins_translate))

					inner_idx_list = []
					outer_idx_list = []

					for k,pos in enumerate(inner_class.trans_geo.total_list):
						print(k)
						try:
							outer_idx_list.append(outer_class.trans_geo.total_list.index(geopy.Point(lat_dict[pos.latitude],lon_dict[pos.longitude])))
							inner_idx_list.append(k)
						except ValueError:
							print('I am passing due to value error')
							outer_idx_list.append([])
							inner_idx_list.append(k)
							pass

					translation_list = []
					for ii in range(len(outer_class.trans_geo.total_list)):
						translation_list.append(np.where(np.array(outer_idx_list)==ii)[0])

					row_idx = []
					col_idx = []
					number_idx = []
					data_idx = []
					dummy_total_list = []
					number_matrix = inner_class.new_sparse_matrix(inner_class.number_data)
					for kk,coord in enumerate(outer_class.trans_geo.total_list):
						print(kk)
						data_holder = inner_class[:,translation_list[kk].tolist()]
						number_holder = number_matrix[:,translation_list[kk].tolist()]
						if data_holder.data.size<1:
							continue
						row_idxs, col_idxs, data = scipy.sparse.find(data_holder)
						dummy, dummy, num_data = scipy.sparse.find(number_holder)

						dummy_total_list.append(tuple(coord.tolist()))
						trans_list_holder = [outer_idx_list[k] for k in row_idxs.tolist() if outer_idx_list[k]]
						for jj in np.unique(trans_list_holder):
							row_idx.append(outer_class.trans_geo.total_list[jj])
							col_idx.append(outer_class.trans_geo.total_list[kk])
							data_idx.append(data[np.where(np.array(trans_list_holder)==jj)[0]].mean())
							number_idx.append(num_data[np.where(np.array(trans_list_holder)==jj)[0]].sum())
					number_idx = np.array(number_idx)
					translation_dict = dict(zip(dummy_total_list,range(len(dummy_total_list))))
					row_idx = [translation_dict[tuple(x)] for x in row_idx]
					col_idx = [translation_dict[tuple(x)] for x in col_idx]
					dummy_total_list = np.array([list(x) for x in dummy_total_list])
					reduced_matrix = scipy.sparse.csc_matrix((data_idx,(row_idx,col_idx))
						,shape=(len(dummy_total_list),len(dummy_total_list)))
					
					string = 'argo_reduced_res'
					for item in outer_class.degree_bins+inner_class.degree_bins:
						string += '-'+str(item).replace('.','_')


					reduced = TransPlot(reduced_matrix,number_data = number_idx,total_list = dummy_total_list
						,shape = reduced_matrix.shape, lat_spacing = outer_class.degree_bins[0]
						,lon_spacing=outer_class.degree_bins[1],time_step=outer_class.time_step
						,traj_file_type=string)
					reduced.rescale()
					reduced.save()


def data_withholding_calc():
	traj_class = TransMatrixWithholding()
	for percentage in np.arange(0.95,0.65,-0.05):
		repeat = 10
		while repeat >0:
			print('repeat is ',repeat) 
			repeat-=1
			test_traj_class = copy.deepcopy(traj_class)
			test_traj_class.trajectory_data_withholding_setup(percentage)
			token_1,test_traj_class = matrix_size_match(copy.deepcopy(traj_class),test_traj_class)
			datalist.append((token,percentage,matrix_difference_compare(token_1.matrix.transition_matrix,test_traj_class.matrix.transition_matrix)))
	with open('transition_matrix_withholding_data.pickle', 'wb') as fp:
		pickle.dump(datalist, fp)



def matrix_datespace_intercomparison():
	datalist = []
	datelist = range(40,300,20)
	coord_list = [(2,3)]
	for lat,lon in coord_list:
		for date2 in datelist:
			for traj_file_type in ['Argo','SOSE']:
				outer_class = transition_class_loader(lat,lon,date2,traj_file_type)
				inner_class = transition_class_loader(lat,lon,20,traj_file_type)

				outer_class,inner_class = matrix_size_match(outer_class,inner_class)
				matrix_token = copy.deepcopy(inner_class.matrix.transition_matrix)
				for dummy in range(int(date2/20)-1):
					inner_class.matrix.transition_matrix = inner_class.matrix.transition_matrix.dot(matrix_token)
					inner_class.matrix.transition_matrix = inner_class.matrix.rescale_matrix(inner_class.matrix.transition_matrix)
				outer_class, inner_class = matrix_size_match(outer_class,inner_class)
				datalist.append((date2,traj_file_type,matrix_difference_compare(inner_class.matrix.transition_matrix,outer_class.matrix.transition_matrix)))
	with open('transition_matrix_datespace_data.pickle', 'wb') as fp:
		pickle.dump(datalist, fp)
