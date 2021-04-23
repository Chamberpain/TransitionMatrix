from transition_matrix.compute.trans_read import TransMat
import numpy as np
import pandas as pd
import copy
import random
from transition_matrix.compute.compute_utils import matrix_size_match
from compute_utilities.list_utilities import flat_list

class TransMatrixWithholding(TransMat):
	pass

	def trajectory_data_withholding_setup(self,percentage):
		# number_matrix = load_sparse_csc(self.base_file+'transition_matrix/number_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
		# standard_error_matrix = load_sparse_csc(self.base_file+'transition_matrix/standard_error_matrix_degree_bins_'+str(self.degree_bins)+'_time_step_'+str(self.date_span_limit)+'.npz')
		old_coord_list = copy.deepcopy(self.list)
		assert_len = len(old_coord_list)
		mylist = self.df.Cruise.unique().tolist() # total list of floats
		remaining_float_list = [mylist[i] for i in sorted(random.sample(xrange(len(mylist)), int(round(len(mylist)*(percentage)))))] #randomly select floats to exclude
		subtracted_float_list = np.array(mylist)[~np.isin(mylist,remaining_float_list)]

		df_holder = self.df[self.df.Cruise.isin(subtracted_float_list)]
		
		index_list = []
		for x in (df_holder[self.end_bin_string].unique().tolist()+df_holder['start bin'].unique().tolist()):
			try:
				index_list.append(old_coord_list.index(list(x)))
			except ValueError:
				print('there was a value error during withholding')
				continue
		index_list = np.unique(index_list)

		new_df = self.df[self.df.Cruise.isin(remaining_float_list)]        
		self.load_df_and_list(new_df)
		assert assert_len == len(old_coord_list)
		transition_matrix,number_matrix = self.matrix_recalc(index_list,old_coord_list,self.transition_matrix,self.number_matrix)
		self.load_transition_and_number_matrix(transition_matrix,number_matrix)

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

					outer_class = TransMat.load_from_type(lat_outer,lon_outer,date_span_limit,traj_file_type)
					outer_class.save()
					outer_class_bins_lat,outer_class_bins_lon = outer_class.bins_generator(outer_class.degree_bins)

					inner_class = TransMat.load_from_type(lat_inner,lon_inner,date_span_limit,traj_file_type)
					inner_class.save()
					inner_class_bins_lat,inner_class_bins_lon = inner_class.bins_generator(inner_class.degree_bins)

					lon_bins_translate = inner_class_bins_lon - inner_class_bins_lon % outer_class.degree_bins[0]
					lat_bins_translate = inner_class_bins_lat - inner_class_bins_lat % outer_class.degree_bins[1]

					lon_dict = dict(zip(inner_class_bins_lon,lon_bins_translate.tolist()))
					lat_dict = dict(zip(inner_class_bins_lat,lat_bins_translate.tolist()))

					inner_idx_list = []
					outer_idx_list = []

					for k,(inner_lat,inner_lon) in enumerate(inner_class.total_list):
						print(k)
						try:
							print(lat_dict[inner_lat])
							print(lon_dict[inner_lon])
							outer_idx_list.append(outer_class.total_list.tolist().index([lat_dict[inner_lat],lon_dict[inner_lon]]))
							inner_idx_list.append(k)
						except ValueError:
							print('I am passing due to value error')
							outer_idx_list.append([])
							inner_idx_list.append(k)
							pass

					translation_list = []
					for ii in range(len(outer_class.total_list)):
						translation_list.append(np.where(np.array(outer_idx_list)==ii)[0])

					row_idx = []
					col_idx = []
					number_idx = []
					data_idx = []
					dummy_total_list = []
					number_matrix = inner_class.new_sparse_matrix(inner_class.number_data)
					for kk,coord in enumerate(outer_class.total_list):
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
							row_idx.append(outer_class.total_list[jj])
							col_idx.append(outer_class.total_list[kk])
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




	# 				base,reduced_res = matrix_size_match(outer_class,reduced) 
	# 				difference_data = base-reduced_res
	# 				reduced = TransPlot(difference_data,number_data = number_idx,total_list = reduced_res.total_list
	# 					,shape = reduced_res.shape, lat_spacing = reduced_res.degree_bins[0]
 # 						,lon_spacing=reduced_res.degree_bins[1],time_step=reduced_res.time_step
	# 					,traj_file_type='reduced')


	# 						data_holder = data_holder[data_holder!=0]
	# 						if data_holder.size>0:
	# 							data_idx.append(data_holder.mean())
	# 							row_idx.append(ii)
	# 							col_idx.append(kk)

					
					


	# 				inner_lats,inner_lons = zip(*inner_class.trans.df[inner_class.info.end_bin_string].unique().tolist())
	# 				outer_lat_relation = pd.cut(inner_lats, outer_class.info.bins_lat,right = False, include_lowest=True,labels=outer_class.info.bins_lat[:-1]).tolist()
	# 				outer_lon_relation = pd.cut(inner_lons, outer_class.info.bins_lon,right = False, include_lowest=True,labels=outer_class.info.bins_lon[:-1]).tolist()
	# 				inner_coord_list = zip(inner_lats,inner_lons)
	# 				outer_coord_relation = zip(outer_lat_relation,outer_lon_relation)
	# 				coord_dictionary_list = []
	# 				for token in zip(inner_coord_list,outer_coord_relation):
	# 					try:
	# 						coord_dictionary_list.append((inner_class.trans.list.index(list(token[0])),outer_class.trans.list.index(list(token[1]))))
	# 					except ValueError:
	# 						continue
	# 				inner_coord_list,outer_coord_list = zip(*coord_dictionary_list)
	# 				inner_coord_lat, inner_coord_lon = zip(*[inner_class.trans.list[x] for x in inner_coord_list])
	# 				outer_coord_lat, outer_coord_lon = zip(*[outer_class.trans.list[x] for x in outer_coord_list])
	# 				assert (abs(np.array(inner_coord_lat)-np.array(outer_coord_lat))<=max_lat).all()
	# 				assert (abs(np.array(inner_coord_lon)-np.array(outer_coord_lon))<=max_lon).all()

	# 				inner_coord_list = np.array(inner_coord_list)
	# 				outer_coord_list = np.array(outer_coord_list)
	# 				test_list = [(inner_coord_list[outer_coord_list==x].tolist(),x) for x in np.sort(np.unique(outer_coord_list))]
	# 				assert (np.array([len(x[0]) for x in test_list])<=max_len).all()

	# 				new_row_list = []
	# 				new_col_list = []
	# 				new_data_list = []
	# 				for k,(col_list,col_index) in enumerate(test_list):
	# 					print('I am working on '+str(k)+' column of outer')
	# 					col_num = len(col_list)
	# 					for n,(row_list,row_index) in enumerate(test_list):
	# 						row_num = len(row_list)
	# 						col_holder = col_list*row_num
	# 						row_list = [val for val in row_list for _ in range(col_num)]
							
	# 						token = inner_class.matrix.transition_matrix[row_list,col_holder]
	# 						if token.mean()!=0:
	# 							new_data_list.append(token.sum())
	# 							new_row_list.append(row_index)
	# 							new_col_list.append(col_index)

	# 						row_dummy = np.where(token!=0)[0].tolist()
	# 						col_dummy = [k]*len(row_dummy)
	# 						data_dummy = token[row_dummy].T.tolist()[0]
	# 						new_row_list += row_dummy
	# 						new_col_list += col_dummy
	# 						new_data_list += data_dummy

	# 				matrix_holder = scipy.sparse.csc_matrix((new_data_list,(new_row_list,new_col_list)),shape=(outer_class.matrix.transition_matrix.shape[0],outer_class.matrix.transition_matrix.shape[1]))
	# 				matrix_holder = inner_class.matrix.rescale_matrix(matrix_holder)

	# 				datalist.append((inner,outer,traj_file_type,matrix_difference_compare(matrix_holder,outer_class.matrix.transition_matrix)))
	# 			else:
	# 				print('they are not divisable')
	# 				continue
	# with open('transition_matrix_resolution_comparison.pickle', 'wb') as fp:
	# 	pickle.dump(datalist, fp)

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