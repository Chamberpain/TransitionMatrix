from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import os, sys
from TransitionMatrix.Utilities.Compute.trans_read import BaseMat,TransMat
# get an absolute path to the directory that contains mypackage
from TransitionMatrix.Utilities.Plot.plot_utils import cartopy_setup,transition_vector_to_plottable,plottable_to_transition_vector
from TransitionMatrix.Utilities.Plot.argo_data import Argo,SOCCOM
from TransitionMatrix.Utilities.Plot.transition_matrix_plot import TransPlot
from GeneralUtilities.Compute.list import find_nearest
import scipy 
from netCDF4 import Dataset
import matplotlib.cm as cm
from TransitionMatrix.__init__ import ROOT_DIR
from scipy.interpolate import interp2d
from shapely.geometry import Point, Polygon
import random

noise_factor = 2
unit_dict = {'salt':'psu','temp':'C','dic':'mol m$^{-2}$','o2':'mol m$^{-2}$'}

class CovBase(BaseMat):
	""" This class uses the L to mean the length scale
	class is mostly used for inverse calculations and generating specifically plottable instances"""
	def __init__(self,arg1,shape=None,total_list=None,lat_spacing=None,lon_spacing=None,l=None,traj_file_type=None,**kwargs):
		self.l = l
		super(CovBase,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing,traj_file_type=traj_file_type,**kwargs)

class InverseInstance(CovBase):
	def __init__(self,arg1,shape=None,total_list=None,lat_spacing=None,lon_spacing=None,l=None,variable_list=None,traj_file_type='covariance',**kwargs):
		super(InverseInstance,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing,l=l,traj_file_type=traj_file_type,**kwargs)
		try:
			variable_list = [x.decode("utf-8") for x in variable_list]
		except (AttributeError,TypeError) as e:
			pass
		self.variable_list = variable_list

	def get_cov(self,row_var,col_var):
		row_idx = self.variable_list.index(row_var)
		col_idx = self.variable_list.index(col_var)
		split_array = np.split(np.split(self.todense(),len(self.variable_list))[row_idx],len(self.variable_list),axis=1)[col_idx]
		return CovInstance(split_array,split_array.shape,self.total_list,self.degree_bins[0],self.degree_bins[1],self.l,row_var,col_var)

	def save(self,filename=False):
		if not filename:
			filename = self.make_filename(traj_type=self.traj_file_type,degree_bins=self.degree_bins,l=self.l)
		np.savez(filename, data=self.data, indices=self.indices,indptr=self.indptr, 
		shape=self.shape, total_list=self.total_list,lat_spacing=self.degree_bins[0],
		lon_spacing=self.degree_bins[1],
		traj_file_type=self.traj_file_type,l=self.l,variable_list=self.variable_list)

	@staticmethod
	def load(filename):
		loader = np.load(filename,allow_pickle=True)
		print(loader['lat_spacing'])
		return InverseInstance((loader['data'], loader['indices'], loader['indptr']),shape=loader['shape'],
			total_list = loader['total_list'],lat_spacing=loader['lat_spacing'],
			lon_spacing=loader['lon_spacing'],l=loader['l'],traj_file_type=loader['traj_file_type'],variable_list = loader['variable_list'].tolist())

	@staticmethod
	def load_from_type(lat_spacing=None,lon_spacing=None,l=None,traj_type=None):
		degree_bins = [np.array(float(lat_spacing)),np.array(float(lon_spacing))]
		file_name = InverseInstance.make_filename(traj_type=traj_type,degree_bins=degree_bins,l=l)
		return InverseInstance.load(file_name)

	@staticmethod
	def make_filename(traj_type=None,degree_bins=None,l=None):
		base = ROOT_DIR+'/Output/Data/'
		degree_bins = [float(degree_bins[0]),float(degree_bins[1])]
		return base+traj_type+'/'+str(l)+'-'+str(degree_bins)+'.npz'


class CovInstance(CovBase):
	def __init__(self,arg1,shape=None,total_list=None,lat_spacing=None,lon_spacing=None,l=None,var1=None,var2=None,traj_file_type='covariance_instance',**kwargs):
		self.var1 = var1
		self.var2 = var2
		super(CovInstance,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing,l=l,traj_file_type=traj_file_type,**kwargs)

	def plot_variance(self):
		base = ROOT_DIR+'/plots/cm2p6_covariance/'
		lat_grid, lon_grid = self.bins_generator(self.degree_bins)
		plottable = transition_vector_to_plottable(lat_grid,lon_grid,self.total_list,self.diagonal())
		plottable_mean = self.diagonal().mean()
		plottable_std = self.diagonal().std()
		vmax_holder = min(plottable_mean+4*plottable_std,self.diagonal().max())
		vmin_holder = max(plottable_mean-4*plottable_std,self.diagonal().min())
		if self.var1==self.var2:
			vmax_holder = min(plottable_mean+4*plottable_std,self.diagonal().max())
			vmin_holder = max(plottable_mean-4*plottable_std,self.diagonal().min())
			vmin_holder = max(0,vmin_holder)
		XX,YY,ax,fig = cartopy_setup(lat_grid,lon_grid,'')
		plottable[plottable>vmax_holder]=vmax_holder
		plt.contourf(XX,YY,plottable,vmax=vmax_holder,vmin=vmin_holder)
		plt.colorbar(label=self.var1+'-'+self.var2+' variance')
		plt.savefig(base+self.var1+'-'+self.var2)
		plt.close()

class HInstance(scipy.sparse.csc_matrix):
	translation_dict ={'temp':'thetao','salt':'so','dic':'ph'}
	def __init__(self, arg1, shape=None,**kwargs):
		super(HInstance,self).__init__(arg1, shape=shape,**kwargs)

	@staticmethod
	def randomly_generate(number,total_list=None,variable_list=None,degree_bins=None,limit=[-181,181,-90,90]):
		if type(total_list)==np.ndarray:
			total_list = total_list.tolist()
		lllon,urlon,lllat,urlat = limit
		coords = [(lllon,lllat),(lllon,urlat),(urlon,urlat),(urlon,lllat),(lllon,lllat)]
		poly = Polygon(coords)
		total_truth = [Point(x[0],x[1]).within(poly) for x in total_list]
		total_list_holder = np.array(total_list)[total_truth].tolist()
		locations = []
		while number > 0:
			if number>len(total_list): 
				print('number is greater than total_list')
				num_holder = len(total_list)
				print(num_holder)
			else:
				num_holder = number
			lat,lon = zip(*random.sample(total_list,num_holder))
			for var in variable_list:
				locations += list(zip(lon,lat,[var]*len(lat)))
			number -= len(total_list)

			# print(locations)
		return HInstance.generate_from_locations(locations,total_list=total_list,variable_list=variable_list,degree_bins=degree_bins)

	@staticmethod
	def generate_from_transition_matrix(float_list,transmat = None,new_total_list=None,variable_list=None,degree_bins=None):
		var_translate_dict = {'thetao':'temp','so':'salt','ph':'dic','o2':'o2','chl':'chl'}
		variable_list = [var_translate_dict[x] for x in variable_list]
		translation_dict = {}
		old_total_list = float_list[0].total_list

		old_total_list = [list(x) for x in old_total_list]
		new_total_list = [list(x) for x in new_total_list]
		for k,bin_tuple in enumerate(old_total_list):
			try:
				translation_dict[k]=new_total_list.index(list(bin_tuple))
			except ValueError:
				pass
		print('completed the dictionary, it looks like')
		print(translation_dict)
		if transmat is None:
			transmat = scipy.sparse.csc_matrix(np.identity(len(old_total_list)))
		row_list = []
		col_list = []
		data_list = []
		k = 0
		for dummy_float in float_list:
			dummy_float.variables = [x for x in dummy_float.variables if x in variable_list]
			for row_idx,col_idx,float_num in zip(*scipy.sparse.find(dummy_float)):
				for _ in range(float_num):
					k += 1

					try:
						row_instance,col_instance,data_instance = scipy.sparse.find(transmat[:,row_idx])
					except IndexError:
						print('there was an error in transmat parsing, it looks like')
						print(transmat)
						raise
					row_mask = [x in translation_dict.keys() for x in row_instance]

					row_instance = [translation_dict[x] for x in row_instance if x in translation_dict.keys()]
					for var in dummy_float.variables:
						print(var)
						var_idx = variable_list.index(var)
						row_list += [var_idx*len(new_total_list)+x for x in row_instance]

					data_list += data_instance[row_mask].tolist()*len(dummy_float.variables)
					col_list += [k] * len(row_instance)*len(dummy_float.variables)
					assert len(col_list)==len(row_list)
					assert len(data_list)==len(row_list)
		return HInstance((data_list,(row_list,col_list)),shape = (len(new_total_list)*len(variable_list),max(col_list)+1))

	@ staticmethod
	def generate_from_float_class(float_class_list,variable_list=None):
		locations = []
		for float_class in float_class_list:
				lat = float_class.df.latitude.tolist()
				lon = float_class.df.longitude.tolist()
				for var in float_class.variables:
						locations += zip(lat,lon,[var]*len(lat))
		return HInstance.generate_from_locations(locations,total_list=float_class.total_list,variable_list=variable_list,degree_bins=float_class.degree_bins)


	@ staticmethod
	def generate_from_locations(locations,total_list=None,variable_list=None,degree_bins=None):
		if type(total_list)==np.ndarray:
			total_list = total_list.tolist()
		lat_list,lon_list = BaseMat.bins_generator(degree_bins)
		lat_loc_list = []
		lon_loc_list = []
		data = []
		col_idx = []
		column_idx = []
		for loc in locations:
			lat,lon,var = loc
			rounded_lat = find_nearest(lat_list,lat)
			rounded_lon = find_nearest(lon_list,lon)
			assert abs(rounded_lat-lat)<2
			assert abs(rounded_lon-lon)<2
			lat_loc_list.append(rounded_lat)
			lon_loc_list.append(rounded_lon)

			try:
				tl_idx = total_list.index([rounded_lon,rounded_lat])
			except ValueError:
				print('lat = '+str(rounded_lat)+' lon = '+str(rounded_lon)+' is a problem')
			if var in variable_list:
				vl_idx = variable_list.index(var)
			else:
				vl_idx = variable_list.index(HInstance.translation_dict[var])


			column_idx.append(len(total_list)*vl_idx+tl_idx)

		for col in np.unique(column_idx):
			data.append(np.sum(np.array(column_idx)==col))
			col_idx.append(col)
		row_idx = range(len(col_idx))
		H = scipy.sparse.csc_matrix((data,(row_idx,col_idx)),shape=(len(col_idx),len(total_list)*len(variable_list)))
		H = H.T
		return HInstance(H,H.shape)

def plot_all_covariance():
	""" There might be a problem with this because the cov in the lower triangular matrix might be transposed, need to ask bruce"""
	cov = InverseInstance.load_from_type(2,2,300,'covariance')
	for var1 in cov.variable_list:
		for var2 in cov.variable_list:
			cov_instance = cov.get_cov(var1,var2)
			cov_instance.plot_variance()

def plot_example_variance_constrained():
	invinst = InverseInstance.load_from_type(2,2,300) 
	lat,lon = zip(*random.sample(invinst.total_list,300)) 
	locations = zip(lat,lon,['dic']*len(lat))
	hinstance = HInstance.generate_from_locations(locations,invinst.total_list,['temp'],invinst.degree_bins)
	cov = invinst.get_cov('dic','dic')
	output_mask = np.array(hinstance.sum(axis=0)>0).ravel()
	noise = scipy.sparse.diags([cov.diagonal().mean()*noise_factor]*hinstance.shape[0])
	denom = hinstance.dot(cov).dot(hinstance.T)+noise
	denom = scipy.sparse.csc_matrix(denom)
	inv_denom = scipy.sparse.linalg.inv(denom)
	cov_subtract = cov.dot(hinstance.T).dot(inv_denom).dot(hinstance).dot(cov)
	p_hat = cov-cov_subtract        

	def plot_data(data):
			(bins_lat,bins_lon)=BaseMat.bins_generator(cov.degree_bins)
			plottable = transition_vector_to_plottable(bins_lat,bins_lon,invinst.total_list,data)
			XX,YY,m = cartopy_setup(bins_lat,bins_lon,'argo')  
			# mean = p_hat.diagonal().mean()
			# std = p_hat.diagonal().std()
			m.pcolormesh(XX,YY,plottable)
			return m 
	m = plot_data(p_hat.diagonal())
	plt.savefig(ROOT_DIR+'/plots/example_phat')
	plt.close()
	m = plot_data(cov_subtract.diagonal())
	plt.savefig(ROOT_DIR+'/plots/example_cov_subtract')
	plt.close()
	eigs = scipy.sparse.linalg.eigs(p_hat)
	for k in range(6):
			dummy = eigs[1][:,k]
			m=plot_data(dummy)
			plt.savefig(ROOT_DIR+'/plots/example_evec'+str(k))
			plt.close()

def plot_array_variance_constrained():
	base = ROOT_DIR+'/plots/cm2p6_covariance/'
	global_cov = InverseInstance.load_from_type(2,2,1500,'global_covariance')
	submeso_cov = InverseInstance.load_from_type(2,2,300,'submeso_covariance')
	argo = Argo.recent_floats(global_cov.degree_bins,global_cov.total_list)
	argo.df = argo.df[argo.df.latitude<-10]
	soccom = SOCCOM.recent_floats(global_cov.degree_bins,global_cov.total_list)

	cov = global_cov+submeso_cov
	hinstance = HInstance.generate_from_float_class([soccom],variable_list=global_cov.variable_list)
	output_mask = np.array(hinstance.sum(axis=0)>0).ravel()
	noise = scipy.sparse.diags(cov.diagonal()[output_mask]*noise_factor)
	def calculate(cov):

		denom = hinstance.dot(cov).dot(hinstance.T)+noise
		inv_denom = scipy.sparse.linalg.inv(scipy.sparse.csc_matrix(denom.todense()))
		cov_subtract = cov.dot(hinstance.T).dot(inv_denom).dot(hinstance).dot(cov)
		# p_hat = cov-cov_subtract
		return cov_subtract

	cov_subtract = calculate(cov)
	p_hat = cov-cov_subtract

	lat_grid, lon_grid = cov.bins_generator(global_cov.degree_bins)
	zipper = zip(np.split(p_hat.diagonal()/cov.diagonal(),4),global_cov.variable_list)
	var1_dict = {'salt':'Salinity','temp': 'Temperature','dic': 'DIC', 
	'o2': 'Oxygen'}
	var2_dict = {'salt':'Salinity','dic':'DIC','temp':'Temperature','o2':'Oxygen'}

	for c,var in zipper:
			plottable = transition_vector_to_plottable(lat_grid,lon_grid,global_cov.total_list,c)
			plottable_mean = c[c!=0].mean()
			plottable_std = c[c!=0].std()
			XX,YY,m = basemap_setup(lat_grid,lon_grid,'Polar')
			m.pcolormesh(XX,YY,plottable*100,vmax=100,vmin=0)
			plt.colorbar(label='Unconstrained '+var1_dict[var]+' Variance (%)')
			# m = argo.scatter_plot(m=m)
			m = soccom.scatter_plot(m=m)
			plt.savefig(var+'_for_matt_no_argo')
			# m = argo.scatter_plot(m=m)
			# m = soccom.scatter_plot(m=m)
			plt.close()


def targeted_array_plots():
	from transition_matrix.makeplots.argo_data import Float
	import pandas as pd
	unconstrained_list = []
	plot_num = 0
	base = ROOT_DIR+'/plots/targeted_array/'
	invinstancelarge = InverseInstance.load_from_type(2,2,1500,traj_type='global_covariance')
	invinstancesmall = InverseInstance.load_from_type(2,2,300,traj_type='submeso_covariance')


	(bins_lat,bins_lon)=BaseMat.bins_generator(invinstancelarge.degree_bins)
	covinstance = invinstancelarge.get_cov('dic','dic')+invinstancesmall.get_cov('dic','dic')
	covinstance.variable_list = ['dic']
	covinstance.total_list = invinstancelarge.get_cov('dic','dic').total_list
	covinstance.var1 = invinstancelarge.get_cov('dic','dic').var1
	covinstance.var2 = invinstancelarge.get_cov('dic','dic').var2

	float_array = Float(np.zeros([len(covinstance.total_list),1]),shape=np.zeros([len(covinstance.total_list),1]).shape,degree_bins=covinstance.degree_bins,total_list=covinstance.total_list)
	float_array.df = pd.DataFrame({'latitude':[],'longitude':[]})
	float_array.variables = ['dic']
	while float_array.sum()<1001:
		# if float_array.sum()==700:
		# 	covinstance = invinstance.get_cov('dic','dic')
		# 	covinstance.variable_list = ['dic']
		hinstance = HInstance.generate_from_float_class([float_array],variable_list=[covinstance.var1])
		def calculate_phat(hinstance,covinstance):
			output_mask = np.array(hinstance.sum(axis=0)>0).ravel()
			noise = scipy.sparse.diags([covinstance.diagonal().mean()*noise_factor]*hinstance.shape[0])
			denom = hinstance.dot(covinstance).dot(hinstance.T)+noise
			denom = scipy.sparse.csc.csc_matrix(denom)
			if not denom.data.tolist():
				return covinstance
			inv_denom = scipy.sparse.linalg.inv(denom)
			if not type(inv_denom)==scipy.sparse.csc.csc_matrix:
				inv_denom = scipy.sparse.csc.csc_matrix(inv_denom)
			cov_subtract = covinstance.dot(hinstance.T).dot(inv_denom).dot(hinstance).dot(covinstance)
			p_hat = covinstance-cov_subtract		
			return p_hat

		def get_index_of_first_eigen_vector(p_hat):
			eigs = scipy.sparse.linalg.eigs(p_hat)
			e_vec = eigs[1][:,0]
			print(e_vec.max())
			print(np.where(e_vec == e_vec.max()))
			idx = np.where(e_vec == e_vec.max())[0][0]
			return idx,e_vec


		p_hat = calculate_phat(hinstance,covinstance)
		unconstrained_list.append(p_hat.diagonal().sum())
		if float_array.sum()%10 == 0:

			plt.subplot(2,1,1)
			XX,YY,m = basemap_setup(bins_lat,bins_lon,'Argo')  
			plottable = transition_vector_to_plottable(bins_lat,bins_lon,covinstance.total_list,p_hat.diagonal().ravel())
			m.pcolormesh(XX,YY,plottable) 
			plt.colorbar(label = unit_dict[covinstance.variable_list[0]])
			float_array.scatter_plot(m)
			plt.subplot(2,1,2)
			XX,YY,m = basemap_setup(bins_lat,bins_lon,'Argo')  
			plottable = transition_vector_to_plottable(bins_lat,bins_lon,covinstance.total_list,p_hat.diagonal()/covinstance.diagonal())
			m.pcolormesh(XX,YY,100-plottable*100,vmin=0,vmax=100) 
			plt.colorbar(label = '% constrained')
			float_array.scatter_plot(m)
			plt.savefig(base+str(plot_num))
			plt.close()	
			plot_num += 1

		idx, e_vec = get_index_of_first_eigen_vector(p_hat)
		if float_array[idx].data.tolist():
			float_array[idx] = float_array[idx].data[0]+1
		else:
			float_array[idx] = 1

	random_error_list = []
	random_std_list = []
	num_list = np.arange(900,1500,10)
	for num in num_list:
		print(num)
		dummy_list = []
		for k in range(10):
			print(k)
			hinstance = HInstance.randomly_generate(num,total_list=invinstancelarge.total_list,variable_list=[covinstance.var1],degree_bins=invinstance.degree_bins)
			p_hat = calculate_phat(hinstance,covinstance)
			dummy_list.append(p_hat.diagonal().sum())
			print(dummy_list)
		random_error_list.append(np.mean(dummy_list))
		random_std_list.append(np.std(dummy_list))
	random_y1 = np.array(random_error_list)-np.array(random_std_list)
	random_y2 = np.array(random_error_list)+np.array(random_std_list)
	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.fill_between(num_list, random_y1,random_y2,color='g',alpha=0.2)
	ax.plot(num_list,random_error_list,color='g',label='Random')
	ax.plot(range(len(unconstrained_list)),unconstrained_list,color='k',label='Targeted')
	ax.plot(num_list,[unconstrained_list[-1]]*len(num_list),linestyle='--',color='k')
	plt.ylabel('Unobserved Scaled Variance $mol\ m^{-2}$')
	plt.xlabel('Float Deployed')
	plt.xlim([900,1500])
	plt.ylim([random_y1[-1],random_y2[0]])
	plt.legend()
	plt.savefig(base+'random_variance_constrained')
	plt.close()


	SNR_list = [10000,5000,1000,500,100,50,10,5,1,0.5,0.1,0.05,0.01]
	mean_targeted_list = []
	std_targeted_list = []
	mean_random_list = []
	std_random_list = []
	scaling = cov_array.calculate_scaling(2,2)
	scaling = scipy.sparse.csc_matrix(scaling)
	targeted_h_instance = HInstance.generate_from_float_class(float_array,variable_list=[covinstance.var1])
	for SNR in SNR_list:
		print(SNR)
		dummy_random_list = []
		dummy_targeted_list = []
		for _ in range(10):
			print(_)
			covinstance = invinstance.get_cov('dic','dic')
			covinstance.data = covinstance.data+scaling.data*np.random.normal(scale=1/np.sqrt(SNR)*np.std(covinstance.data),size=len(scaling.data))
			random_h_instance = HInstance.randomly_generate(1000,total_list=invinstance.total_list,variable_list=[covinstance.var1],degree_bins=invinstance.degree_bins)
			p_hat = calculate_phat(random_h_instance,covinstance)
			p_hat.data[p_hat.data<=0]=0.01

			dummy_random_list.append(p_hat.diagonal().sum())
			print(dummy_random_list[-1])
			p_hat = calculate_phat(targeted_h_instance,covinstance)
			p_hat.data[p_hat.data<=0]=0.01
			dummy_targeted_list.append(p_hat.diagonal().sum())
			print(dummy_targeted_list[-1])

		mean_targeted_list.append(np.mean(dummy_targeted_list))
		std_targeted_list.append(np.std(dummy_targeted_list))
		mean_random_list.append(np.mean(dummy_random_list))
		std_random_list.append(np.std(dummy_random_list))

	plot_num = 12

	target_y_1 = np.array(mean_targeted_list[:plot_num])+np.array(std_targeted_list[:plot_num])
	target_y_2 = np.array(mean_targeted_list[:plot_num])-np.array(std_targeted_list[:plot_num])
	random_y_1 = np.array(mean_random_list[:plot_num])+np.array(std_random_list[:plot_num])
	random_y_2 = np.array(mean_random_list[:plot_num])-np.array(std_random_list[:plot_num])

	fig = plt.figure()
	ax = fig.add_subplot(1, 1, 1)
	ax.fill_between(SNR_list[:plot_num], random_y_2,random_y_1,color='g',alpha=0.2)
	ax.plot(SNR_list[:plot_num],mean_random_list[:plot_num],color='g',label='Random')
	ax.fill_between(SNR_list[:plot_num], target_y_2,target_y_1,color='k',alpha=0.2)
	ax.plot(SNR_list[:plot_num],mean_targeted_list[:plot_num],color='k',label='Targeted')
	plt.ylim([min(target_y_2)-200,max(random_y_1)])
	plt.xlim([min(SNR_list[:plot_num]),max(SNR_list[:plot_num])])

	ax.set_xscale('log')
	ax.set_yscale('log')
	plt.ylabel('Unobserved Scaled Variance $(mol\ m^{-2})$')
	plt.xlabel('SNR')
	plt.legend()
	plt.savefig(base+'SNR_plot')
	plt.close()				

def argo_evolution_plot():
	base = ROOT_DIR+'/plots/argo_evolution/'        
	for k,days in enumerate([10,20,30,40,50,60,70,80,90,100,120,140,160,180]):
			transmat = TransMat.load_from_type(1,1,days)
			argo = Argo.recent_floats(transmat.degree_bins,transmat.total_list)
			argo_holder = Argo(transmat.todense().dot(argo.todense()),degree_bins=transmat.degree_bins,total_list=transmat.total_list)
			argo_holder.data = argo_holder.data*100 
			argo_holder.grid_plot()
			plt.colorbar(label='Probability (%)')
			plt.title('Day '+str(days))
			plt.savefig(base+str(k))
			plt.close()

def goship_line_plot(depth_level=0):
	def calculate_and_save_p_hat(depth_level,cov):
		argo = Argo.recent_floats(cov.degree_bins,cov.total_list,age=False)
		soccom = SOCCOM.recent_floats(cov.degree_bins,cov.total_list,age=False)
		hinstance = HInstance.generate_from_float_class([argo,soccom],variable_list=cov.variable_list)
		hinstance = hinstance.T
		noise = scipy.sparse.diags([cov.diagonal().mean()*noise_factor]*hinstance.shape[0])
		denom = hinstance.dot(cov).dot(hinstance.T)+noise
		denom = scipy.sparse.csc_matrix(denom)
		inv_denom = scipy.sparse.linalg.inv(denom)

		cov_subtract = scipy.sparse.csc_matrix(cov).dot(hinstance.T.dot(inv_denom).dot(hinstance)).dot(scipy.sparse.csc_matrix(cov))
		cov_subtract_holder = InverseInstance(cov_subtract,shape=cov_subtract.shape,total_list=global_cov.total_list,
			lat_spacing=global_cov.degree_bins[0],lon_spacing=global_cov.degree_bins[1]
			,l='p_hat',variable_list=global_cov.variable_list,
			traj_file_type='cm4_submeso_covariance_'+str(depth_level))
		p_hat = scipy.sparse.csc_matrix(cov)-scipy.sparse.csc_matrix(cov_subtract)
		holder = InverseInstance(p_hat,shape=p_hat.shape,total_list=global_cov.total_list,
			lat_spacing=global_cov.degree_bins[0],lon_spacing=global_cov.degree_bins[1]
			,l='p_hat',variable_list=global_cov.variable_list,
			traj_file_type='cm4_submeso_covariance_'+str(depth_level))
		holder.save()
		return holder
	global_cov = InverseInstance.load_from_type(2,2,1500,'cm4_global_covariance_'+str(depth_level))
	submeso_cov = InverseInstance.load_from_type(2,2,300,'cm4_submeso_covariance_'+str(depth_level))
	cov = global_cov+submeso_cov
	try: 
		p_hat = InverseInstance.load_from_type(lat_spacing=2,lon_spacing=2,l='p_hat',traj_type='cm4_submeso_covariance_'+str(depth_level))
	except FileNotFoundError:
		p_hat = calculate_and_save_p_hat(depth_level,cov)
	cov_subtract = cov - p_hat

	def return_goship_locs():
		import pandas as pd
		import re
		base_goship_folder = ROOT_DIR+'/data/goship_lines/'
		df_list = []
		df = pd.read_csv(base_goship_folder+'p04_hy1.csv',skiprows=3,usecols=[1,10,11],names=['Cruise','Lats','Lons'])
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		df = pd.read_csv(base_goship_folder+'33AT20120324_hy1.csv',skiprows=52,usecols=[1,9,10],names=['Cruise','Lats','Lons'])
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		df = pd.read_csv(base_goship_folder+'33AT20120419_hy1.csv',skiprows=52,usecols=[1,9,10],names=['Cruise','Lats','Lons'])
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		df = pd.read_csv(base_goship_folder+'49NZ20140717_hy1.csv',skiprows=7,usecols=[1,9,10],names=['Cruise','Lats','Lons'])
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		holder = []
		for file in os.listdir(base_goship_folder+'ar07_74JC20140606_ct1'):
				open_file = open(base_goship_folder+'ar07_74JC20140606_ct1/'+file,'r')
				lat_lon = []
				for line in open_file.readlines()[10:12]:
						lat_lon.append(re.findall(r'[-+]?\d+.\d+', line)[0])
				holder.append(tuple(lat_lon))
		lats,lons = zip(*holder)
		df = pd.DataFrame({'Cruise':['AR07']*len(lats),'Lats':lats,'Lons':lons})
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		return df_list
	lat_grid, lon_grid = cov.bins_generator(cov.degree_bins)
	zipper = zip(np.split(cov_subtract.diagonal(),len(cov.variable_list)),np.split(cov.diagonal(),len(cov.variable_list)),cov.variable_list)

	# var1_dict = {'salt':'Salinity (psu m)','temp': 'Temperature (C m)','dic': 'DIC ($mol\ m^{-2}$)', 
	# 'o2': 'Oxygen ($mol\ m^{-2}$)'}
	var1_dict = {'so':'Salinity (psu)','thetao': 'Temperature (C)','ph':'','dic':'DIC ($mol\ m^{-2}$)', 
	'o2': 'Oxygen ($mol\ m^{-2}$)','chl': 'Chlorophyll ($mg\ m^{-3}$)'}
	data_dict = {}
	df_list = return_goship_locs()
	base = ROOT_DIR+'/output/'

	for c,cs,var in list(zipper):
		print(var)
		plottable = transition_vector_to_plottable(lat_grid,lon_grid,cov.total_list,c/cs*100)
		f = interp2d(lon_grid,lat_grid,plottable)
		XX,YY,ax,fig = cartopy_setup(lat_grid,lon_grid,'')
		im = ax.pcolor(XX,YY,plottable,vmin=0,alpha=0.6)
		fig.colorbar(im,label=var1_dict[var]+' % Constrained Variance')
		for df in df_list:

				
				lats = df['Lats'].tolist()
				lats = [float(x) for x in lats]
				lons = df['Lons'].tolist()
				lons = [float(x) for x in lons]
				interp_list = [f(coord[0],coord[1])[0] for coord in zip(lons,lats)]
				cruise = df.Cruise.tolist()[0]
				cruise = cruise.replace(' ','')
				print(cruise)
				try:
						data_dict[cruise][var] = interp_list
						data_dict[cruise]['lon'] = lons
						data_dict[cruise]['lat'] = lats
				except KeyError:
						data_dict[cruise] = {}
						data_dict[cruise][var] = interp_list      
						data_dict[cruise]['lon'] = lons
						data_dict[cruise]['lat'] = lats                      
				ax.scatter(lons,lats,zorder=15)

		# argo.scatter_plot(ax=ax)
		# soccom.scatter_plot(ax=ax)
		plt.savefig(base+var+'_all_cruises')
		plt.close()

	for cruise in data_dict.keys():
		print(cruise)
		temp_dict = data_dict[cruise]
		for var in temp_dict.keys():
				if var in ['lat','lon']:
					continue
				lat = temp_dict['lat']
				data = abs(np.array(temp_dict[var]))
				plt.plot(lat,data,label=var)
		plt.legend()
		plt.title(cruise+' Along Track Variance')
		plt.ylabel('% Constrained Variance')
		plt.xlabel('Latitude')
		plt.savefig(base+'lineplot_'+cruise)
		# plt.show()
		plt.close()



def goship_line_plot(depth_level=0):
	def calculate_and_save_p_hat(depth_level,cov):

		hinstance = HInstance.generate_from_float_class([argo,soccom],variable_list=cov.variable_list)
		hinstance = hinstance.T
		noise = scipy.sparse.diags([cov.diagonal().mean()*noise_factor]*hinstance.shape[0])
		denom = hinstance.dot(cov).dot(hinstance.T)+noise
		denom = scipy.sparse.csc_matrix(denom)
		inv_denom = scipy.sparse.linalg.inv(denom)

		cov_subtract = scipy.sparse.csc_matrix(cov).dot(hinstance.T.dot(inv_denom).dot(hinstance)).dot(scipy.sparse.csc_matrix(cov))
		cov_subtract_holder = InverseInstance(cov_subtract,shape=cov_subtract.shape,total_list=global_cov.total_list,
			lat_spacing=global_cov.degree_bins[0],lon_spacing=global_cov.degree_bins[1]
			,l='p_hat',variable_list=global_cov.variable_list,
			traj_file_type='cm4_submeso_covariance_'+str(depth_level))
		p_hat = scipy.sparse.csc_matrix(cov)-scipy.sparse.csc_matrix(cov_subtract)
		holder = InverseInstance(p_hat,shape=p_hat.shape,total_list=global_cov.total_list,
			lat_spacing=global_cov.degree_bins[0],lon_spacing=global_cov.degree_bins[1]
			,l='p_hat',variable_list=global_cov.variable_list,
			traj_file_type='cm4_submeso_covariance_'+str(depth_level))
		holder.save()
		return holder
	global_cov = InverseInstance.load_from_type(2,2,1500,'cm4_global_covariance_'+str(depth_level))
	submeso_cov = InverseInstance.load_from_type(2,2,300,'cm4_submeso_covariance_'+str(depth_level))
	cov = global_cov+submeso_cov
	try: 
		p_hat = InverseInstance.load_from_type(lat_spacing=2,lon_spacing=2,l='p_hat',traj_type='cm4_submeso_covariance_'+str(depth_level))
	except FileNotFoundError:
		p_hat = calculate_and_save_p_hat(depth_level,cov)
	cov_subtract = cov - p_hat

	def return_goship_locs():
		import pandas as pd
		import re
		base_goship_folder = ROOT_DIR+'/data/goship_lines/'
		df_list = []
		df = pd.read_csv(base_goship_folder+'p04_hy1.csv',skiprows=3,usecols=[1,10,11],names=['Cruise','Lats','Lons'])
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		df = pd.read_csv(base_goship_folder+'33AT20120324_hy1.csv',skiprows=52,usecols=[1,9,10],names=['Cruise','Lats','Lons'])
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		df = pd.read_csv(base_goship_folder+'33AT20120419_hy1.csv',skiprows=52,usecols=[1,9,10],names=['Cruise','Lats','Lons'])
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		df = pd.read_csv(base_goship_folder+'49NZ20140717_hy1.csv',skiprows=7,usecols=[1,9,10],names=['Cruise','Lats','Lons'])
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		holder = []
		for file in os.listdir(base_goship_folder+'ar07_74JC20140606_ct1'):
				open_file = open(base_goship_folder+'ar07_74JC20140606_ct1/'+file,'r')
				lat_lon = []
				for line in open_file.readlines()[10:12]:
						lat_lon.append(re.findall(r'[-+]?\d+.\d+', line)[0])
				holder.append(tuple(lat_lon))
		lats,lons = zip(*holder)
		df = pd.DataFrame({'Cruise':['AR07']*len(lats),'Lats':lats,'Lons':lons})
		df = df.dropna().drop_duplicates()
		df_list.append(df)
		return df_list
	lat_grid, lon_grid = cov.bins_generator(cov.degree_bins)
	zipper = zip(np.split(cov_subtract.diagonal(),len(cov.variable_list)),np.split(cov.diagonal(),len(cov.variable_list)),cov.variable_list)

	# var1_dict = {'salt':'Salinity (psu m)','temp': 'Temperature (C m)','dic': 'DIC ($mol\ m^{-2}$)', 
	# 'o2': 'Oxygen ($mol\ m^{-2}$)'}
	var1_dict = {'so':'Salinity (psu)','thetao': 'Temperature (C)','ph':'','dic':'DIC ($mol\ m^{-2}$)', 
	'o2': 'Oxygen ($mol\ m^{-2}$)','chl': 'Chlorophyll ($mg\ m^{-3}$)'}
	data_dict = {}
	df_list = return_goship_locs()
	base = ROOT_DIR+'/output/'

	for c,cs,var in list(zipper):
		print(var)
		plottable = transition_vector_to_plottable(lat_grid,lon_grid,cov.total_list,c/cs*100)
		f = interp2d(lon_grid,lat_grid,plottable)
		XX,YY,ax,fig = cartopy_setup(lat_grid,lon_grid,'')
		im = ax.pcolor(XX,YY,plottable,vmin=0,alpha=0.6)
		fig.colorbar(im,label=var1_dict[var]+' % Constrained Variance')
		for df in df_list:

				
				lats = df['Lats'].tolist()
				lats = [float(x) for x in lats]
				lons = df['Lons'].tolist()
				lons = [float(x) for x in lons]
				interp_list = [f(coord[0],coord[1])[0] for coord in zip(lons,lats)]
				cruise = df.Cruise.tolist()[0]
				cruise = cruise.replace(' ','')
				print(cruise)
				try:
						data_dict[cruise][var] = interp_list
						data_dict[cruise]['lon'] = lons
						data_dict[cruise]['lat'] = lats
				except KeyError:
						data_dict[cruise] = {}
						data_dict[cruise][var] = interp_list      
						data_dict[cruise]['lon'] = lons
						data_dict[cruise]['lat'] = lats                      
				ax.scatter(lons,lats,zorder=15)

		# argo.scatter_plot(ax=ax)
		# soccom.scatter_plot(ax=ax)
		plt.savefig(base+var+'_all_cruises')
		plt.close()

	for cruise in data_dict.keys():
		print(cruise)
		temp_dict = data_dict[cruise]
		for var in temp_dict.keys():
				if var in ['lat','lon']:
					continue
				lat = temp_dict['lat']
				data = abs(np.array(temp_dict[var]))
				plt.plot(lat,data,label=var)
		plt.legend()
		plt.title(cruise+' Along Track Variance')
		plt.ylabel('% Constrained Variance')
		plt.xlabel('Latitude')
		plt.savefig(base+'lineplot_'+cruise)
		# plt.show()
		plt.close()





















def regional_variance():
	depth_list = np.arange(0,21,2).tolist()
	num_list = [3,6,12,15,18,21,24,27,30,33]
	for name, region in [('GOMMEC',[-100,-81.5,20.5,30.5]),('CCS',[-135,-105,20,55])]:
		lllon,urlon,lllat,urlat = region
		

		out_list = []

		for depth_level in depth_list:


			global_cov = InverseInstance.load_from_type(2,2,1500,'cm4_global_covariance_'+str(depth_level))
			submeso_cov = InverseInstance.load_from_type(2,2,300,'cm4_submeso_covariance_'+str(depth_level))
			p_hat = global_cov+submeso_cov
			coords = [(lllon,lllat),(lllon,urlat),(urlon,urlat),(urlon,lllat),(lllon,lllat)]
			poly = Polygon(coords)
			total_truth = [Point(x[0],x[1]).within(poly) for x in p_hat.total_list]
			idx = np.where(total_truth)[0]
			total_idx = []
			for i in range(len(p_hat.variable_list)):
				total_idx+=(i*len(p_hat.total_list)+idx).tolist()

			p_hat_holder = p_hat[total_idx,:]
			p_hat_holder = p_hat_holder[:,total_idx]
			total_list = np.array(p_hat.total_list)[total_truth]

			lat_grid, lon_grid = p_hat.bins_generator(p_hat.degree_bins)
			zipper = zip(np.split(p_hat_holder.diagonal(),len(p_hat.variable_list)),p_hat.variable_list)
			scale_dict = {}
			for c,var in list(zipper):
				print(var)
				plottable = transition_vector_to_plottable(lat_grid,lon_grid,total_list,c)
				plottable[plottable<0]=0
				scale_dict[var]=plottable.sum()


			for num in num_list:
				for i in range(50):
					print('this is instance '+str(i)+' of float number '+str(num))
					hinstance = HInstance.randomly_generate(num,total_list=total_list,variable_list=p_hat.variable_list,degree_bins=p_hat.degree_bins,limit=[lllon,urlon,lllat,urlat])
					assert hinstance.data.sum()==num*len(p_hat.variable_list)

					hinstance = hinstance.T
					noise = scipy.sparse.diags([p_hat_holder.diagonal().mean()*noise_factor]*hinstance.shape[0])

					new_p_hat = p_hat_holder-p_hat_holder.dot(hinstance.T.dot(scipy.sparse.linalg.inv(hinstance.dot(p_hat_holder).dot(hinstance.T)+noise))).dot(hinstance).dot(p_hat_holder)
					zipper = zip(np.split(new_p_hat.diagonal(),len(p_hat.variable_list)),p_hat.variable_list)

					for c,var in list(zipper):
						print(var)
						plottable = transition_vector_to_plottable(lat_grid,lon_grid,total_list,c)
						plottable[plottable<0]=0
						percent_constrained = plottable.sum()
						out_list.append((percent_constrained/scale_dict[var],var,num,depth_level))
		np.save(DATA_OUTPUT_DIR+'regional_calcs/'+name,np.array(out_list))
		out_list = np.load(DATA_OUTPUT_DIR+'regional_calcs/'+name+'.npy')
		filepath = ROOT_DIR + '/../../data/cm4/thetao/thetao_Omon_GFDL-CM4_historical_r1i1p1f1_gr_185001-186912.nc'
		ncfid = Dataset(filepath)

		val,var,num,depth = zip(*out_list)
		val = [float(x) for x in val]
		num = [int(x) for x in num]
		depth = [int(x) for x in depth]

		translation_dict = {'chl':'Chl','o2':'O2','ph':'Ph','so':'Salinity','thetao':'Temperature'}
		for var_dummy in np.unique(var):

			depths_num = np.sort(np.unique(np.array(depth)[(np.array(var)==var_dummy)]))
			std_array = np.zeros([len(depths_num),len(num_list)])
			mean_array = np.zeros([len(depths_num),len(num_list)])		
			depths = ncfid['lev'][:][depths_num]
			XX,YY = np.meshgrid(num_list,depths)
			for i,depth_dummy in enumerate(depths_num):
				for k,num_dummy in enumerate(num_list):
					mask = (np.array(var)==var_dummy)&(np.array(num)==num_dummy)&(np.array(depth)==depth_dummy)
					val_list = np.array(val)[mask]
					num_idx = num_list.index(num_dummy)
					depth_idx = depth_list.index(depth_dummy)
					std_array[depth_idx,num_idx]=100-val_list.std()*100
					mean_array[depth_idx,num_idx]=100-val_list.mean()*100

			plt.pcolor(XX,YY,mean_array,vmin=0,vmax=100)
			plt.colorbar()
			plt.ylabel('Depth (m)')
			plt.xlabel('Number of floats deployed')
			plt.title('Mean Variance Constrained for '+translation_dict[var_dummy])
			plt.gca().invert_yaxis()
			plt.show()
			plt.savefig(PLOT_OUTPUT_DIR+'regional_calcs/'+name+'_'+var_dummy+'_mean')
			plt.close()
	# @staticmethod
	# def gradient_calc(data):
	# 		dy = 111.7*1000
	# 		# bins_lat_holder = self.bins_lat
	# 		# bins_lat_holder[0]=-89.5
	# 		# bins_lat_holder[-1]=89.5
	# 		# x = np.cos(np.deg2rad(bins_lat_holder))*dy #scaled meridional distance
	# 		XX,YY = np.gradient(data,dy,dy)
	# 		return XX+YY

	# @staticmethod		
	# def data_return(cls,variable):
	# 	data = np.load(ROOT_DIR+cls.file_dictionary[variable]) 
	# 	return data

	# @staticmethod
	# def translation_list_construct(degree_bins,total_list):
	# 	translation = []
	# 	lats,lons,mask = InverseBase.lat_lon_setup()
	# 	# grab only the whole degree values
	# 	for x in total_list:
	# 		mask = (lats==x[0])&(lons==x[1])      
	# 		if not mask.any():
	# 			for lat in np.arange(x[0]-degree_bins[0],x[0]+degree_bins[0],0.5):
	# 				for lon in np.arange(x[1]-degree_bins[1],x[1]+degree_bins[1],0.5):
	# 					mask = (lats==lat)&(lons==lon)
	# 					if mask.any():
	# 						t = np.where(mask)
	# 						break
	# 		else:
	# 			t = np.where(mask)
	# 		assert len(t[0])==1
	# 		translation.append(t[0][0])
	# 	return translation

	# @staticmethod
	# def lat_lon_setup():
	# 	lat_list = np.load(ROOT_DIR+'/data/lat_list.npy')
	# 	lon_list = np.load(ROOT_DIR+'/data/lon_list.npy')
	# 	lon_list[lon_list<-180]=lon_list[lon_list<-180]+360
		
	# 	mask = (lon_list%1==0)&(lat_list%1==0) # pick out whole degrees only. 
	# 	lats = lat_list[mask]
	# 	lons = lon_list[mask]
	# 	return (lats,lons,mask)





####### target correlation ###########

# class TargetCorrelation(InverseBase):
# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(TargetCorrelation,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)

# 	def plot(self,m=False):
# 		if not m:
# 			m = self.quiver_plot(arrows=False,degree_sep=6,scale_factor=.6)
# 		else:
# 			self.quiver_plot(arrows=False,degree_sep=6,m=m,scale_factor=.6)
# 		return m

# 	def scale_direction(self):
# 		mask = np.array(self.matrix.todense()!=0)
# 		self.east_west = np.multiply(mask,self.east_west)
# 		self.north_south = np.multiply(mask,self.north_south)

# 	def test(self):
# 		# assert abs(np.trace(self.matrix.todense())-self.matrix.shape[0])<.1
# 		assert (self.data<=1).all()
# 		assert (self.data>=0).all()

# 	@staticmethod
# 	def traj_type_gen(variable):
# 		return variable+'_corr'

# 	def return_signal_to_noise(self,SNR,exploration_factor):
# 		self.data[self.data<=0]=0.01 # we do this because find excludes values that are zero
# 		row_list, column_list, data_array = scipy.sparse.find(self)
# 		self.get_direction_matrix()
# 		east_west_data = self.east_west[row_list,column_list]*data_array
# 		north_south_data = self.north_south[row_list,column_list]*data_array
# 		self.east_west = self.new_sparse_matrix(east_west_data)
# 		self.north_south = self.new_sparse_matrix(north_south_data)

# 		scale = self.new_sparse_matrix(np.exp(-(self.east_west.data**2)/(exploration_factor/2.)-(self.north_south.data**2)/(exploration_factor/4.)))
# 		scale.data = scale.data*np.random.normal(scale=1/float(np.sqrt(SNR))*np.std(self.data),size=len(self.data))
# 		self.data+=scale.data
# 		self.data[self.data<=0]=0.01
# 		self.rescale()

# 	def rescale(self,checksum=10**-3):
# 		div_array = np.abs(self.max(axis=0)).data
# 		row_idx,column_idx,data = scipy.sparse.find(self)
# 		col_count = []
# 		for col in column_idx:
# 		    col_count.append(float(div_array[col]))
# 		self.data = np.array(data)/np.array(col_count)
# 		diag_idx = range(self.shape[0])
# 		self[diag_idx,diag_idx] = 1 
# 		# self.test()


# class CM2p6Correlation(TargetCorrelation):
# 	file_dictionary = 	{'surf_o2':'/data/o2_surf_corr.npy','surf_dic':'/data/dic_surf_corr.npy',\
# 		'surf_pco2':'/data/pco2_surf_corr.npy','100m_dic':'/data/dic_100m_corr.npy','100m_o2':'/data/o2_100m_corr.npy'\
# 		}	
# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(CM2p6Correlation,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)
# 		self.scale_factor=.1
# 		# corr_file_path = ROOT_DIR + '/data/correlation/cm2p6_corr_'+str(variable)+'_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])

# 		# self.test()
# 		# self.scale_direction()

# 	@classmethod
# 	def compile_corr(cls,variable,base_cls_instance):

# 		# grab only the whole degree values
# 		total_list = base_cls_instance.total_list.tolist()
# 		data = InverseBase.data_return(cls,variable)
# 		total_set = Set([tuple(x) for x in total_list])
# 		row_list = []
# 		col_list = []
# 		data_list = []
# 		translation = InverseBase.translation_list_construct(degree_bins,total_list)
# 		lats,lons,mask = InverseBase.lat_lon_setup()
# 		lats = lats[translation] #mask out non whole degrees and only grab at locations that match to the self.transition.index
# 		lons = lons[translation] 
# 		corr_list = data[translation]

# 		# test_Y,test_X = zip(*self.transition.list)
# 		for k,(base_lat,base_lon,corr) in enumerate(zip(lats,lons,corr_list)):
# 			if k % 100 ==0:
# 				print str(k/float(len(total_list)))+' done'
# 			lat_index_list = np.arange(base_lat-12,base_lat+12.1,0.5)
# 			lon_index_list = np.arange(base_lon-12,base_lon+12.1,0.5)
# 			lon_index_list[lon_index_list<-180]=lon_index_list[lon_index_list<-180]+360
# 			Y,X = np.meshgrid(lat_index_list,lon_index_list) #we construct in this way to match how the correlation matrix was made
# 			test_set = Set(zip(Y.flatten(),X.flatten()))
# 			intersection_set = total_set.intersection(test_set)
# 			location_idx = [total_list.index(list(_)) for _ in intersection_set]
# 			data_idx = [zip(Y.flatten(),X.flatten()).index(_) for _ in intersection_set]

# 			data = abs(corr.flatten()[data_idx])
# 			assert len(location_idx)==len(data)
# 			try:
# 				assert abs(data.max()-1)<10**-3
# 			except AssertionError:
# 				try:
# 					data[location_idx.index(k)]=1
# 				except ValueError:
# 					dummy = data.tolist()
# 					dummy.append(1)
# 					location_idx.append(k)
# 					data = np.array(dummy)
# 			assert data.min()>=0
# 			row_list += location_idx
# 			col_list += [k]*len(location_idx)
# 			data_list += data.tolist()
# 			assert len(row_list)==len(col_list)
# 			assert len(col_list)==len(data_list)
# 			assert len(row_list)==len(data_list)
# 		return (cls((data_list,(row_list,col_list)),shape=(len(total_list)
# 		                ,len(total_list)),total_list=np.array(total_list),time_step = base_cls_instance.time_step,number_data='na'
# 		                ,lat_spacing = degree_bins[0],lon_spacing=degree_bins[1]
# 		              	,traj_file_type=variable+'_corr'))

# class GlodapCorrelation(TargetCorrelation):
# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(GlodapCorrelation,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)
# 		self.matrix = np.exp(-(self.east_west**2)/(12.)-(self.north_south**2)/(6.)) # glodap uses a 7 degree zonal correlation and 14 degree longitudinal correlation
# 		self.matrix[self.matrix<0.3] = 0
# 		self.matrix = np.multiply(self.matrix,np.random.random(self.matrix.shape))
# 		k = range(self.matrix.shape[0])
# 		self.matrix[k,k]=1
# 		self.matrix = scipy.sparse.csc_matrix(self.matrix)
# 		self.scale_direction()
# 		self.test()

# ####### target vector #########

# class TargetVector(InverseBase):
# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(TargetVector,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)


# 	@classmethod
# 	def compile(cls,variable,lat,lon,time_step):
# 		base_cls_instance = TransMat.load_from_type(traj_type='argo',lat_spacing=lat,lon_spacing=lon,time_step=time_step)
# 		total_list = base_cls_instance.total_list.tolist()
# 		data = cls.data_compile(cls,variable,total_list,base_cls_instance.degree_bins)
# 		traj = cls.traj_type_gen(variable)
# 		row_list = np.arange(len(total_list)).tolist()
# 		col_list = [0]*len(total_list)
# 		print variable
# 		return (InverseBase((data,(row_list,col_list)),shape=(len(total_list)
# 		                ,1),total_list=np.array(total_list),time_step =time_step,number_data='na'
# 		                ,lat_spacing = lat,lon_spacing=lon
# 		              	,traj_file_type=traj))

# 	def return_signal_to_noise(self,SNR):
# 		self.data[self.data==0]=0.01 # we do this because find excludes values that are zero
# 		row_list, col_list, data_array = scipy.sparse.find(self)
# 		noise = np.random.normal(scale=1/float(np.sqrt(SNR))*np.std(data_array),size=len(data_array))
# 		signal = data_array+noise
# 		signal[signal<=0]=.01
# 		self.data = signal

# 	def plot(self,XX=False,YY=False,m=False):
# 		bins_lat,bins_lon = self.bins_generator(self.degree_bins)
# 		plot_vector = abs(transition_vector_to_plottable(bins_lat,bins_lon,self.total_list,self.data))
# 		print 'shape of plot vector is ',plot_vector.shape
# 		if not m:
# 			XX,YY,m = basemap_setup(bins_lat,bins_lon,'argo')
# 		print 'shape of the coordinate matrix is ',XX.shape 
# 		cm = self.cm_dict[self.base_variable_generator()]
# 		vmin,vmax = self.plot_range_dict[self.traj_file_type.tolist()]
# 		m.pcolormesh(XX,YY,np.ma.masked_equal(plot_vector,0),cmap=cm,vmin=vmin,vmax=vmax)
# 		plt.colorbar(label=self.unit)
# 		plt.title(self.title)
# 		return (XX,YY,m)



# class LandschutzerCO2Flux(TargetVector):
# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(LandschutzerCO2Flux,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)
# 		self.cm = plt.cm.PRGn

# 		self.label = 'CO2 Flux $gm C/m^2/yr$'
# 		if var:
# 			file_path = './data/landschutzer_vector_time_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
# 		else:
# 			file_path = './data/landschutzer_vector_space_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
# 		try:
# 			self.vector = self.load_vector(file_path)
# 		except IOError:
# 			print 'landchutzer target vector file not found, recompiling'
# 			self.vector = self.compile_vector(var,file_path)

# 	def compile_vector(self,var,save_file):

# 		file_ = self.base_file+'../spco2_MPI_SOM-FFN_v2018.nc'
# 		nc_fid = Dataset(file_)
# 		y = nc_fid['lat'][:]
# 		x = nc_fid['lon'][:]
		
# 		data = np.ma.masked_greater(nc_fid['fgco2_smoothed'][:],10**19) 
# 		if var:
# 			data = np.nanvar(data,axis=0)
# 		else:
# 			XX,YY = np.gradient(np.nanmean(data,axis=0)) # take the time mean, then take the gradient of the 2d array
# 			data = np.abs(XX+YY)
# 		x[0] = -180
# 		x[-1] = 180
# 		vector = plottable_to_transition_vector(y,x,self.list,data)
# 		self.save()
# 		return vector

# class MODISVector(TargetVector):
# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(MODISVector,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)
# 		self.cm = plt.cm.PRGn
# 		self.vmin = None
# 		self.vmax = None
# 		self.unit = '$mg m^-3$'
# 		self.title = None
# 		if var=='time':
# 			self.label = '*** MODIS TIME VARIANCE ***'
# 		else:
# 			self.label = '*** MODIS SPACE VARIANCE ***'			
# 		file_path = './data/modis_vector_'+str(var)+'_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
# 		try:
# 			self.vector = self.load_vector(file_path)
# 		except IOError:
# 			print 'modis file not found, recompiling'
# 			self.vector = self.compile_vector(var,file_path)

# 	def compile_vector(self,var,save_file):
# 		file_ = self.base_file+'../MODIS/'
# 		datalist = []
# 		for _ in os.listdir(file_):
# 			if _ == '.DS_Store':
# 				continue
# 			if _ == 'array_processed.pkl':
# 				continue
# 			nc_fid = Dataset(file_ + _)
# 			datalist.append(nc_fid.variables['chlor_a'][::12,::12])
# 		y = nc_fid.variables['lat'][::12]
# 		x = nc_fid.variables['lon'][::12]
# 		dat = np.ma.stack(datalist)

# 		if var=='time':
# 			data = np.ma.var(dat,axis=0)
# 		else:		
# 			dummy = np.ma.mean(dat,axis=0)
# 			data = self.gradient_calc(dummy)

# 		vector = plottable_to_transition_vector(self.bins_lat,self.bins_lon,self.list,data)
# 		np.save(save_file,vector)
# 		return vector

# class GlodapVector(TargetVector):
# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(GlodapVector,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)
# 		file_path = './data/glodap_vector_'+str(flux)+'_'+str(self.degree_bins[0])+'_'+str(self.degree_bins[1])
# 		try:
# 			self.vector = self.load_vector(file_path)
# 		except IOError:
# 			print 'glodap file not found, recompiling'
# 			self.vector = self.compile_vector(flux,file_path)

# 	def compile_vector(self,flux,save_file):
# 		file_ = self.base_file+'../GLODAP/'
# 		for _ in os.listdir(file_):
# 			variable = _.split('.')[-2]
# 			if variable in [flux]:
# 				nc_fid = Dataset(file_ + _)
# 				data = nc_fid[variable][0,:,:]
# 				data = self.gradient_calc(data)
# 		y = nc_fid['lat'][:]
# 		x = nc_fid['lon'][:]
# 		x[x>180] = x[x>180]-360
# 		x[159] = 180
# 		x[160] = -180
# 		datalist = [data[:,_] for _ in range(data.shape[1])]
# 		sorted_data = [_ for __,_ in sorted(zip(x,datalist))]
# 		sorted_data = np.ma.stack(sorted_data).T
# 		sorted_x = sorted(x)
# 		print 'I am working on ',flux
# 		vector = plottable_to_transition_vector(y,sorted_x,self.list,sorted_data)
# 		vector[np.where(np.isnan(vector))] = max(vector)
# 		assert ~np.isnan(vector).any()
# 		np.save(save_file,vector)
# 		return vector

# class CM2p6Vector(TargetVector):
# 	file_dictionary = 	{'surf_o2':'/data/subsampled_o2_surf.npy','surf_dic':'/data/subsampled_dic_surf.npy',\
# 		'surf_pco2':'/data/subsampled_pco2_surf.npy','100m_dic':'/data/subsampled_dic_100m.npy','100m_o2':'/data/subsampled_o2_100m.npy'\
# 		}	
# 	
# 		}	
# 	cm_dict = {'surf_o2':cm.BrBG,'surf_dic':cm.PRGn,\
# 		'surf_pco2':cm.PiYG,'100m_dic':cm.PRGn,'100m_o2':cm.BrBG\
# 		}	
# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(CM2p6Vector,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)
# 		self.title = ''


# 	@staticmethod
# 	def data_compile(cls,variable,total_list,degree_bins):
# 		data = InverseBase.data_return(cls,variable)
# 		lat,lon,mask = InverseBase.lat_lon_setup()
# 		data = data[:,mask]
# 		data = cls.data_transform(cls,data,total_list,degree_bins)
# 		return data

# 	def base_variable_generator(self):
# 		basename = '_'.join(self.traj_file_type.tolist().split('_')[0:2])
# 		return basename

# class CM2p6VectorSpatialGradient(CM2p6Vector):	
# 	plot_range_dict = {'surf_o2_spatial_gradient_vector':(0,10**-8),'surf_dic_spatial_gradient_vector':(10**-5,10**-4),\
# 	'surf_pco2_spatial_gradient_vector':(10**-4,10**-5),'100m_dic_spatial_gradient_vector':(2*10**-5,5*10**-6),\
# 	'100m_o2_spatial_gradient_vector':(2*10**-5,10**-6)\
# 	}	

# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(CM2p6VectorSpatialGradient,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)
# 		basename = '_'.join(traj_file_type.tolist().split('_')[0:2])
# 		self.unit = self.unit_dict[self.base_variable_generator()]+' m$^{-1}$'

# 	@staticmethod
# 	def data_transform(cls,data,total_list,degree_bins):
# 		bins_lat,bins_lon = InverseBase.bins_generator(degree_bins)
# 		translation_list = InverseBase.translation_list_construct(degree_bins,total_list)
# 		data = transition_vector_to_plottable(bins_lat,bins_lon,total_list,data.mean(axis=0)[translation_list])
# 		data = cls.gradient_calc(data)
# 		data = plottable_to_transition_vector(bins_lat,bins_lon,total_list,data)
# 		return data

# 	@staticmethod
# 	def traj_type_gen(variable):
# 		return variable+'_spatial_gradient_vector'

# class CM2p6VectorTemporalVariance(CM2p6Vector):	
# 	plot_range_dict = {'surf_o2_temporal_variance_vector':(10**-14,10**-10),'surf_dic_temporal_variance_vector':(10**-17,10**-15),\
# 	'surf_pco2_temporal_variance_vector':(200,600),'100m_dic_temporal_variance_vector':(.1,5),
# 	'100m_o2_temporal_variance_vector':(.1,2)\
# 	}	

# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):

# 		super(CM2p6VectorTemporalVariance,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)

# 		self.unit = '('+self.unit_dict[self.base_variable_generator()]+')$^2$'

# 	@staticmethod
# 	def data_transform(cls,data,total_list,degree_bins):
# 		translation_list = InverseBase.translation_list_construct(degree_bins,total_list)
# 		N = 90
# 		data = data.T[translation_list].T
# 		data = [np.var(np.convolve(x, np.ones((N,))/N, mode='valid')) for x in data.T]
# 		return data

# 	@staticmethod
# 	def traj_type_gen(variable):
# 		return variable+'_temporal_variance_vector'

# class CM2p6VectorMean(CM2p6Vector):	
# 	plot_range_dict = {'surf_o2_mean_vector':(.00010,.00037),'surf_dic_mean_vector':(0,10**-7),\
# 	'surf_pco2_mean_vector':(200,400),'100m_dic_mean_vector':(195,225),'100m_o2_mean_vector':(15,35)}	
# 	def __init__(self, arg1, shape=None,total_list=None,lat_spacing=None,lon_spacing=None
# 		,time_step=None,number_data=None,traj_file_type=None):
# 		super(CM2p6VectorMean,self).__init__(arg1, shape=shape,total_list=total_list,lat_spacing=lat_spacing,lon_spacing=lon_spacing
# 		,time_step=time_step,number_data=number_data,traj_file_type=traj_file_type)
# 		basename = '_'.join(traj_file_type.tolist().split('_')[0:2])
# 		self.unit = self.unit_dict[self.base_variable_generator()]

# 	@staticmethod
# 	def data_transform(cls,data,total_list,degree_bins):
# 		translation_list = InverseBase.translation_list_construct(degree_bins,total_list)
# 		data = data.mean(axis=0)[translation_list]
# 		return data

# 	@staticmethod
# 	def traj_type_gen(variable):
# 		return variable+'_mean_vector'


# def plot_all_correlations():
# 	trans_plot = TransitionPlot()
# 	trans_plot.get_direction_matrix()
# 	for name,corr_class in [('CM2p6',CM2p6Correlation)]:
# 		for variable in ['o2','pco2','hybrid']:
# 			try:
# 				dummy_class = corr_class(transition_plot=trans_plot,variable=variable)
# 				m = dummy_class.plot()
# 				plt.title(name+' '+variable+' '+'correlation')
# 				plt.savefig(ROOT_DIR + '/plots/'+name+'_'+variable+'_'+'correlation')
# 				plt.close()
# 			except AttributeError:
# 				pass