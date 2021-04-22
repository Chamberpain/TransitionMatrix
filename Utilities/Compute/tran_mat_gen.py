from __future__ import print_function
from data_save_utilities.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list
# from data_save_utilities.lagrangian.AOML.aoml_read import AOMLRead,aggregate_aoml_list
from data_save_utilities.lagrangian.SOSE.SOSE_read import SOSEReader,aggregate_sose_list
# from data_save_utilities.lagrangian.ocean_parcels.OneDJet.trans_read import OneDJet
# from data_save_utilities.lagrangian.ocean_parcels.OneD.trans_read import OneD
# from data_save_utilities.lagrangian.ocean_parcels.OneDDiffusion.trans_read import OneDDiffusion
# from data_save_utilities.lagrangian.ocean_parcels.OneDDoubleJet.trans_read import OneDDoubleJet
# from data_save_utilities.lagrangian.ocean_parcels.OneDDoubleJetDiffusion.trans_read import OneDDoubleJetDiffusion



from transition_matrix.compute.trans_read import TransitionClassAgg

# gen_list = [(OneDDiffusion,OneDDiffusion.aggregate_OP_list),
# 			(OneDJet,OneDJet.aggregate_OP_list),(OneDDoubleJet,OneDDoubleJet.aggregate_OP_list),
# 			(OneDDoubleJetDiffusion,OneDDoubleJetDiffusion.aggregate_OP_list)]

gen_list = [(ArgoReader,aggregate_argo_list),(SOSEReader,aggregate_sose_list)]
lat_lon_spacing_list = [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]
time_step_list = [30,60,90,120,140,160,180]

for read_class,gen_holder in gen_list:
	looper = True
	gen = gen_holder()
	while looper:
		try:
			dummy = gen.__next__()
		except StopIteration:
			looper = False
	# gps_dict = {key:val for key, val in ArgoReader.all_dict.items() if val.meta.positioning_system == 'GPS'}
	# argos_dict = {key:val for key, val in ArgoReader.all_dict.items() if val.meta.positioning_system == 'ARGOS'}


	for lat_spacing,lon_spacing in lat_lon_spacing_list:
		for time_delta in time_step_list:
			# read_class.all_dict = gps_dict
			# read_class.data_description = 'argo_gps'
			TransitionClassAgg(lat_spacing=lat_spacing,lon_spacing=lon_spacing,time_step=time_delta,drifter_class=read_class,eig_vecs_flag=True)

			# read_class.all_dict = argos_dict
			# read_class.data_description = 'argo_argos'
			TransitionClassAgg(lat_spacing=lat_spacing,lon_spacing=lon_spacing,time_step=time_delta,drifter_class=read_class,eig_vecs_flag=True)


from transition_matrix.makeplots.transition_matrix_plot import TransPlot
gen_list = ['argo','SOSE']
lat_lon_spacing_list = [(1,1),(2,2),(3,3),(4,4),(2,3),(3,6)]
time_step_list = [30,60,90,120,140,160,180]
for traj_type in gen_list:
	for lat,lon in lat_lon_spacing_list:
		for time_step in time_step_list:
			holder = TransPlot.load_from_type(lon,lat,time_step,traj_type)
			holder.abs_mean_and_dispersion()


k = 30
dummy_list = TransitionClassAggPrep(ArgoReader)

percentage_list = [0.5,0.6,0.7,0.8,0.9]
for percentage in percentage_list:
	for i in range(20,30):
		holder = TransitionClassAgg(ArgoReader,dummy_list,percentage=percentage)
		filename = holder.make_filename(traj_type=holder.traj_file_type,degree_bins=holder.degree_bins,time_step = holder.time_step)
		filename = os.path.dirname(filename)+'/'+str(percentage).replace('.','_')+'-'+str(i)+'.npz'
		holder.save(filename=filename)