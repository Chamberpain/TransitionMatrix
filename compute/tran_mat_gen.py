from data_save_utilities.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list
from data_save_utilities.lagrangian.AOML.aoml_read import AOMLRead,aggregate_aoml_list
from transition_matrix.compute.trans_read import TransitionClassAgg

gen_list = [(ArgoReader,aggregate_argo_list)]
lat_lon_spacing_list = [(1,1),(1,2),(2,2),(2,3),(4,4),(4,6)]
time_step_list = [10,20,30,40,50,60,70,80,90,100,120,140,160,180]

for read_class,gen_holder in gen_list:
	looper = True
	gen = gen_holder()
	while looper:
		try:
			dummy = gen.next()
		except StopIteration:
			looper = False
	for lat_spacing,lon_spacing in lat_lon_spacing_list:
		for time_delta in time_step_list:
			TransitionClassAgg(lat_spacing=lat_spacing,lon_spacing=lon_spacing,time_step=time_delta,drifter_class=dummy)