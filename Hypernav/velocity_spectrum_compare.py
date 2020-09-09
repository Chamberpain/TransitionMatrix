from data_save_utilities.lagrangian.AOML.aoml_read import AOMLRead,aggregate_aoml_list
from data_save_utilities.lagrangian.argo.wetlabs_read import WetLabsRead,aggregate_wetlabs_list
from data_save_utilities.lagrangian.argo.argo_read import ArgoReader,aggregate_argo_list
from data_save_utilities.lagrangian.drifter_base_class import all_drift_holder
from sets import Set
import matplotlib.pyplot as plt
import numpy as np 


wetlabs_gen = aggregate_wetlabs_list()
loop_val = True
while loop_val:
	try:
		print loop_val
		drift_holder = next(wetlabs_gen)
	except StopIteration:
		loop_val = False

all_wet = all_drift_holder(WetLabsRead,lat_spacing=1,lon_spacing=1)
wet_bin_list = all_wet.get_full_bins()
wet_bin_list = list(Set(wet_bin_list))
wet_bin_list = [list(_) for _ in wet_bin_list]
wet_speed_list = drift_holder.get_full_speed_list()

lat_bins = np.arange(-90,90,1)
lon_bins = np.arange(-180,180,1)

gen_list = [aggregate_aoml_list,aggregate_argo_list]
return_list = []
for generator in gen_list:
	gen = generator()
	partial_speed_list = []
	loop_val = True
	while loop_val:
		try:
			drift_holder = next(gen)
			print drift_holder.meta.id
			if drift_holder.is_problem():
				continue
			bin_list = drift_holder.prof.pos.return_pos_bins(lat_bins,lon_bins,index_return=False)
			bin_list = [list(_) for _ in bin_list]
			mask = (np.array(bin_list)[:,None] == wet_bin_list).all(2).any(1)			
			partial_speed_list+=np.array(drift_holder.prof.speed._list)[mask[:-1]].tolist()
		except StopIteration:
			loop_val=False
	return_list.append(partial_speed_list)

bins = np.arange(0,0.5,0.01)
plt.hist(return_list[1],bins=bins,label='Argo',normed=True,alpha=0.2)
plt.hist(return_list[0],bins=bins,label='AOML',normed=True,alpha=0.2)
plt.hist(wet_speed_list,bins=bins,label='Wetlabs',normed=True,alpha=0.2)
plt.title('PDF of Instrument Speeds')
plt.xlabel('Speed (kts)')
plt.ylabel('Probability')
plt.legend()
plt.savefig('velocity_pdf')
plt.show()