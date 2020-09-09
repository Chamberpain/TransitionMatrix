from data_save_utilities.lagrangian.AOML.aoml_read import AOMLRead,aggregate_aoml_list
from data_save_utilities.lagrangian.spotter.spotter_read import SpotRead,aggregate_spotter_list
from data_save_utilities.lagrangian.drifter_base_class import all_drift_holder
from sets import Set
import matplotlib.pyplot as plt

aggregate_spotter_list()
all_spot = all_drift_holder(SpotRead)
spot_bin_list = all_spot.get_full_bins()
spot_bin_list = list(Set(spot_bin_list))
spot_bin_list = [list(_) for _ in spot_bin_list]
spot_speed_list = all_spot.get_full_speed_list()
partial_speed_list = []
lat_bins = np.arange(-90,90,2)
lon_bins = np.arange(-180,180,2)
aoml_gen = aggregate_aoml_list()
while True:
	drift_holder = next(aoml_gen)
	print drift_holder.meta.id
	if drift_holder.is_problem():
		continue
	bin_list = drift_holder.prof.pos.return_pos_bins(lat_bins,lon_bins,index_return=False)
	bin_list = [list(_) for _ in bin_list]
	mask = (np.array(bin_list)[:,None] == spot_bin_list).all(2).any(1)			
	partial_speed_list+=np.array(drift_holder.prof.speed._list)[mask[:-1]].tolist()

bins = np.arange(0,2.5,0.05)
plt.hist(partial_speed_list,bins=bins,label='AOML',normed=True,alpha=0.2)
plt.hist(spot_speed_list,bins=bins,label='Spotter',normed=True,alpha=0.2)
plt.title('PDF of Instrument Speeds')
plt.legend()
plt.show()