from transition_matrix.makeplots.inversion.target_load import CM2p6Correlation
import matplotlib.pyplot as plt
import numpy as np 
lat_lon_spacing_list = [(2,2)]
variable_list = ['surf_o2','surf_pco2','surf_dic','100m_o2','100m_dic']

for lat,lon in lat_lon_spacing_list[::-1]:
	for variable in variable_list:
		for cor_class in [CM2p6Correlation]:
			print variable
			print cor_class
			traj = cor_class.traj_type_gen(variable)
			cor = cor_class.load_from_type(traj_type=traj,lat_spacing=lat,lon_spacing=lon,time_step=60)
			cor.plot()
			traj_type = cor_class.traj_type_gen(variable)
			degree_bins = np.array([lat,lon])
			time_step=60
			filename = cor_class.make_filename(traj_type,degree_bins,time_step)
			filename = filename[:-4]+'.png'
			plt.savefig(filename)
			plt.close()