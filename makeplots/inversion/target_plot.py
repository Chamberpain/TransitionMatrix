from transition_matrix.makeplots.inversion.target_load import CM2p6VectorSpatialGradient,CM2p6VectorTemporalVariance,CM2p6VectorMean
import matplotlib.pyplot as plt

lat_lon_spacing_list = [(2,2)]
variable_list = ['surf_o2','surf_pco2','surf_dic','100m_o2','100m_dic']

for lat,lon in lat_lon_spacing_list[::-1]:
	for variable in variable_list[:1]:
		for target_class in [CM2p6VectorMean]:
			print variable
			print target_class
			traj = target_class.traj_type_gen(variable)
			target = target_class.load_from_type(traj_type=traj,lat_spacing=lat,lon_spacing=lon,time_step=60)
			target.plot()
			traj_type = target_class.traj_type_gen(variable)
			degree_bins = np.array([lat,lon])
			time_step=60
			filename = target_class.make_filename(traj_type,degree_bins,time_step)
			filename = filename[:-4]+'.png'
			plt.savefig(filename)
			plt.close()