from transition_matrix.datasave.cm2p6_cor_calculation import CovCM4


for depth in [0,2,4,6,8,10,12,14,16,18,20]:
	# goship_line_plot(depth_level=depth)
	dummy = CovCM4(depth_idx = depth)
	try:
		dummy.scale_cov()
	except FileNotFoundError:
		dummy.calculate_cov()
		dummy.scale_cov()
	dummy.save()