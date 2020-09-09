from transition_matrix_compute import TransMatrix
import json 
import copy


for span_limit in [120]:
	traj_class = TransMatrix(date_span_limit=span_limit)
	matrix = traj_class.transition_matrix
	trans_matrix = copy.deepcopy(matrix)
	for _ in range(16):
		try:
			os.mkdir(str(_))
		except OSError:
			pass
		traj_class.save_trans_matrix_to_json(str(_)
		traj_class.transition_matrix = matrix.dot(traj_class.transition_matrix)
