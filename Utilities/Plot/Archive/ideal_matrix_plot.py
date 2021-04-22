
from scipy.linalg import circulant
import numpy as np 
import matplotlib.pyplot as plt
import os


for noise in [0,0.05,0.1,0.15,0.2,0.25]:
	k = 100
	diffusivity = 0.1
	advection = 0.3



	# dif_mat = toeplitz(col,col)

	col = np.zeros([k,1])
	col[0] = 1
	col[1] = 0
	col[2] = 0.5
	col[3] = 1

	file_name = str(noise).replace('.','dot')+'_'+str(diffusivity).replace('.','dot')+'_'+str(col[0][0]).replace('.','dot')+'_'+str(col[1][0]).replace('.','dot')+'_'+str(col[2][0]).replace('.','dot')+'_'+str(col[4][0]).replace('.','dot')

	mat = circulant(col)
	mat[mat>0] = [dummy+np.random.normal(scale = noise) for dummy in mat[mat>0].flatten()]   
	mat = mat/mat.sum(axis=0)



	def plot(mat_token,i):
		evals, evecs = np.linalg.eig(mat_token)
		u,s,v = np.linalg.svd(mat_token)
		evecs = [evecs[:,x] for _,x in sorted(zip(evals,range(len(evals))))]
		evals = sorted(evals)
		plt.subplot(4,1,1)
		for evec_token in evecs[-4:]:
			plt.plot(evec_token)
		plt.subplot(4,1,2)
		plt.plot(sorted(evals))
		plt.yscale('log')
		plt.subplot(4,1,3)
		plt.pcolormesh(mat_token)
		plt.gca().invert_yaxis()
		plt.colorbar()
		plt.subplot(4,1,4)
		plt.plot(mat_token[:,0])
		plt.suptitle('noise = '+str(noise))
		try:
			plt.savefig('./'+file_name+'/'+str(i/10))
		except IOError:
			os.mkdir(file_name)
		plt.close()

	base_mat = np.diag(np.ones(mat.shape[0]))
	for i in range(1501):
		base_mat = base_mat.dot(mat)
		if (i%10)==0:
			plot(base_mat,i)