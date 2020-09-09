from transition_matrix.makeplots.transition_matrix_plot import TransPlot
import pandas as pd 
from transition_matrix.makeplots.argo_data import Float
from transition_matrix.makeplots.plot_utils import basemap_setup,transition_vector_to_plottable
import numpy as np
import matplotlib.pyplot as plt
import datetime
matrix_path = '/Users/pchamberlain/Projects/transition_matrix/output/aoml/60-[2.0, 2.0].npz'
sofar_df = pd.read_csv('/Users/pchamberlain/Projects/sofar_oceanography_colab/Data/Spotter_track_data.csv',index_col=0)
fmt = '%Y-%m-%dT%H:%M:%S.000Z'

df_list = []
for spot in sofar_df.spotterId.unique():
	print spot
	df_holder = sofar_df[sofar_df.spotterId==spot]
	df_holder['timestamp'] = [datetime.datetime.strptime(dummy,fmt) for dummy in df_holder['timestamp'].tolist()]
	print df_holder.timestamp.min()
	df_holder = df_holder[(df_holder['timestamp']>datetime.datetime(2019,11,1))&(df_holder['timestamp']<datetime.datetime(2019,11,2))]
	dummy = df_holder.tail(1)
	df_list.append(dummy)
df = pd.concat(df_list)
df['Lat'] = df.latitude
df['Lon'] = df.longitude
df['Age'] = 1
trans_mat = TransPlot.load(matrix_path)
float_array = Float(trans_mat)
float_array.df = df
float_array.reshape_float_vector(age_return=False)
plottable = transition_vector_to_plottable(trans_mat.bins_lat,trans_mat.bins_lon,trans_mat.total_list.tolist(),trans_mat.dot(float_array.vector))
XX,YY,m = basemap_setup(trans_mat.bins_lat,trans_mat.bins_lon,trans_mat.traj_file_type)  
m.pcolormesh(XX,YY,np.ma.masked_equal(plottable,0),cmap=plt.cm.magma)
plt.colorbar(label='Chance of Spotter')
plt.title('Spotter Array on Jan 1,2020')
float_array.plot(m=m)
plt.savefig('spotter_prediciton')
np.save('jan1_prob',plottable)
