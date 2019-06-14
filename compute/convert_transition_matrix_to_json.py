from transition_matrix_compute import argo_traj_data
import json 

for span_limit in [60,140]:
	traj_class = argo_traj_data(degree_bins=2,date_span_limit=span_limit)
	for column in range(traj_class.transition_matrix.shape[1]):
		print 'there are ',(traj_class.transition_matrix.shape[0]-column),'columns remaining'
		p_lat,p_lon = tuple(traj_class.total_list[column])
		data = traj_class.transition_matrix[:,column].data
		lat,lon = zip(*[tuple(traj_class.total_list[x]) for x in traj_class.transition_matrix[:,column].indices.tolist()])
		feature_list = zip(lat,lon,data)
		geojson = {'type':'FeatureCollection', 'features':[]}
		for token in feature_list:
			lat,lon,prob = token
			feature = {'type':'Feature',
						'properties':{},
						'geometry':{'type':'Point',
						'coordinates':[]}}
			feature['geometry']['coordinates'] = [lon,lat]
			feature['properties']['Probability'] = prob
			geojson['features'].append(feature)
		output_filename = './'+str(span_limit)+'_day/'+str(span_limit)+'_lat_'+str(p_lat)+'_lon_'+str(p_lon)+'.js'
		with open(output_filename,'wb') as output_file:
			output_file.write('var dataset = ')
			json.dump(geojson, output_file, indent=2) 