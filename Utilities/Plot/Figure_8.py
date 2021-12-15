def figure_7():
	trans_geo = TransPlot.load_from_type(GeoClass=TransitionGeo,lat_spacing = 2,lon_spacing = 2,time_step = 90)
	trans_geo.quiver_plot()