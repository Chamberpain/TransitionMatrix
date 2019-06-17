import datetime
import numpy as np
import os
import fnmatch
import time # used for expressing how long each routine lasted.

def wrap_lon360(lon):
    lon = np.atleast_1d(lon).copy()
    positive = lon > 0
    lon = lon % 360
    lon[np.logical_and(lon == 0, positive)] = 360
    return lon

def wrap_lon180(lon): #stolen from oceans module, not up to date
    lon = np.atleast_1d(lon).copy()
    angles = np.logical_or((lon < -180), (180 < lon))
    lon[angles] = wrap_lon360(lon[angles] + 180) - 180
    return lon

def loop_data(data_directory,fmt,function):
	frames = []
	matches = []
	for root, dirnames, filenames in os.walk(data_directory):
		for filename in fnmatch.filter(filenames,fmt):
			matches.append(os.path.join(root, filename))
	for n, match in enumerate(matches):
		print 'file is ',match,', there are ',len(matches[:])-n,'floats left'
		t = time.time()
		frames.append(function(match))
		print 'Building and merging datasets took ', time.time()-t
	return frames

def time_parser(juld_list,ref_date = datetime.datetime(1950,1,1,1,1)):
	return [ref_date + datetime.timedelta(days=x) for x in juld_list]
