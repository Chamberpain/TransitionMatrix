import numpy as np 
import scipy.sparse
import geopy
from TransitionMatrix.Utilities.TransMat import TransMat
import matplotlib.collections as mcoll
import matplotlib.pyplot as plt
import matplotlib
import cartopy.crs as ccrs
import numpy.ma as ma


def shiftgrid(lon0,datain,lonsin,start=True,cyclic=360.0):
    """
    Shift global lat/lon grid east or west.
    .. tabularcolumns:: |l|L|
    ==============   ====================================================
    Arguments        Description
    ==============   ====================================================
    lon0             starting longitude for shifted grid
                     (ending longitude if start=False). lon0 must be on
                     input grid (within the range of lonsin).
    datain           original data with longitude the right-most
                     dimension.
    lonsin           original longitudes.
    ==============   ====================================================
    .. tabularcolumns:: |l|L|
    ==============   ====================================================
    Keywords         Description
    ==============   ====================================================
    start            if True, lon0 represents the starting longitude
                     of the new grid. if False, lon0 is the ending
                     longitude. Default True.
    cyclic           width of periodic domain (default 360)
    ==============   ====================================================
    returns ``dataout,lonsout`` (data and longitudes on shifted grid).
    """
    if np.fabs(lonsin[-1]-lonsin[0]-cyclic) > 1.e-4:
        # Use all data instead of raise ValueError, 'cyclic point not included'
        start_idx = 0
    else:
        # If cyclic, remove the duplicate point
        start_idx = 1
    if lon0 < lonsin[0] or lon0 > lonsin[-1]:
        raise ValueError('lon0 outside of range of lonsin')
    i0 = np.argmin(np.fabs(lonsin-lon0))
    i0_shift = len(lonsin)-i0
    if ma.isMA(datain):
        dataout  = ma.zeros(datain.shape,datain.dtype)
    else:
        dataout  = np.zeros(datain.shape,datain.dtype)
    if ma.isMA(lonsin):
        lonsout = ma.zeros(lonsin.shape,lonsin.dtype)
    else:
        lonsout = np.zeros(lonsin.shape,lonsin.dtype)
    if start:
        lonsout[0:i0_shift] = lonsin[i0:]
    else:
        lonsout[0:i0_shift] = lonsin[i0:]-cyclic
    dataout[...,0:i0_shift] = datain[...,i0:]
    if start:
        lonsout[i0_shift:] = lonsin[start_idx:i0+start_idx]+cyclic
    else:
        lonsout[i0_shift:] = lonsin[start_idx:i0+start_idx]
    dataout[...,i0_shift:] = datain[...,start_idx:i0+start_idx]
    return dataout,lonsout


def get_cmap():
    norm = matplotlib.colors.Normalize(0,400/256.)
    colors = [[norm(0), "yellow"],
              [norm(10/256.), "lightgoldenrodyellow"],
              [norm(30/256.), "lightyellow"],
              [norm(50/256.), "powderblue"],
              [norm(75/256.), "skyblue"],
              [norm(100/256.), "deepskyblue"],
              [norm(400/256.), "dodgerblue"],
              ]
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", colors)
    return cmap


def colorline(
        x, y, axes, z=None, cmap='plasma', norm=plt.Normalize(0.0, 1080.0),
        linewidth=1.5, alpha=1.0,):

    def make_segments(x, y):
        points = np.array([x, y]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        return segments

    if z is None:
        z = np.linspace(0.0, 900.0, len(x))
    if not hasattr(z, "__iter__"):
        z = np.array([z])
    z = np.asarray(z)
    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha,transform=ccrs.PlateCarree())
    axes = plt.gca()
    axes.add_collection(lc)
    return lc

def matrix_difference_compare(matrix_1,matrix_2):
    # eig_vals,l_eig_vecs,r_eig_vecs = scipy.linalg.eig(matrix_1.todense(),left=True)
    # test_eig_vals,test_l_eig_vecs,test_r_eig_vecs = scipy.linalg.eig(matrix_2.todense(),left=True)
    # q = zip(eig_vals,test_eig_vals)

    l2_residual = (matrix_1-matrix_2).data*(matrix_1-matrix_2).data
    traj_1 = matrix_1.trans_geo
    traj_2 = matrix_2.trans_geo
    return ((traj_1.lat_sep,traj_1.lon_sep,traj_1.time_step,traj_1.file_type),
        (traj_2.lat_sep,traj_2.lon_sep,traj_2.time_step,traj_2.file_type),l2_residual.mean(),l2_residual.std())

def matrix_size_match(matrix_1,matrix_2):
    matrix_1_set = set(matrix_1.trans_geo.total_list.tuple_total_list())
    matrix_2_set = set(matrix_2.trans_geo.total_list.tuple_total_list())
    new_index_list = [geopy.Point(x[0],x[1]) for x in list(matrix_1_set.intersection(matrix_2_set))]

    new_matrix_1 = matrix_1.new_coordinate_list(new_index_list)
    new_matrix_2 = matrix_2.new_coordinate_list(new_index_list)

    return(new_matrix_1, new_matrix_2)

def z_test(self,p_1,p_2,n_1,n_2):
    p_1 = np.ma.array(p_1,mask = (n_1==0))
    n_1 = np.ma.array(n_1,mask = (n_1==0))
    p_2 = np.ma.array(p_2,mask = (n_2==0))
    n_2 = np.ma.array(n_2,mask = (n_2==0))      
    z_stat = (p_1-p_2)/np.sqrt(self.transition_matrix.todense()*(1-self.transition_matrix.todense())*(1/n_1+1/n_2))
    assert (np.abs(z_stat)<1.96).data.all()


def figure_20_matrix_compare(matrix_1,matrix_2):
    east_west_lr, north_south_lr = matrix_1.return_mean()
    east_west_lr = matrix_1.trans_geo.transition_vector_to_plottable(east_west_lr)
    north_south_lr = matrix_1.trans_geo.transition_vector_to_plottable(north_south_lr)

    east_west_hr, north_south_hr = matrix_2.return_mean()
    east_west_hr = matrix_2.trans_geo.transition_vector_to_plottable(east_west_hr)
    north_south_hr = matrix_2.trans_geo.transition_vector_to_plottable(north_south_hr)

    ew_std_lr, ns_std_lr = matrix_1.return_std()
    ew_std_lr = matrix_1.trans_geo.transition_vector_to_plottable(ew_std_lr)
    ns_std_lr = matrix_1.trans_geo.transition_vector_to_plottable(ns_std_lr)

    ew_std_hr, ns_std_hr = matrix_2.return_std()
    ew_std_hr = matrix_2.trans_geo.transition_vector_to_plottable(ew_std_hr)
    ns_std_hr = matrix_2.trans_geo.transition_vector_to_plottable(ns_std_hr)

    ew_mean_diff = (east_west_lr-east_west_hr)
    ns_mean_diff = (north_south_lr-north_south_hr)
    ew_std_diff = (ew_std_lr-ew_std_hr)
    ns_std_diff = (ns_std_lr-ns_std_hr)
    return (ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff)


def matrix_compare(matrix_1,matrix_2,description):
    east_west_lr, north_south_lr = matrix_1.return_mean()
    east_west_lr = matrix_1.trans_geo.transition_vector_to_plottable(east_west_lr)
    north_south_lr = matrix_1.trans_geo.transition_vector_to_plottable(north_south_lr)

    east_west_hr, north_south_hr = matrix_2.return_mean()
    east_west_hr = matrix_2.trans_geo.transition_vector_to_plottable(east_west_hr)
    north_south_hr = matrix_2.trans_geo.transition_vector_to_plottable(north_south_hr)

    mask = (~north_south_lr.mask)&(~north_south_hr.mask)

    east_west_lr = east_west_lr.data[mask]
    north_south_lr = north_south_lr.data[mask]
    east_west_hr = east_west_hr.data[mask]
    north_south_hr = north_south_hr.data[mask]

    ew_std_lr, ns_std_lr = matrix_1.return_std()
    ew_std_lr = matrix_1.trans_geo.transition_vector_to_plottable(ew_std_lr)
    ns_std_lr = matrix_1.trans_geo.transition_vector_to_plottable(ns_std_lr)

    ew_std_hr, ns_std_hr = matrix_2.return_std()
    ew_std_hr = matrix_2.trans_geo.transition_vector_to_plottable(ew_std_hr)
    ns_std_hr = matrix_2.trans_geo.transition_vector_to_plottable(ns_std_hr)
                    
    ew_std_lr = ew_std_lr.data[mask]
    ns_std_lr = ns_std_lr.data[mask]
    ew_std_hr = ew_std_hr.data[mask]
    ns_std_hr = ns_std_hr.data[mask]

    ew_mean_diff = abs(east_west_lr-east_west_hr).mean()
    ns_mean_diff = abs(north_south_lr-north_south_hr).mean()
    ew_std_diff = abs(ew_std_lr-ew_std_hr).mean()
    ns_std_diff = abs(ns_std_lr-ns_std_hr).mean()
    return (ew_mean_diff,ns_mean_diff,ew_std_diff,ns_std_diff,matrix_1.trans_geo.lat_sep,matrix_1.trans_geo.lon_sep,matrix_1.trans_geo.time_step,description)
