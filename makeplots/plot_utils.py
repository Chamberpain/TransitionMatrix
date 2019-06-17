from mpl_toolkits.basemap import Basemap
import pyproj
from matplotlib.patches import Polygon
import numpy as np

class TBasemap(Basemap):
    def ellipse(self, x0, y0, a, b, n, phi=0, ax=None, **kwargs):
        """
        Draws a polygon centered at ``x0, y0``. The polygon approximates an
        ellipse on the surface of the Earth with semi-major-axis ``a`` and 
        semi-minor axis ``b`` degrees longitude and latitude, made up of 
        ``n`` vertices.

        For a description of the properties of ellipsis, please refer to [1].
        
        The polygon is based upon code written do plot Tissot's indicatrix
        found on the matplotlib mailing list at [2].
        Extra keyword ``ax`` can be used to override the default axis instance.

        Other \**kwargs passed on to matplotlib.patches.Polygon

        RETURNS
            poly : a maptplotlib.patches.Polygon object.

        REFERENCES
            [1] : http://en.wikipedia.org/wiki/Ellipse
        """
        ax = kwargs.pop('ax', None) or self._check_ax()
        g = pyproj.Geod(a=self.rmajor, b=self.rminor)
        # Gets forward and back azimuths, plus distances between initial
        # points (x0, y0)
        azf, azb, dist = g.inv([x0, x0], [y0, y0], [x0+a, x0], [y0, y0+b])
        tsid = dist[0] * dist[1] # a * b

        # Initializes list of segments, calculates \del azimuth, and goes on 
        # for every vertex
        seg = []
        AZ = np.linspace(azf[0], 360. + azf[0], n)
        for i, az in enumerate(AZ):
            # Skips segments along equator (Geod can't handle equatorial arcs).
            if np.allclose(0., y0) and (np.allclose(90., az) or
                np.allclose(270., az)):
                continue
            # In polar coordinates, with the origin at the center of the 
            # ellipse and with the angular coordinate ``az`` measured from the
            # major axis, the ellipse's equation  is [1]:
            #
            #                           a * b
            # r(az) = ------------------------------------------
            #         ((b * cos(az))**2 + (a * sin(az))**2)**0.5
            #
            # Azymuth angle in radial coordinates and corrected for reference
            # angle.
            azr = 2. * np.pi / 360. * (phi+az + 90.)
            A = dist[0] * np.sin(azr)
            B = dist[1] * np.cos(azr)
            r = tsid / (B**2. + A**2.)**0.5
            lon, lat, azb = g.fwd(x0, y0, az, r)
            x, y = self(lon, lat)

            # Add segment if it is in the map projection region.
            if x < 1e20 and y < 1e20:
                seg.append((x, y))
        # print seg
        poly = Polygon(seg, **kwargs)
        ax.add_patch(poly)

        # Set axes limits to fit map region.
        self.set_axes_limits(ax=ax)
        return poly

def basemap_setup(lat_grid,lon_grid,traj_type):
    X,Y = np.meshgrid(lon_grid,lat_grid)
    if traj_type == 'SOSE':
        print 'I am plotting antarctic region'
        m = Basemap(llcrnrlon=-180.,llcrnrlat=-80.,urcrnrlon=180.,urcrnrlat=-25,projection='cea',fix_aspect=False)
    elif traj_type == 'Argo':
        print 'I am plotting global region'
        m = Basemap(projection='cea',llcrnrlon=-180.,llcrnrlat=-80.,urcrnrlon=180.,urcrnrlat=80,fix_aspect=False)
    elif traj_type == 'Crete':
        print 'I am plotting Crete'
        m = Basemap(projection='cea',llcrnrlon=20.,llcrnrlat=30,urcrnrlon=30,urcrnrlat=40,fix_aspect=False)
    # m.fillcontinents(color='coral',lake_color='aqua')
    m.drawcoastlines()
    XX,YY = m(X,Y)
    return XX,YY,m