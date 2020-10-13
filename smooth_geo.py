#
"""
   smooth_goe.py

   This script reads geo_em.d01.nc file and computes terrain slope
   then performs soothing as needed.

   The script was developed by:

   Branko Kosovic, NCAR, 2020-10-12
"""
import sys
import os
import re
import numpy as np
import math
import time
import matplotlib
from scipy import signal, special
from netCDF4 import Dataset, stringtoarr
from subprocess import Popen, PIPE
from matplotlib import pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib.path import Path
import matplotlib.patches as patches
from matplotlib.patches import Circle, Wedge, Polygon
from matplotlib.collections import PatchCollection
from numba import jit

################
###### Plotting
def plot_gridcells(xl,yl,patch_xlong_m,patch_xlat_m):

    fig = plt.figure()
    ax  = fig.add_subplot(111)

    for k in range(patch_hgt_m.shape[0]-1):
        for l in range(patch_hgt_m.shape[1]-1):
            poly = [ (patch_xlong_m[k,l], patch_xlat_m[k,l]), \
                     (patch_xlong_m[k+1,l], patch_xlat_m[k+1,l]), \
                     (patch_xlong_m[k+1,l+1], patch_xlat_m[k+1,l+1]), \
                     (patch_xlong_m[k,l+1], patch_xlat_m[k,l+1]) ]

            polygon=Polygon(poly,closed=True)
            patches.append(polygon)

    p = PatchCollection(patches,cmap=matplotlib.cm.jet,alpha=0.4)
    colors=100*np.random.rand(len(patches))
    p.set_array(np.array(colors))

    ax.add_collection(p)

    xmin=np.min(patch_xlong_m)
    xmax=np.max(patch_xlong_m)
    ymin=np.min(patch_xlat_m)
    ymax=np.max(patch_xlat_m)

    ax.set_xlim([xmin,xmax])
    ax.set_ylim([ymin,ymax])
    plt.plot(xl,yl,color='green',marker=r'$\clubsuit$')

################
###### Plotting dots on terrain
def plot_dots(z,xl,yl,xh,yh,dx,dy):

    y, x = np.mgrid[slice(0, z.shape[0]*dx, dx),
                    slice(0, z.shape[1]*dy, dy)]

    # pick the desired colormap, 
    #cmap = plt.get_cmap('gist_earth')
    cmap = plt.get_cmap('terrain')

    # determine sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    levels = MaxNLocator(nbins=70).tick_values(z.min(), z.max())
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig0, ax0 = plt.subplots()

    im = ax0.pcolormesh(yl, xl, z, cmap=cmap, norm=norm)

    im = ax0.scatter(yl.flatten(),xl.flatten(),marker='o',color='r',s=0.5)
    im = ax0.scatter(yh.flatten(),xh.flatten(),marker='o',color='b',s=0.5)
    #fig0.colorbar(im, ax=ax0)
    ax0.set_title('Terrain - pcolormesh')
    plt.axis([yl.min(), yl.max(), xl.min(), xl.max()])


    return

#########


################
###### Plotting terrain
def plot_terrain(z,dx,dy):

    y, x = np.mgrid[slice(0, z.shape[0]*dx, dx),
                    slice(0, z.shape[1]*dy, dy)]

    # pick the desired colormap, 
    #cmap = plt.get_cmap('gist_earth')
    cmap = plt.get_cmap('terrain')

    # determine sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    levels = MaxNLocator(nbins=70).tick_values(z.min(), z.max())
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)

    fig0, ax0 = plt.subplots()

    im = ax0.pcolormesh(x, y, z, cmap=cmap, norm=norm)

    fig0.colorbar(im, ax=ax0)
    ax0.set_title('Terrain - pcolormesh')
    plt.axis([x.min(), x.max(), y.min(), y.max()])

    # contours are *point* based plots, so convert our bound into point
    # centers

    #fig1, ax1 = plt.subplots()

    #cf = ax1.contourf(x, y, z, levels=levels, cmap=cmap)

    #fig1.colorbar(cf, ax=ax1)
    #ax1.set_title('Terrain - contourf')

    #plt.show()

    return

#########

################
###### Bilinear Interpolation
@jit(nopython=True)
def bilinear(x,y,x1,y1,x2,y2,z11,z21,z22,z12):

    z = 1./((x2-x1)*(y2-y1))*(z11*(x2-x)*(y2-y)+z21*(x-x1)*(y2-y)+ \
                              z12*(x2-x)*(y-y1)+z22*(x-x1)*(y-y1))

    return z

@jit(nopython=True)
def unit_bilinear(x,y,z11,z21,z12,z22):

    z = (z11*(1.-x)*(1.-y)+z21*x*(1.-y)+z12*(1.-x)*y+z22*x*y)

    return z

@jit(nopython=True)
def check_in_square(x,y):

    inside = False

    if((x > 0.) and (x <= 1.) and (y > 0.) and (y <= 1.)):
       inside = True

    return inside

#
def quadri(xl,yl,xql,yql,zql):

    #am = np.array([[1.,0.,0.,0.],[1.,1.,0.,0.],[1.,1.,1.,1.],[1.,0.,1.,0.]])
    #ami = np.linalg.inv(am)
    ami = np.array([[ 1., 0., 0., 0.], \
                    [-1., 1., 0., 0.], \
                    [-1., 0., 0., 1.], \
                    [ 1.,-1., 1.,-1.]])

    a = ami.dot(xql)
    b = ami.dot(yql)

    aa = a[3]*b[2] - a[2]*b[3]
    bb = a[3]*b[0] - a[0]*b[3] + a[1]*b[2] - a[2]*b[1] + xl*b[3] - yl*a[3]
    cc = a[1]*b[0] - a[0]*b[1] + xl*b[1] - yl*a[1]

    if (aa != 0.):

        ys1 = (-bb - math.sqrt(bb*bb - 4*aa*cc))/(2.*aa)
        ys2 = (-bb + math.sqrt(bb*bb - 4*aa*cc))/(2.*aa)

        if ((ys1 < 0.) or (ys1 > 1.)):
            ys = ys2

        if ((ys2 < 0.) or (ys2 > 1.)):
            ys = ys1

    if (aa == 0.):
        ys = -cc/bb

    xs = (xl - a[0] - a[2]*ys) / (a[1] + a[3]*ys)

    z11=zql[0]
    z21=zql[1]
    z22=zql[2]
    z12=zql[3]

    inside = check_in_square(xs,ys)

    if not inside:
        z = None

    if inside:
        z = unit_bilinear(xs,ys,z11,z21,z22,z12)

    return z

#
def nearest_neighbor(low_xlong_m,low_xlat_m, \
                     high_xlong_m,high_xlat_m,high_hgt_m):

    new_low_hgt_m = np.zeros(low_xlong_m.shape)

    for i in range(low_xlat_m.shape[0]):
        for j in range(low_xlat_m.shape[1]):

            xl = low_xlong_m[i,j]
            yl = low_xlat_m[i,j]

            difflong = high_xlong_m.astype(np.float64) - xl.astype(np.float64)
            difflat  = high_xlat_m.astype(np.float64) - yl.astype(np.float64)
            dist = difflong*difflong + difflat*difflat

            nx, ny = np.unravel_index(dist.argmin(),dist.shape)

            z = high_hgt_m[nx,ny]

            new_low_hgt_m[i,j] = z

    return new_low_hgt_m

#
def interpolate(low_xlong_m,low_xlat_m,high_xlong_m,high_xlat_m,high_hgt_m):

    b = 1

    new_low_hgt_m = np.zeros(low_xlong_m.shape)

    for i in range(low_xlat_m.shape[0]):
        for j in range(low_xlat_m.shape[1]):

            xl = low_xlong_m[i,j]
            yl = low_xlat_m[i,j]

            difflong = high_xlong_m.astype(np.float64) - xl.astype(np.float64)
            difflat  = high_xlat_m.astype(np.float64) - yl.astype(np.float64)
            #difflong = high_xlong_m - xl
            #difflat  = high_xlat_m - yl
            dist = difflong*difflong + difflat*difflat

            nx, ny = np.unravel_index(dist.argmin(),dist.shape)

            patch_xlong_m = high_xlong_m[nx-b:nx+b+1,ny-b:ny+b+1]
            patch_xlat_m  = high_xlat_m[nx-b:nx+b+1,ny-b:ny+b+1]
            patch_hgt_m   = high_hgt_m[nx-b:nx+b+1,ny-b:ny+b+1]

            z = interpolate_one(xl,yl,patch_xlong_m,patch_xlat_m,patch_hgt_m)

            new_low_hgt_m[i,j] = z

    return new_low_hgt_m

#
def interpolate2(low_xlong_m,low_xlat_m,high_xlong_m,high_xlat_m,high_hgt_m):

    b = 1

    new_low_hgt_m = np.zeros(low_xlong_m.shape)

    for i in range(low_xlat_m.shape[0]):
        for j in range(low_xlat_m.shape[1]):

            xl = low_xlong_m[i,j]
            yl = low_xlat_m[i,j]

            difflong = high_xlong_m.astype(np.float64) - xl.astype(np.float64)
            difflat  = high_xlat_m.astype(np.float64) - yl.astype(np.float64)
            dist = difflong*difflong + difflat*difflat

            nx, ny = np.unravel_index(dist.argmin(),dist.shape)

            patch_xlong_m = high_xlong_m[nx-b:nx+b+1,ny-b:ny+b+1]
            patch_xlat_m  = high_xlat_m[nx-b:nx+b+1,ny-b:ny+b+1]
            patch_hgt_m   = high_hgt_m[nx-b:nx+b+1,ny-b:ny+b+1]

            z = interpolate_two(xl,yl,patch_xlong_m,patch_xlat_m,patch_hgt_m)

            new_low_hgt_m[i,j] = z

    return new_low_hgt_m

#
def interpolate3(low_xlong_m,low_xlat_m,high_xlong_m,high_xlat_m,high_hgt_m):

    b = 1

    new_low_hgt_m = np.zeros(low_xlong_m.shape)

    for i in range(low_xlat_m.shape[0]):
        for j in range(low_xlat_m.shape[1]):

            xl = low_xlong_m[i,j]
            yl = low_xlat_m[i,j]

            difflong = high_xlong_m.astype(np.float64) - xl.astype(np.float64)
            difflat  = high_xlat_m.astype(np.float64) - yl.astype(np.float64)
            dist = difflong*difflong + difflat*difflat

            nx, ny = np.unravel_index(dist.argmin(),dist.shape)

            patch_xlong_m = high_xlong_m[nx-b:nx+b+1,ny-b:ny+b+1]
            patch_xlat_m  = high_xlat_m[nx-b:nx+b+1,ny-b:ny+b+1]
            patch_hgt_m   = high_hgt_m[nx-b:nx+b+1,ny-b:ny+b+1]

            z = interpolate_three(xl,yl,patch_xlong_m,patch_xlat_m,patch_hgt_m)

            new_low_hgt_m[i,j] = z

    return new_low_hgt_m

#
def interpolate_one(xl,yl,patch_xlong_m,patch_xlat_m,patch_hgt_m):

    m = 0
    p = 1

    z = None

    for i in range(patch_hgt_m.shape[0]-1):
        for j in range(patch_hgt_m.shape[1]-1):

            xql1 = patch_xlong_m[i+m,j+m]
            yql1 = patch_xlat_m[i+m,j+m]
            xql2 = patch_xlong_m[i+p,j+m]
            yql2 = patch_xlat_m[i+p,j+m]
            xql3 = patch_xlong_m[i+p,j+p]
            yql3 = patch_xlat_m[i+p,j+p]
            xql4 = patch_xlong_m[i+m,j+p]
            yql4 = patch_xlat_m[i+m,j+p]
            zql11 = patch_hgt_m[i+m,j+m]
            zql21 = patch_hgt_m[i+p,j+m]
            zql22 = patch_hgt_m[i+p,j+p]
            zql12 = patch_hgt_m[i+m,j+p]

            xql = np.array([xql1, xql2, xql3, xql4])
            yql = np.array([yql1, yql2, yql3, yql4])
            zql = np.array([zql11,zql21,zql22,zql12])

            z = quadri(xl,yl,xql,yql,zql)

            if z != None:
                break

        if z != None:
            break

    return z

#
def interpolate_two(xl,yl,patch_xlong_m,patch_xlat_m,patch_hgt_m):

    m = 0
    p = 2

    z = None

    xql1 = patch_xlong_m[m,m]
    yql1 = patch_xlat_m[m,m]
    xql2 = patch_xlong_m[p,m]
    yql2 = patch_xlat_m[p,m]
    xql3 = patch_xlong_m[p,p]
    yql3 = patch_xlat_m[p,p]
    xql4 = patch_xlong_m[m,p]
    yql4 = patch_xlat_m[m,p]
    zql11 = patch_hgt_m[m,m]
    zql21 = patch_hgt_m[p,m]
    zql22 = patch_hgt_m[p,p]
    zql12 = patch_hgt_m[m,p]

    xql = np.array([xql1, xql2, xql3, xql4])
    yql = np.array([yql1, yql2, yql3, yql4])
    zql = np.array([zql11,zql21,zql22,zql12])

    z = quadri(xl,yl,xql,yql,zql)

    return z

#
def interpolate_three(xl,yl,patch_xlong_m,patch_xlat_m,patch_hgt_m):

    z = 0.

    for i in range(patch_hgt_m.shape[0]):
        for j in range(patch_hgt_m.shape[1]):

            z = z + patch_hgt_m[i,j]

    z = z/9.

    return z

@jit(nopython=True)
def smoothing(new_low_hgt_m):

    snew_low_hgt_m = new_low_hgt_m

    for i in range(1,new_low_hgt_m.shape[0]-1):
        for j in range(1,new_low_hgt_m.shape[1]-1):
            snew_low_hgt_m[i,j] = (3.*new_low_hgt_m[i,j] +      \
                                   2.*new_low_hgt_m[i-1,j] +    \
                                   2.*new_low_hgt_m[i,j-1] +    \
                                   2.*new_low_hgt_m[i+1,j] +    \
                                   2.*new_low_hgt_m[i,j+1] +    \
                                      new_low_hgt_m[i-1,j-1] +  \
                                      new_low_hgt_m[i-1,j+1] +  \
                                      new_low_hgt_m[i+1,j-1] +  \
                                      new_low_hgt_m[i+1,j+1])/15.

    return snew_low_hgt_m

#########

def read_terrain(workdir,infile):

    nc_fid = Dataset(workdir+infile,'r')

    hgt_m    = nc_fid.variables['HGT_M'][:]
    xlong_m  = nc_fid.variables['XLONG_M'][:]
    xlat_m   = nc_fid.variables['XLAT_M'][:]
    mapfac_m = nc_fid.variables['MAPFAC_M'][:]

    dx      = nc_fid.getncattr('DX')
    dy      = nc_fid.getncattr('DY')

    nc_fid.close()

    hgt_m_t    = np.transpose(hgt_m[0,:,:])
    xlong_m_t  = np.transpose(xlong_m[0,:,:])
    xlat_m_t   = np.transpose(xlat_m[0,:,:])
    mapfac_m_t = np.transpose(mapfac_m[0,:,:])

    return hgt_m_t,xlong_m_t,xlat_m_t,dx,dy

#########

def find_patch(low_xlong_m,low_xlat_m,high_xlong_m,high_xlat_m,high_hgt_m):

    xl = low_xlong_m[0,0]
    yl = low_xlat_m[0,0]

    dist = np.sqrt(np.power((high_xlong_m - xl),2)  + \
                   np.power((high_xlat_m  - yl),2))

    mx, my = np.unravel_index(dist.argmin(),dist.shape)

    b = 20

    patch_xlong_m = high_xlong_m[mx-b:mx+116+b,my-b:my+116+b]
    patch_xlat_m  = high_xlat_m[mx-b:mx+116+b,my-b:my+116+b]
    patch_hgt_m   = high_hgt_m[mx-b:mx+116+b,my-b:my+116+b]

    return mx,my,patch_xlong_m,patch_xlat_m,patch_hgt_m

#########


@jit(nopython=True)
def tricube(z,dx,dy,s):

    y, x = np.mgrid[slice(0, z.shape[0]*dx, dx),
                    slice(0, z.shape[1]*dy, dy)]

    xm = np.mean(x)
    ym = np.mean(y)

    d = np.sqrt( np.power((x-xm),2)+np.power((y-ym),2) )

    d = d/s

    g = np.zeros((z.shape[0],z.shape[1]))
   
    g[np.where(d <= 1.)] = np.power((1 - np.power(d[np.where(d <= 1.)],3)),3)

    return g

#########

@jit(nopython=True)
def gauss(z,dx,dy,sigmax,sigmay):

    y, x = np.mgrid[slice(0, z.shape[0]*dx, dx),
                    slice(0, z.shape[1]*dy, dy)]

    xm = np.mean(x)
    ym = np.mean(y)

    g = np.exp(-0.5*(np.power((x-xm),2)/(sigmax*sigmax)   \
                    +np.power((y-ym),2)/(sigmay*sigmay))) \
        /(2.*math.pi*sigmax*sigmay)

    g = g/np.sum(g)

    #print(g.max(),g.min(),np.sum(g))

    return g

#########
#@jit(nopython=True)
def weight(a,ainflection,agrad):

    c = 0.5
    a = a/90.- ainflection/90.
    a = a * agrad

    w = c*(1.+special.erf(a/math.sqrt(2)))

    return w

#########

#@jit(nopython=True)
def slope(hgt,dx,dy):

    rdx = 1./dx
    rdy = 1./dy

    hx   = hgt - np.roll(hgt,1,axis=0)
    hy   = hgt - np.roll(hgt,1,axis=1)
    hxy  = hgt - np.roll(np.roll(hgt,1,axis=0),1,axis=1)
    diag= math.sqrt(dx*dx + dy*dy)

    hx[0,:]  = 0.
    hx[:,0]  = 0.
    hy[0,:]  = 0.
    hy[:,0]  = 0.
    hxy[0,:]  = 0.
    hxy[:,0]  = 0.

    alpha1 = 180./math.pi*np.arctan(hx/dx)
    alpha2 = 180./math.pi*np.arctan(hy/dy)
    alpha3 = 180./math.pi*np.arctan(hxy/diag)

    alpha4   = np.maximum(alpha1,alpha2)
    alpha    = np.maximum(alpha3,alpha4)

    return np.absolute(alpha)

#########

def write_netcdf(filename,varvalue,varname,lats,lons,description):

    ncfile = Dataset(filename , 'w' , format='NETCDF4_CLASSIC')

    ncfile.createDimension('lat' , lats.shape[0])
    ncfile.createDimension('lon' , lons.shape[0])
    ncfile.createDimension('time' , None)

    for dimname in ncfile.dimensions.keys():
        dim = ncfile.dimensions[dimname]
        #print dimname, len(dim), dim.isunlimited()

    times = ncfile.createVariable('time', np.float32,('time',))
    latitudes = ncfile.createVariable('latitude', np.float32,('lat',))
    longitudes = ncfile.createVariable('longitude', np.float32,('lon',))

    var = ncfile.createVariable(varname, np.float32, \
                                ('time','lat','lon'))

    for varname in ncfile.variables.keys():
        vari = ncfile.variables[varname]
        #print varname, vari.dtype, vari.dimensions, vari.shape

    latitudes[:]  = lats
    longitudes[:] = lons
    var[0,:,:]    = varvalue

    # Global Attributes
    ncfile.description = description+' terrain'
    ncfile.history = 'Created ' + time.ctime(time.time())
    ncfile.source = 'netCDF4 python module tutorial'

    # Variable Attributes
    latitudes.units = 'degree_north'
    longitudes.units = 'degree_east'
    var.units = 'K'

    times.units = 'hours since 0001-01-01 00:00:00'
    times.calendar = 'gregorian'

    ncfile.close()

    return

#########
