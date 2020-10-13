#!/glade/u/apps/ch/opt/python/3.7.5/gnu/8.3.0/pkg-library/20200417/bin/python
"""
   geo_localfilter.py

   This script reads geo_em.d01.nc file and computes terrain slope
   then performs soothing as needed.

   The script was developed by:

   Branko Kosovic, NCAR, 2020-10-12
"""
import sys
import os
from subprocess import Popen, PIPE
from smooth_geo import *
from scipy.ndimage import gaussian_filter

if len(sys.argv) > 3:
    sys.exit("Only 2 integer arguments should be given!")

if len(sys.argv) > 1:
    try:
        sigmaxy = int(sys.argv[1])
        if (sigmaxy < 100) or (sigmaxy > 1000):
            sys.exit("Filter width must be between 100 and 1000 m.")
    except:
        sys.exit("The first argument must be an integer between 100 and 1000!")
else:
    print("Filter width will be set to the default value, 100 m.")
    sigmaxy=100.
                
print("Filter width = ",sigmaxy," m")

if len(sys.argv) == 3:
    try:
        auxfile = int(sys.argv[2]) 
        if (auxfile != 0) and (auxfile != 1):
            sys.exit("The second argument must be 0 or 1!")
    except:
        sys.exit("The second argument must be an integer, 0 or 1!")
else:
    print("Auxiliary files will not be produced.")
    auxfile = 0
     
sigmax=sigmaxy
sigmay=sigmaxy

strsigma=str(int(sigmaxy))

workdir = os.getcwd()

hi_file = 'geo_em.d01.nc'

hi_infile = "geo_em_"+strsigma+".d01.nc" 

pcopy = Popen(["/bin/cp","-f",hi_file,hi_infile], \
              cwd=workdir, stdout=PIPE, stderr=PIPE)
stdout = pcopy.communicate()[0]

nc_fid = Dataset(workdir+"/"+hi_infile,'r+')

hgt_m   = nc_fid.variables['HGT_M'][:]
xlong_c = nc_fid.variables['XLONG_C'][:]
xlat_c  = nc_fid.variables['XLAT_C'][:]

nt, ni, nj = hgt_m.shape

##########################
# Original terrain data
hgt = np.squeeze(hgt_m[0,:,:])

lats=np.arange(hgt.shape[0])
lons=np.arange(hgt.shape[1])

ncattrs = nc_fid.ncattrs()

dx = nc_fid.getncattr('DX')
dy = nc_fid.getncattr('DY')

# original slope
beta1 = slope(hgt,dx,dy)
print("Original data, maximum slope = ",beta1.max()," degrees")
gamma1 = np.reshape(beta1,beta1.shape[0]*beta1.shape[1])
n, bins, patches = plt.hist(gamma1, 30, density=True, facecolor='red', alpha=1.0)

##########################
# Filtered terrain data
fhgt = gaussian_filter(hgt,sigma=sigmaxy/dx)
beta2 = slope(fhgt,dx,dy)
print("Filtered data, maximum slope = ",beta2.max()," degrees")

# slope of the filtered terrain
delta1 = 180./math.pi*np.arctan(np.absolute(hgt-fhgt)/dx)


if (100 <= sigmaxy < 200):
    w = weight(delta1,10.,40.)
if (200 <= sigmaxy < 300):
    w = weight(delta1,15.,50.)
if (300 <= sigmaxy < 500):
    w = weight(delta1,15.,50.)
if (500 <= sigmaxy <= 1000):
    w = weight(delta1,20.,50.)

##########################
# Locally filtered terrain data
hgtnew = (1.-w)*hgt+w*fhgt

hgt_m[0,:,:] = hgtnew

nc_fid.variables['HGT_M'][:] = hgt_m

nc_fid.close()

lats=np.arange(hgtnew.shape[0])
lons=np.arange(hgtnew.shape[1])

beta3 = slope(hgtnew,dx,dy)
print("Locally filtered data, maximum slope = ",beta3.max()," degrees")

gamma2 = np.reshape(beta2,beta2.shape[0]*beta2.shape[1])
gamma3 = np.reshape(beta3,beta3.shape[0]*beta3.shape[1])

n, bins, patches = plt.hist(gamma3, 30, density=True, facecolor='green', alpha=1.0)
n, bins, patches = plt.hist(gamma2, 30, density=True, facecolor='blue', alpha=1.0)

if auxfile == 1:
    write_netcdf('hgt_terrain.nc',hgt,'hgt',lats,lons,'High resolution')
    write_netcdf('hgt_filtered_terrain.nc',fhgt,'fhgt',lats,lons, \
                 'Filtered high resolution')
    write_netcdf('hgt_locally_filtered_terrain.nc',hgtnew,'hgtnew',lats,lons, \
                 'Filtered high resolution')
    write_netcdf('hgt_terrain_difference.nc',hgt-fhgt,'dhgt',lats,lons, \
                 'Terrain high difference')

    write_netcdf('hgt_slope.nc',beta1,'beta1',lats,lons,'High resolution')
    write_netcdf('hgt_filtered_slope.nc',beta2,'beta2',lats,lons, \
                 'Filtered high resolution')
    write_netcdf('hgt_locally_filtered_slope.nc',beta3,'beta3',lats,lons, \
                 'Locally filtered high resolution')
    write_netcdf('hgt_slope_difference.nc',beta1-beta3,'dbeta',lats,lons, \
                 'Slope difference high resolution')

    write_netcdf('hgt_weight.nc',w,'w',lats,lons,'Local filtering weight')

print("Original data, minimum and maximum heights = ",hgt.min(),hgt.max()," m")
print("Filtered data, minimum and maximum heights = ",         \
      fhgt.min(),fhgt.max()," m")
print("Locally filtered data, minimum and maximum heights = ", \
      hgtnew.min(),hgtnew.max()," m")

plt.yscale('log', nonposy='clip')
plt.title("Distribution of slopes in degrees")
plt.text(10.,0.2,"Red - original, Blue - filtered, Green - locally filtered")
plt.show()
