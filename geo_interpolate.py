#!/glade/u/apps/ch/opt/python/3.7.5/gnu/8.3.0/pkg-library/20200417/bin/python
"""

  This script reads geo_em_YYY.d01.nc file and ingests and interpolates 
  terrain height (HGT variable) on the grid provided by geo_em.d02.nc file
  then performs soothing as needed.

  To execute this script issue the following command:
  % geo_interpolate.py YYY Z
  where YYY is thee number defining the resolution of geo_em1_YYY.d01.nc file 
  and Z defines the type of interpolation:
   Z = 1 - bilinear interpolation using a rectangle with i,j to i+1,j+1 corners 
   Z = 2 - nearest neighbor
   Z = 3 - bilinear interpolation using a rectangle with i,j to i+2,j+2 corners 
   Z = 4 - average height using 3x3 gird

  The script was developed by:

  Branko Kosovic, NCAR, 2020-10-12
"""
import sys
import os.path
from smooth_geo import *

###########

plot_results = 1

smooth    = sys.argv[1]
infile = "geo_em_"+smooth+".d01.nc"
if os.path.isfile(infile):
    print("file geo_em_"+smooth+".d01.nc")
else:
    sys.exit("File "+infile+" does not exist!")

algorithm = sys.argv[2]
print("algorithm = "+algorithm)

if (algorithm != "1") and (algorithm != "2") and \
   (algorithm != "3") and (algorithm != "4"):
    sys.exit("Wrong algorithm option, it should be 1, 2, 3, or 4.")

strsmooth = "_"+str(smooth)

workdir = os.getcwd()+"/"
print(workdir)

pcopyorig=Popen(["/bin/cp", "-f","orig1_geo_em.d02.nc" ,"geo_em.d02.nc"], \
            cwd=workdir, stdout=PIPE, stderr=PIPE)
stdout = pcopyorig.communicate()[0]

pcopy=Popen(["/bin/cp", "-f","geo_em.d02.nc" ,"low_geo_em.d02.nc"], \
            cwd=workdir, stdout=PIPE, stderr=PIPE)
stdout = pcopy.communicate()[0]

low_infile = 'geo_em.d02.nc'

print(workdir+low_infile)

low_hgt_m,low_xlong_m,low_xlat_m,low_dx,low_dy = \
                                  read_terrain(workdir,low_infile)

high_infile = 'geo_em'+strsmooth+'.d01.nc'

high_hgt_m,high_xlong_m,high_xlat_m,high_dx,high_dy = \
                                  read_terrain(workdir,high_infile)

mx,my,patch_xlong_m,patch_xlat_m,patch_hgt_m = \
       find_patch(low_xlong_m,low_xlat_m,high_xlong_m,high_xlat_m,high_hgt_m)

if (algorithm == "1"):
    new_low_hgt_m = interpolate(low_xlong_m,low_xlat_m, \
                                patch_xlong_m,patch_xlat_m,patch_hgt_m)

if (algorithm == "2"):
    new_low_hgt_m = nearest_neighbor(low_xlong_m,low_xlat_m, \
                                     patch_xlong_m,patch_xlat_m,patch_hgt_m)

if (algorithm == "3"):
    new_low_hgt_m = interpolate2(low_xlong_m,low_xlat_m, \
                                 patch_xlong_m,patch_xlat_m,patch_hgt_m)

if (algorithm == "4"):
    new_low_hgt_m = interpolate3(low_xlong_m,low_xlat_m, \
                                 patch_xlong_m,patch_xlat_m,patch_hgt_m)

# Just a little bit of smoothing to remove the effects of interpolation
snew_low_hgt_m = smoothing(new_low_hgt_m)
new_low_hgt_m = snew_low_hgt_m

nc_fid = Dataset(workdir+low_infile,'r+')
hgt_m  = nc_fid.variables['HGT_M'][:]
hgt_m[0,:,:]  = np.transpose(new_low_hgt_m)

nc_fid.variables['HGT_M'][:] = hgt_m
nc_fid.close()

dx = high_dx
dy = high_dy

if plot_results == 1:
    plot_terrain(low_hgt_m[:,:],dx,dy)
    plot_terrain(high_hgt_m[mx:mx+116,my:my+116],dx,dy)
    plot_terrain(new_low_hgt_m[:,:],dx,dy)
    #plot_dots(new_low_hgt_m[:,:],low_xlong_m,low_xlat_m, \
    #                             patch_xlong_m,patch_xlat_m,dx,dy)
    plt.show()

print("Successful completion of geo.py!")

