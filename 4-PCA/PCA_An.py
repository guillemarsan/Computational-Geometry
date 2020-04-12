# -*- coding: utf-8 -*-
"""
Coursework 4: PCA and Analogous finding


References:
    
    https://www.esrl.noaa.gov/psd/data/gridded/data.ncep.reanalysis2.pressure.html
    
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1498
    
    https://www.esrl.noaa.gov/psd/cgi-bin/db_search/DBListFiles.pl?did=59&tid=81620&vid=1497

"""
import os
import datetime as dt  # Python standard library datetime  module
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import netcdf as nc
from sklearn.decomposition import PCA

# Compute the euclidean distance between two days
def distance(a,b):
    fact1 = 0.5*np.sum((a[5,:,:] - b[5,:,:]).astype('int64')**2)
    fact2 = 0.5*np.sum((a[0,:,:] - b[0,:,:]).astype('int64')**2)
    return np.sqrt(fact1 + fact2)

# Compute the n most analogous days to a given target day a0 from a set an
def analogues(a0,an,n):
    dis = [distance(a0,a) for a in an]
    ind = np.argsort(dis)[:n]
    return ind
    
#%%
"""
Exercise 1: PCA
"""

# Load data and attributes

workpath = "C:/Users/guill/Documents/Carrera/GEOComp/PCA"
os.getcwd()
files = os.listdir(workpath)

f = nc.netcdf_file(workpath + "/hgt.2019.nc", 'r')

print(f.history)
print(f.dimensions)
print(f.variables)
time = f.variables['time'][:].copy()
time_bnds = f.variables['time_bnds'][:].copy()
time_units = f.variables['time'].units
level = f.variables['level'][:].copy()
lats = f.variables['lat'][:].copy()
lons = f.variables['lon'][:].copy()
hgt = f.variables['hgt'][:].copy()
hgt_units = f.variables['hgt'].units
hgt_scale = f.variables['hgt'].scale_factor
hgt_offset = f.variables['hgt'].add_offset
print(hgt.shape)

f.close()

"""
Example of the evolution of an air element
"""
plt.plot(time, hgt_offset + hgt[:, 1, 1, 1]*hgt_scale, c='r')
plt.show()


dt_time = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) 
           for t in time]
np.min(dt_time)
np.max(dt_time)

"""
Spatial distribution of the geopotential altitude at level 500hPa, for the first day
"""
plt.contour(lons, lats, hgt[0,5,:,:])
plt.show()

hgt2 = hgt[:,5,:,:].reshape(len(time),len(lats)*len(lons))

# Find with PCA the 4 principal components
n_components=4
Y = hgt2.transpose()
pca = PCA(n_components=n_components)
pca.fit(Y)
print(pca.explained_variance_ratio_)
out = pca.singular_values_
Element_pca = pca.fit_transform(Y)
Element_pca = Element_pca.transpose(1,0).reshape(n_components,len(lats),len(lons))

# Plot 4 principal components spacially
fig = plt.figure()
fig.subplots_adjust(hspace=0.4, wspace=0.4)
for i in range(1, 5):
    ax = fig.add_subplot(2, 2, i)
    ax.text(0.5, 90, 'PCA-'+str(i),
           fontsize=18, ha='center')
    plt.contour(lons, lats, Element_pca[i-1,:,:])
plt.show()

#%%
"""
Exercise 2: Analogous finding
"""

f = nc.netcdf_file(workpath + "/hgt.2019.nc", 'r')
hgt_19 = f.variables['hgt'][:].copy()
f.close()

f = nc.netcdf_file(workpath + "/hgt.2020.nc", 'r')
hgt_20 = f.variables['hgt'][:].copy()
time_bnds_20 = f.variables['time_bnds'][:].copy()
f.close()

# Indexes of x in (-20,20) and y in (30,50)
lats_index = np.arange(16,25)
lons_index = np.arange(-8,9)

# Get day 2020/01/20 in desired subset
hours = (dt.date(2020,1,20) - dt.date(1800,1,1)).days*24
idx = np.where(time_bnds_20[:,0] == hours)
a0 = hgt_20[idx[0][0],:,:,:]
aux = a0[:,lats_index,:]
a0_sub = aux[:,:,lons_index]

# Get 2019 days in desired subset
aux = hgt_19[:,:,lats_index,:]
an = aux[:,:,:,lons_index]

# Find the 4 days most analogous to 2020/01/20 in 2019
days = analogues(a0_sub,an,4)
dt_time = [dt.date(1800, 1, 1) + dt.timedelta(hours=t) 
           for t in time_bnds[days][:,0]]
print("The 4 days more analogous to 2020-01-20 are", [str(date) for date in dt_time])

f = nc.netcdf_file(workpath + "/air.2020.nc", 'r')
air_20 = f.variables['air'][:].copy()
air_scale = f.variables['air'].scale_factor
f.close()

f = nc.netcdf_file(workpath + "/air.2019.nc", 'r')
air_19 = f.variables['air'][:].copy()
f.close()

# Get day 2020/01/20 in desired subset with p = 1000hPa
ta0 = air_20[idx[0][0],:,:,:]
aux = ta0[:,lats_index,:]
aux2 = aux[:,:,lons_index]
ta0_sub = aux2[0,:,:]

# Get 2019 analogous days in desired subset with p = 1000hPa
tdays = air_19[days,:,:,:]
aux = tdays[:,:,lats_index,:]
aux2 = aux[:,:,:,lons_index]
tdays_sub = aux2[:,0,:,:]

# Compute the mean temperature of the analogous days in each point
av = np.mean(tdays_sub,axis = 0)

# Compute the Mean Absolute Error with 2020/01/20
diff = np.abs(ta0_sub - av)
mae = np.sum(diff)/(9*17)*air_scale
print('Mean absolute error = ',mae, 'K')

