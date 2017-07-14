#!/usr/bin/env cdat
"""
% Cloud Property Histograms. Part I: Cloud Radiative Kernels. J. Climate, 25, 3715?3735. doi:10.1175/JCLI-D-11-00248.1.

% v2: This script is written to demonstrate how to compute the cloud feedback using for a 
% short (2-year) period of MPI-ESM-LR using the difference between amipFuture and amip runs.
% One should difference longer periods for more robust results -- these are just for demonstrative purposes

% Data that are used in this script:
% 1. model clisccp field
% 2. model rsuscs field
% 3. model rsdscs field
% 4. model tas field
% 5. cloud radiative kernels

% This script written by Mark Zelinka (zelinka1@llnl.gov) on 14 July 2017

"""

#IMPORT STUFF:
#=====================
import cdms2 as cdms
import cdutil
import MV2 as MV
import numpy as np
import pylab as pl

###########################################################################
# HELPFUL FUNCTIONS FOLLOW 
###########################################################################

###########################################################################
def add_cyclic(data):
    # Add Cyclic point around 360 degrees longitude:
    lons=data.getLongitude()[:]
    dx=np.gradient(lons)[-1]
    data2 = data(longitude=(0, dx+np.max(lons)), squeeze=True)    
    return data2

###########################################################################
def nanarray(vector):

    # this generates a masked array with the size given by vector
    # example: vector = (90,144,28)

    # similar to this=NaN*ones(x,y,z) in matlab

    this=MV.zeros(vector)
    this=MV.masked_where(this==0,this)

    return this

###########################################################################
def map_SWkern_to_lon(Ksw,albcsmap):

    from scipy.interpolate import interp1d
    ## Map each location's clear-sky surface albedo to the correct albedo bin
    # Ksw is size 12,7,7,lats,3
    # albcsmap is size A,lats,lons
    albcs=np.arange(0.0,1.5,0.5) 
    A=albcsmap.shape[0]
    TT=Ksw.shape[1]
    PP=Ksw.shape[2]
    lenlat=Ksw.shape[3]
    lenlon=albcsmap.shape[2]
    SWkernel_map=nanarray((A,TT,PP,lenlat,lenlon))
    for M in range(A):
        MM=M
        while MM>11:
            MM=MM-12
        for LA in range(lenlat):
            alon=albcsmap[M,LA,:] 
            # interp1d can't handle mask but it can deal with NaN (?)
            try:
                alon2=MV.where(alon.mask,np.nan,alon)   
            except:
                alon2=alon
            if np.ma.count(alon2)>1: # at least 1 unmasked value
                if len(pl.find(Ksw[MM,:,:,LA,:]>0))==0:
                    SWkernel_map[M,:,:,LA,:] = 0
                else:
                    f = interp1d(albcs,Ksw[MM,:,:,LA,:],axis=2)
                    ynew = f(alon2.data)
                    ynew=MV.masked_where(alon2.mask,ynew)
                    SWkernel_map[M,:,:,LA,:] = ynew
            else:
                continue

    return SWkernel_map

###########################################################################
# MAIN ROUTINE FOLLOWS
###########################################################################
direc='/work/zelinka1/git/cloud-radiative-kernels/data/'

# Load in the Zelinka et al 2012 kernels:
f=cdms.open(direc+'cloud_kernels2.nc')
LWkernel=f('LWkernel')
SWkernel=f('SWkernel')
f.close()

albcs=np.arange(0.0,1.5,0.5) # the clear-sky albedos over which the kernel is computed

# LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
LWkernel_map=np.tile(np.tile(LWkernel[:,:,:,:,0],(1,1,1,1,1)),(144,1,1,1,1))(order=[1,2,3,4,0])

# Define the cloud kernel axis attributes
lats=LWkernel.getLatitude()[:]
lons=np.arange(1.25,360,2.5)
grid = cdms.createGenericGrid(lats,lons)

# Load in clisccp from two models
f=cdms.open(direc+'clisccp_cfMon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc','r')
clisccp1=f('clisccp')
f.close()
f=cdms.open(direc+'clisccp_cfMon_MPI-ESM-LR_amipFuture_r1i1p1_197901-198112.nc','r')
clisccp2=f('clisccp')
f.close()

# Make sure clisccp is in percent  
sumclisccp1=MV.sum(MV.sum(clisccp1,2),1)
sumclisccp2=MV.sum(MV.sum(clisccp2,2),1)   
if np.max(sumclisccp1) <= 1.:
    clisccp1 = clisccp1*100.        
if np.max(sumclisccp2) <= 1.:
    clisccp2 = clisccp2*100.

# Compute climatological annual cycle:
avgclisccp1=cdutil.ANNUALCYCLE.climatology(clisccp1) #(12, TAU, CTP, LAT, LON)
avgclisccp2=cdutil.ANNUALCYCLE.climatology(clisccp2) #(12, TAU, CTP, LAT, LON)
del(clisccp1,clisccp2)

# Compute clisccp anomalies
anomclisccp = avgclisccp2 - avgclisccp1

# Compute clear-sky surface albedo
f=cdms.open(direc+'rsuscs_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc','r')
rsuscs1 = f('rsuscs')
f.close()
f=cdms.open(direc+'rsdscs_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc','r')
rsdscs1 = f('rsdscs')
f.close()

albcs1=rsuscs1/rsdscs1
avgalbcs1=cdutil.ANNUALCYCLE.climatology(albcs1) #(12, 90, 144)
avgalbcs1=MV.where(avgalbcs1>1.,1,avgalbcs1) # where(condition, x, y) is x where condition is true, y otherwise
avgalbcs1=MV.where(avgalbcs1<0.,0,avgalbcs1)
del(rsuscs1,rsdscs1,albcs1)

# Load surface air temperature
f=cdms.open(direc+'tas_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc','r')
tas1 = f('tas')
f.close()
f=cdms.open(direc+'tas_Amon_MPI-ESM-LR_amipFuture_r1i1p1_197901-198112.nc','r')
tas2 = f('tas')
f.close()

# Compute climatological annual cycle:
avgtas1=cdutil.ANNUALCYCLE.climatology(tas1) #(12, 90, 144)
avgtas2=cdutil.ANNUALCYCLE.climatology(tas2) #(12, 90, 144)
del(tas1,tas2)

# Compute global annual mean tas anomalies
anomtas = avgtas2 - avgtas1
avgdtas = cdutil.averager(MV.average(anomtas,axis=0), axis='xy', weights='weighted') # (scalar)

# Regrid everything to the kernel grid:
avgalbcs1 = add_cyclic(avgalbcs1)
avgclisccp1 = add_cyclic(avgclisccp1)
avgclisccp2 = add_cyclic(avgclisccp2)
avganomclisccp = add_cyclic(anomclisccp)
avgalbcs1_grd = avgalbcs1.regrid(grid,regridTool="esmf",regridMethod = "linear")
avgclisccp1_grd = avgclisccp1.regrid(grid,regridTool="esmf",regridMethod = "linear")
avgclisccp2_grd = avgclisccp2.regrid(grid,regridTool="esmf",regridMethod = "linear")
avganomclisccp_grd = avganomclisccp.regrid(grid,regridTool="esmf",regridMethod = "linear")

# Use control albcs to map SW kernel to appropriate longitudes
SWkernel_map = map_SWkern_to_lon(SWkernel,avgalbcs1_grd)

# Compute clisccp anomalies normalized by global mean delta tas
anomclisccp = avganomclisccp_grd/avgdtas

# Compute feedbacks: Multiply clisccp anomalies by kernels
SW0 = SWkernel_map*anomclisccp
LW_cld_fbk = LWkernel_map*anomclisccp
LW_cld_fbk.setAxisList(anomclisccp.getAxisList())

# Set the SW cloud feedbacks to zero in the polar night
# The sun is down if every bin of the SW kernel is zero:
sundown=MV.sum(MV.sum(SWkernel_map,axis=2),axis=1)  #12,90,144
repsundown=np.tile(np.tile(sundown,(1,1,1,1,1)),(7,7,1,1,1))(order=[2,1,0,3,4])
SW1 = MV.where(repsundown==0, 0, SW0) # where(condition, x, y) is x where condition is true, y otherwise
SW_cld_fbk = MV.where(repsundown.mask, 0, SW1) # where(condition, x, y) is x where condition is true, y otherwise
SW_cld_fbk.setAxisList(anomclisccp.getAxisList())

# SW_cld_fbk and LW_cld_fbk contain the contributions to the feedback from cloud anomalies in each bin of the histogram


# Quick sanity check:
# print the global, annual mean LW and SW cloud feedbacks:
sumLW = MV.average(MV.sum(MV.sum(LW_cld_fbk,axis=2),axis=1),axis=0)
avgLW_cld_fbk = cdutil.averager(sumLW, axis='xy', weights='weighted')
print 'avg LW cloud feedback = '+str(avgLW_cld_fbk)
sumSW = MV.average(MV.sum(MV.sum(SW_cld_fbk,axis=2),axis=1),axis=0)
avgSW_cld_fbk = cdutil.averager(sumSW, axis='xy', weights='weighted')
print 'avg SW cloud feedback = '+str(avgSW_cld_fbk)

# Some sample global mean figures
tau=[0.,0.3,1.3,3.6,9.4,23.,60.,380.]
ctp=[1000,800,680,560,440,310,180,50]

# amip cloud fraction histogram:
pl.subplots()
data = cdutil.averager(MV.average(avgclisccp1_grd,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='Blues_r',vmin=0, vmax=10)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean amip cloud fraction')
pl.xlabel('TAU')
pl.ylabel('CTP')
pl.colorbar()

# amipFuture cloud fraction histogram:
pl.subplots()
data = cdutil.averager(MV.average(avgclisccp2_grd,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='Blues_r',vmin=0, vmax=10)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean amipFuture cloud fraction')
pl.xlabel('TAU')
pl.ylabel('CTP')
pl.colorbar()

# difference of cloud fraction histograms:
pl.subplots()
data = cdutil.averager(MV.average(anomclisccp,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='RdBu_r',vmin=-0.75, vmax=0.75)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean normalized change in cloud fraction')
pl.colorbar()

# LW cloud feedback contributions:
pl.subplots()
data = cdutil.averager(MV.average(LW_cld_fbk,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='RdBu_r',vmin=-0.75, vmax=0.75)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean LW cloud feedback contributions')
pl.colorbar()

# SW cloud feedback contributions:
pl.subplots()
data = cdutil.averager(MV.average(SW_cld_fbk,axis=0), axis='xy', weights='weighted').transpose()
pl.pcolor(data,shading='flat',cmap='RdBu_r',vmin=-0.75, vmax=0.75)
pl.xticks(np.arange(8), tau)
pl.yticks(np.arange(8), ctp)
pl.title('Global mean SW cloud feedback contributions')
pl.colorbar()
pl.show()
