#!/usr/bin/env python
# coding: utf-8

# ## Description
# Same as my local version of the cloud Adjustments code, but now targeting rapid cloud adjustments to 4xCO2

# In[1]:


# User decision:
# flavor='amip' # use the CFMIP exps
flavor='sstClim' # use the RFMIP exps


# ## Import useful functions

# In[2]:


import cal_CloudRadKernel_xr as CRK
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xsearch as xs
import pandas as pd
import xarray as xr
import numpy as np
import warnings
import glob
import copy
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


# In[3]:


plt.rcParams['figure.facecolor'] = 'w'
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['font.size'] = 12
plt.rcParams['legend.frameon'] = False
plt.rcParams['legend.handletextpad'] = 0.4


# ## Get filepaths based on user input

# In[4]:


freq='mon' 
realm = 'atmos'
eras = ['CMIP5','CMIP6']
xs_paths = {}
variables = ['tas','rsdscs','rsuscs','clisccp'] # necessary for cloud Adjustment calcs
for era in eras:
    xs_paths[era] = {}
    if era=='CMIP5':
        if flavor=='amip':
            exps = ['amip','amip4xCO2']
        elif flavor=='sstClim':
            exps = ['sstClim','sstClim4xCO2']
    elif era=='CMIP6':
        if flavor=='amip':
            exps = ['amip','amip-4xCO2']
        elif flavor=='sstClim':
            exps = ['piClim-control','piClim-4xCO2']
    for exp in exps:
        xs_paths[era][exp] = {}
        for var in variables:
            xs_paths[era][exp][var] = {}
            dpaths = xs.findPaths(exp, var, freq, mip_era=era, realm='atmos')
            models = xs.natural_sort(xs.getGroupValues(dpaths, 'model'))
            for model in models:
                xs_paths[era][exp][var][model] = {}
                dpaths_model = xs.retainDataByFacetValue(dpaths, 'model', model)
                members = xs.natural_sort(xs.getGroupValues(dpaths_model, 'member'))
                for member in members:
                    dpaths_model_member_list = xs.getValuesForFacet(dpaths_model, 'member', member)
                    if len(dpaths_model_member_list) > 1:
                        print('Error: multiple paths detected for ', model, member, ': ', dpaths_model_member_list)
                    else:
                        dpath = dpaths_model_member_list[0]
                        # ncfiles = xs.natural_sort(glob.glob(os.path.join(dpath, '*.nc')))
                        xs_paths[era][exp][var][model][member] = dpath #ncfiles


# In[5]:


# Re-order the nesting
filepath = {}
for era, exp_dict in xs_paths.items():
    for exp, var_dict in exp_dict.items():
        for var, model_dict in var_dict.items():
            for model, member_dict in model_dict.items():
                for member, value in member_dict.items():
                    if era not in filepath:
                        filepath[era] = {}
                    if model not in filepath[era]:
                        filepath[era][model] = {}
                    if member not in filepath[era][model]:
                        filepath[era][model][member] = {}
                    if exp not in filepath[era][model][member]:
                        filepath[era][model][member][exp] = {}
                    filepath[era][model][member][exp][var] = value


# ## Do all the cloud Adjustment calculations

# In[6]:


def clean_up(output,obsc_output,era,model,member,exps):
    
    trueHI={}
    bands=['LW','SW','NET']
    fields = ['tot','amt','alt','tau','err']
    for field in fields:
        for band in bands:
            trueHI[band+'cld_'+field] = copy.deepcopy(output['HI680'][band+'cld_'+field])

    LOobsc = obsc_output['LO680']
    # update the high cloud amount Adjustment to include the change in obscuration term:
    trueHI['LWcld_amt'] += LOobsc['LWdobsc_fbk']
    trueHI['SWcld_amt'] += LOobsc['SWdobsc_fbk']
    trueHI['NETcld_amt'] = trueHI['LWcld_amt'] + trueHI['SWcld_amt']
    # update the total high cloud Adjustment:
    trueHI['LWcld_tot'] = trueHI['LWcld_amt'] + trueHI['LWcld_alt'] + trueHI['LWcld_tau'] + trueHI['LWcld_err']
    trueHI['SWcld_tot'] = trueHI['SWcld_amt'] + trueHI['SWcld_alt'] + trueHI['SWcld_tau'] + trueHI['SWcld_err']
    trueHI['NETcld_tot'] = trueHI['LWcld_tot'] + trueHI['SWcld_tot']

    LOobsc = obsc_output['LO680']
    # SWcld_tot is equivalent to SWdunobsc_fbk; ditto for LW compoent
    # SWdobsc_fbk is already incorporated into high cloud amount Adjustment; ditto for LW component

    trueLO={}
    bands=['LW','SW','NET']
    fields = ['tot','amt','alt','tau','err']
    for field in fields:
        for band in bands:
            trueLO[band+'cld_'+field] = copy.deepcopy(LOobsc[band+'cld_'+field])
    # include the obscuration covariance term with the kernel residual:
    trueLO['LWcld_err'] += LOobsc['LWdobsc_cov_fbk']
    trueLO['SWcld_err'] += LOobsc['SWdobsc_cov_fbk']
    trueLO['NETcld_err'] = trueLO['LWcld_err']+trueLO['SWcld_err']
    # update the total low cloud Adjustment:
    trueLO['LWcld_tot'] = trueLO['LWcld_amt'] + trueLO['LWcld_alt'] + trueLO['LWcld_tau'] + trueLO['LWcld_err']
    trueLO['SWcld_tot'] = trueLO['SWcld_amt'] + trueLO['SWcld_alt'] + trueLO['SWcld_tau'] + trueLO['SWcld_err']
    trueLO['NETcld_tot'] = trueLO['LWcld_tot'] + trueLO['SWcld_tot']

    ALL = xr.Dataset(output['ALL'])
    LO = xr.Dataset(output['LO680'])
    HI = xr.Dataset(output['HI680'])
    trueLO = xr.Dataset(trueLO)
    trueHI = xr.Dataset(trueHI)

    # SAVE MAPS TO NETCDF
    DSnames = ['ALL','LO','HI','trueLO','trueHI']
    for d,DS in enumerate([ALL,LO,HI,trueLO,trueHI]):
        name = DSnames[d]
        if era=='CMIP5':
            path = '/p/user_pub/climate_work/zelinka1/cmip5/'+exps[1]+'/'
        else:
            path = '/p/user_pub/climate_work/zelinka1/cmip6/'+exps[1]+'/'
        savefile2 = path+model+'.'+member+'.'+name+'_adjs.nc'
        if os.path.exists(savefile2):
            os.remove(savefile2)
            
        DS['time'] = np.arange(12)
        ds=DS
        ds1=ds.reset_coords(names='albcs', drop=True)
        try:
            ds2=ds1.reset_coords(names='height', drop=True)
        except:
            ds2=ds1
        ds2.coords    
        ds2.to_netcdf(savefile2)
        print('Saved '+savefile2)

    return(ALL,LO,HI,trueLO,trueHI)


# In[7]:


# CCSM4 messed up their amip4xCO2 run; it seems to be identical to amip


# In[8]:


# # models=list(filepath['CMIP5'].keys())
# # for model in models:
# #     ripfs=list(filepath['CMIP5'][model].keys())
# #     for ripf in ripfs:
# #         exps=list(filepath['CMIP5'][model][ripf].keys())
# #         for exp in exps:
# #             vars=list(filepath['CMIP5'][model][ripf][exp].keys())
# #             for var in vars[:1]:
# #                 ds=xr.open_mfdataset(filepath['CMIP5'][model][ripf][exp][var]+'*nc')
# #                 print(filepath['CMIP5'][model][ripf][exp][var])
# #                 print(ds.time.dt.year)
# ds=xr.open_mfdataset(filepath['CMIP5'][model][ripf][exp][var]+'*nc')
# print(filepath['CMIP5'][model][ripf][exp][var])
# print(ds.time.dt.year)                
# # tslice=slice(None,None)
# # print(ds.sel(time=tslice).time.dt.year)                


# In[ ]:


AVGS={}
for era in eras:
    models = list(filepath[era].keys())
    for model in models:
        members = list(filepath[era][model].keys())
        for member in members:
            if model=='IPSL-CM5A-LR' and member=='r2i1p1':
                continue  
                                
            if era=='CMIP5':
                path = '/p/user_pub/climate_work/zelinka1/cmip5/'+exps[1]+'/'
            else:
                path = '/p/user_pub/climate_work/zelinka1/cmip6/'+exps[1]+'/'
            savefile2 = path+model+'.'+member+'.ALL_adjs.nc' 
            if os.path.exists(savefile2): 
                print('Already saved '+model+'.'+member)
                continue              
            
            cnt=0
            exps=list(filepath[era][model][member].keys())
            if len(exps)<2:
                print('Skipping '+model+'.'+member+' -- not enough exps')
                continue
            for exp in exps:
                variables=list(filepath[era][model][member][exp].keys())
                for var in variables:
                    cnt+=1
            if cnt<8:
                print('Skipping '+model+'.'+member+' -- not enough fields')
                continue
                                
            print('Working on '+model+'.'+member)
            
            (output,obsc_output) = CRK.CloudRadKernel(filepath[era][model][member],rapidAdj = True) 
            
            # Finalize the calculations and save netcdfs
            (ALL,LO,HI,trueLO,trueHI) = clean_up(output,obsc_output,era,model,member,exps)

            print('Done with '+model+'.'+member)


# In[ ]:


def global_avgs(DS):
    # Compute global averages:
    GLavgs={}
    avgmap = DS.mean('time')
    avgmap = avgmap.bounds.add_missing_bounds()
    for var in avgmap.data_vars:
        if '_bnds' in var:
            continue
        GLavgs[var] = avgmap.spatial.average(var, axis=["X", "Y"])[var].values
    return GLavgs        


# In[ ]:


MAPS={}
AVGS={}
DSnames = ['ALL','LO','HI','trueLO','trueHI']
# for d,DS in enumerate([ALL,LO,HI,trueLO,trueHI]):
for sec in DSnames:
    plt.figure(figsize=(12,12))
    # sec = DSnames[d]
    if flavor=='amip':
        gg=glob.glob('/p/user_pub/climate_work/zelinka1/cmip*/amip*4xCO2/*.'+sec+'_adjs.nc')
    elif flavor=='sstClim':
        gg=glob.glob('/p/user_pub/climate_work/zelinka1/cmip*/*Clim*4xCO2/*.'+sec+'_adjs.nc')        
    gg.sort()
    modripfs=[]
    winners=[]
    for g in gg:
        model = g.split('/')[-1].split('.')[0]
        member = g.split('/')[-1].split('.')[1]   
        if model=='IPSL-CM6A-LR' and member=='r22i1p1f1': # include only one IPSL-CM6A-LR model variant
            continue
        modripfs.append(model+'.'+member)
        winners.append(g)

    ds0 = xr.open_mfdataset(winners, concat_dim = 'modripf', combine = 'nested')#,coords='minimal')#,compat='override',decode_times=False)
    C = ds0.coords
    # Give model info to the model dimension
    ds = ds0.assign_coords({
        'modripf': modripfs,
        'time': C['time'],
        'lat': C['lat'],
        'lon': C['lon'],
    })
    ds.mean('time')['NETcld_tot'].plot(col='modripf',col_wrap=4,vmin=-7.5,vmax=7.5,cmap='RdBu_r')
    plt.suptitle(sec,y=1.02)

    # compute global averages:
    AVGS[sec] = global_avgs(ds)
    MAPS[sec] = ds.mean('time')


# In[ ]:


MAPS[sec]


# In[ ]:


modripfs = MAPS['ALL'].modripf
modripfs5=[]
modripfs6=[]
for modripf in modripfs:
    mr = str(modripf.values)
    ripf = mr.split('.')[-1]
    if 'f' in ripf:
        modripfs6.append(mr)
    else:
        modripfs5.append(mr)
modripfs5,modripfs6


# In[ ]:


lons = MAPS['LO'].lon
lats = MAPS['LO'].lat
LEVELS = np.arange(-2.0,2.25,0.25)
modripfs=modripfs5+modripfs6
fig, axs = plt.subplots(2, 3, figsize=(12,6), subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})

axs = axs.flatten()

ax=axs[0]
A=MAPS['HI'].mean('modripf')['NETcld_amt']
ax.coastlines()
ax.contourf(lons, lats, A, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
ax.text(0, 0.5,'Nonlow', fontsize=14, ha='right', va='center', rotation=90, transform=ax.transAxes) # no clue why set_ylabel doesn't work
ax.set_title('Original')

ax=axs[1]
B=MAPS['trueHI'].mean('modripf')['NETcld_amt']
ax.coastlines()
ax.contourf(lons, lats, B, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
ax.set_title('Multi-Model Mean Cloud Amount Adjustment\nAdjusted')

ax=axs[2]
C = B-A
ax.coastlines()
pl=ax.contourf(lons, lats, C, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
ax.set_title('Adjusted minus Original')

ax=axs[3]
A=MAPS['LO'].mean('modripf')['NETcld_amt']
ax.coastlines()
ax.contourf(lons, lats, A, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
ax.text(0, 0.5,'Low', fontsize=14, ha='right', va='center', rotation=90, transform=ax.transAxes) # no clue why set_ylabel doesn't work

ax=axs[4]
B=MAPS['trueLO'].mean('modripf')['NETcld_amt']
ax.coastlines()
ax.contourf(lons, lats, B, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')

ax=axs[5]
C = B-A
ax.coastlines()
pl=ax.contourf(lons, lats, C, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')

# Adjust layout
plt.tight_layout()#h_pad=0.0)

# Add a single colorbar at bottom
cbar = plt.colorbar(pl, ax=axs, orientation='horizontal', pad = 0.04, shrink = 0.5, aspect = 25)
cbar.set_label('W/m$^2$')


# In[ ]:


lons = MAPS['LO'].lon
lats = MAPS['LO'].lat
LEVELS1 = np.arange(0,1.3,0.1)
LEVELS2 = np.arange(-0.8,0.9,0.1)
modripfs=modripfs5+modripfs6
fig, axs = plt.subplots(2, 3, figsize=(12,6), subplot_kw={'projection': ccrs.Robinson(central_longitude=180)})
axs = axs.flatten()

ax=axs[0]
A=MAPS['HI'].std('modripf')['NETcld_amt']
ax.coastlines()
ax.contourf(lons, lats, A, transform=ccrs.PlateCarree(),levels=LEVELS1,extend='max')#,cmap='RdBu_r')
ax.text(0, 0.5,'Nonlow', fontsize=14, ha='right', va='center', rotation=90, transform=ax.transAxes) # no clue why set_ylabel doesn't work
ax.set_title('Original')

ax=axs[1]
B=MAPS['trueHI'].std('modripf')['NETcld_amt']
ax.coastlines()
ax.contourf(lons, lats, B, transform=ccrs.PlateCarree(),levels=LEVELS1,extend='max')#,cmap='RdBu_r')
ax.set_title('Across-Model Standard Deviation Cloud Amount Adjustment\nAdjusted')

ax=axs[2]
C = B-A
ax.coastlines()
pl=ax.contourf(lons, lats, C, transform=ccrs.PlateCarree(),levels=LEVELS2,extend='both',cmap='RdBu_r')
ax.set_title('Adjusted minus Original')

ax=axs[3]
A=MAPS['LO'].std('modripf')['NETcld_amt']
ax.coastlines()
ax.contourf(lons, lats, A, transform=ccrs.PlateCarree(),levels=LEVELS1,extend='max')#,cmap='RdBu_r')
ax.text(0, 0.5,'Low', fontsize=14, ha='right', va='center', rotation=90, transform=ax.transAxes) # no clue why set_ylabel doesn't work

ax=axs[4]
B=MAPS['trueLO'].std('modripf')['NETcld_amt']
ax.coastlines()
pl1=ax.contourf(lons, lats, B, transform=ccrs.PlateCarree(),levels=LEVELS1,extend='max')#,cmap='RdBu_r')

ax=axs[5]
C = B-A
ax.coastlines()
pl2=ax.contourf(lons, lats, C, transform=ccrs.PlateCarree(),levels=LEVELS2,extend='both',cmap='RdBu_r')

# Adjust layout
plt.tight_layout(h_pad=-7.5)

# Add a single horizontal colorbar for the first two columns
cbar=fig.colorbar(pl1, ax=axs[3:5], orientation='horizontal', pad = 0.04, shrink = 0.5, aspect = 25)
cbar.set_label('W/m$^2$')

# Add a horizontal colorbar for the last column below it
cbar=fig.colorbar(pl2, ax=axs[5], orientation='horizontal', pad = 0.04, aspect = 25)
cbar.set_label('W/m$^2$')


# In[ ]:


lons = MAPS['LO'].lon
lats = MAPS['LO'].lat
LEVELS = np.arange(-2.75,3.0,0.25)
for era in ['CMIP5','CMIP6']:
    if era=='CMIP5':
        modripfs=modripfs5
    else:
        modripfs=modripfs6
    LM = len(modripfs)
    fig, axs = plt.subplots(LM, 3, figsize=(12,2.5*LM), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    axs = axs.flatten()

    cnt=-1
    for m,mo in enumerate(modripfs):
        cnt+=1
        ax=axs[cnt]
        A=MAPS['HI'].sel(modripf=mo)['NETcld_amt']
        ax.coastlines()
        ax.contourf(lons, lats, A, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
        ax.text(-180, 0,mo, fontsize=10, ha='right', va='center', rotation=90) # no clue why set_ylabel doesn't work
        if m==0: # if we are in the first row:
            ax.set_title('Original')

        cnt+=1
        ax=axs[cnt]
        B=MAPS['trueHI'].sel(modripf=mo)['NETcld_amt']
        ax.coastlines()
        ax.contourf(lons, lats, B, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
        if m==0: # if we are in the first row:
            ax.set_title(era+' Nonlow Cloud Amount Adjustment\nObscuration-Adjusted')

        cnt+=1
        ax=axs[cnt]
        C = B-A
        ax.coastlines()
        pl=ax.contourf(lons, lats, C, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
        if m==0: # if we are in the first row:
            ax.set_title('Adjusted minus Original')

    # Adjust layout
    plt.tight_layout()#h_pad=0.2)

    # Add a single colorbar at bottom
    cbar = plt.colorbar(pl, ax=axs, orientation='horizontal', pad = 0.005, shrink = 0.5, aspect = 15)
    cbar.set_label('W/m$^2$')


# In[ ]:


fig, axs = plt.subplots(2, 1, figsize=(6,8), sharex=True)

A=MAPS['HI']['NETcld_amt'].mean('lon')
mean = A.mean('modripf')
std = A.std('modripf')
upper_bound = mean + std
lower_bound = mean - std
axs[0].plot(lats, mean, color='k', label='Original')
axs[0].fill_between(lats, upper_bound, lower_bound, color='k', alpha=0.2)
A=MAPS['trueHI']['NETcld_amt'].mean('lon')
mean = A.mean('modripf')
std = A.std('modripf')
upper_bound = mean + std
lower_bound = mean - std
axs[0].plot(lats, mean, color='C3', label='Obscuration-Adjusted')
axs[0].fill_between(lats, upper_bound, lower_bound, color='C3', alpha=0.2)
axs[0].set_xlim(-90,90)
axs[0].set_ylim(-1.5,1.5)
axs[0].axhline(y=0,ls='--',color='gray')
axs[0].legend()
axs[0].set_title('Nonlow Cloud Amount Adjustment',loc='left')

A=MAPS['LO']['NETcld_amt'].mean('lon')
mean = A.mean('modripf')
std = A.std('modripf')
upper_bound = mean + std
lower_bound = mean - std
axs[1].plot(lats, mean, color='k', label='Original')
axs[1].fill_between(lats, upper_bound, lower_bound, color='k', alpha=0.2)
A=MAPS['trueLO']['NETcld_amt'].mean('lon')
mean = A.mean('modripf')
std = A.std('modripf')
upper_bound = mean + std
lower_bound = mean - std
axs[1].plot(lats, mean, color='C3', label='Unobscured')
axs[1].fill_between(lats, upper_bound, lower_bound, color='C3', alpha=0.2)
axs[1].set_xlim(-90,90)
axs[0].set_ylim(-1.5,1.5)
axs[1].set_ylabel('W/m$^2$')
axs[1].set_xlabel('Latitude')
axs[1].axhline(y=0,ls='--',color='gray')
axs[1].legend()
axs[1].set_title('Low Cloud Amount Adjustment',loc='left')


# In[ ]:


lons = MAPS['LO'].lon
lats = MAPS['LO'].lat
LEVELS = np.arange(-2.75,3.0,0.25)
for era in ['CMIP5','CMIP6']:
    if era=='CMIP5':
        modripfs=modripfs5
    else:
        modripfs=modripfs6
    LM = len(modripfs)
    fig, axs = plt.subplots(LM, 3, figsize=(12,2.5*LM), subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)})
    axs = axs.flatten()

    cnt=-1
    for m,mo in enumerate(modripfs):
        cnt+=1
        ax=axs[cnt]
        A=MAPS['LO'].sel(modripf=mo)['NETcld_amt']
        ax.coastlines()
        ax.contourf(lons, lats, A, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
        ax.text(-180, 0,mo, fontsize=10, ha='right', va='center', rotation=90) # no clue why set_ylabel doesn't work
        if m==0: # if we are in the first row:
            ax.set_title('Original')

        cnt+=1
        ax=axs[cnt]
        B=MAPS['trueLO'].sel(modripf=mo)['NETcld_amt']
        ax.coastlines()
        ax.contourf(lons, lats, B, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
        if m==0: # if we are in the first row:
            ax.set_title(era+' Low Cloud Amount Adjustment\nUnobscured')

        cnt+=1
        ax=axs[cnt]
        C = B-A
        ax.coastlines()
        pl=ax.contourf(lons, lats, C, transform=ccrs.PlateCarree(),levels=LEVELS,extend='both',cmap='RdBu_r')
        if m==0: # if we are in the first row:
            ax.set_title('Unobscured minus Original')


    # Adjust layout
    plt.tight_layout()#h_pad=0.2)

    # Add a single colorbar at bottom
    cbar = plt.colorbar(pl, ax=axs, orientation='horizontal', pad = 0.005, shrink = 0.5, aspect = 15)
    cbar.set_label('W/m$^2$')


# In[ ]:


A=MAPS['HI']['NETcld_amt'].mean('lon')
B=MAPS['trueHI']['NETcld_amt'].mean('lon')
C1 = A-B
A=MAPS['LO']['NETcld_amt'].mean('lon')
B=MAPS['trueLO']['NETcld_amt'].mean('lon')
C2 = B-A
mean = C2.mean('modripf')
std = C2.std('modripf')
upper_bound = mean + std
lower_bound = mean - std
plt.plot(lats, C2.mean('modripf'), color='k', label='Low: Unobscured minus Original')
plt.plot(lats, C1.mean('modripf'), color='C3', ls='-', label='Nonlow: Original minus Adjusted')
plt.fill_between(lats, upper_bound, lower_bound, color='k', alpha=0.2)
plt.xlim(-90,90)
plt.ylim(-0.6,1.2)
plt.ylabel('W/m$^2$')
plt.xlabel('Latitude')
plt.axhline(y=0,ls='--',color='gray')
plt.legend()
plt.title('Cloud Amount Adjustment Adjustments',loc='left')


# In[ ]:


x,y = AVGS['trueLO'],AVGS['LO']
plt.plot(x['SWcld_tau'],y['SWcld_tau'],'o',label='tau')
plt.plot(x['SWcld_amt'],y['SWcld_amt'],'o',label='amt')
plt.plot(x['SWcld_err'],y['SWcld_err'],'o',label='err')
dummy=np.linspace(-0.25,0.8,10)
plt.plot(dummy,dummy,color='grey',ls='--')
plt.legend()
plt.xlabel('unobscured')
plt.ylabel('original')


# In[ ]:


pd.DataFrame(AVGS['LO'])


# In[ ]:


print(AVGS['trueLO']['SWcld_amt'].std(),AVGS['LO']['SWcld_amt'].std())
print(AVGS['trueHI']['SWcld_amt'].std(),AVGS['HI']['SWcld_amt'].std())
print(AVGS['trueLO']['NETcld_amt'].std(),AVGS['LO']['NETcld_amt'].std())
print(AVGS['trueHI']['NETcld_amt'].std(),AVGS['HI']['NETcld_amt'].std())


# In[ ]:


plt.figure(figsize=(6,9))
plt.suptitle('Nonlow Cloud Adjustments',fontsize=16,x=0.11,y=0.95,ha='left')
bands = ['LW','NET','SW']
flavors = ['tot','amt','alt','tau','err']
cnt=0
handles = []
labels = []
for b,band in enumerate(bands):
    cnt+=1
    plt.subplot(3,1,cnt)
    plt.axhline(y=0,color='gray',ls='--')
    for f,flav in enumerate(flavors):
        orig = AVGS['HI'][band+'cld_'+flav]
        adj = AVGS['trueHI'][band+'cld_'+flav]
        handle_orig, = plt.plot(-0.1+f*np.ones(len(orig)),orig,marker='o',ls='',mec='k',mfc='k',alpha=0.25)
        handle_orig, = plt.plot(-0.1+f*np.ones(len(orig)),orig,marker='o',ls='',mec='k',mfc='None')
        # plt.plot(-0.1+f,np.average(orig),marker='o',ls='',mfc='k',mec='k')#,ms=10)
        plt.bar(-0.1+f,np.average(orig),width=0.2,edgecolor='k',facecolor='None',lw=2)
        handle_adj, = plt.plot(0.1+f*np.ones(len(adj)),adj,marker='o',ls='',mec='C3',mfc='C3',alpha=0.25)
        handle_adj, = plt.plot(0.1+f*np.ones(len(adj)),adj,marker='o',ls='',mec='C3',mfc='None')
        # plt.plot(0.1+f,np.average(adj),marker='o',ls='',mfc='C3',mec='C3')#,ms=10)
        plt.bar(0.1+f,np.average(adj),width=0.2,edgecolor='C3',facecolor='None',lw=2)
    plt.title(band,loc='left')
    plt.ylim(-1.5,2.5)
    if cnt==3:
        plt.xticks(range(f+1),flavors)
    else:
        plt.xticks(range(f+1),'')
        
# Add handles and labels for the legend
handles.extend([handle_orig, handle_adj])
labels.extend(['original', 'obscuration-adjusted'])
# Display the legend with only the first two entries
plt.legend(handles[:2], labels[:2])


# In[ ]:


plt.figure(figsize=(6,9))
plt.suptitle('Low Cloud Adjustments',fontsize=16,x=0.11,y=0.95,ha='left')
bands = ['LW','NET','SW']
flavors = ['tot','amt','alt','tau','err']
cnt=0
handles = []
labels = []
for b,band in enumerate(bands):
    cnt+=1
    plt.subplot(3,1,cnt)
    plt.axhline(y=0,color='gray',ls='--')
    for f,flav in enumerate(flavors):
        orig = AVGS['LO'][band+'cld_'+flav]
        adj = AVGS['trueLO'][band+'cld_'+flav]
        handle_orig, = plt.plot(-0.1+f*np.ones(len(orig)),orig,marker='o',ls='',mec='k',mfc='k',alpha=0.25)
        handle_orig, = plt.plot(-0.1+f*np.ones(len(orig)),orig,marker='o',ls='',mec='k',mfc='None')
        # plt.plot(-0.1+f,np.average(orig),marker='o',ls='',mfc='k',mec='k')#,ms=10)
        plt.bar(-0.1+f,np.average(orig),width=0.2,edgecolor='k',facecolor='None',lw=2)
        handle_adj, = plt.plot(0.1+f*np.ones(len(adj)),adj,marker='o',ls='',mec='C3',mfc='C3',alpha=0.25)
        handle_adj, = plt.plot(0.1+f*np.ones(len(adj)),adj,marker='o',ls='',mec='C3',mfc='None')
        # plt.plot(0.1+f,np.average(adj),marker='o',ls='',mfc='C3',mec='C3')#,ms=10)
        plt.bar(0.1+f,np.average(adj),width=0.2,edgecolor='C3',facecolor='None',lw=2)
    plt.title(band,loc='left')
    plt.ylim(-1,1)
    if cnt==3:
        plt.xticks(range(f+1),flavors)
    else:
        plt.xticks(range(f+1),'')
        
# Add handles and labels for the legend
handles.extend([handle_orig, handle_adj])
labels.extend(['original', 'obscuration-adjusted'])
# Display the legend with only the first two entries
plt.legend(handles[:2], labels[:2])


# In[ ]:


def stacked_barplot(x,y,negavg,posavg, **kwargs):#,WIDTH,COLOR,ALPHA,ZORDER,LABEL):
    if y<0:
        p1 = plt.bar(x, y, bottom=negavg, **kwargs)#, width=WIDTH, color=COLOR, alpha=ALPHA, zorder=ZORDER, label=LABEL)
        negavg+=y
    else:        
        p1 = plt.bar(x, y, bottom=posavg, **kwargs)#, width=WIDTH, color=COLOR, alpha=ALPHA, zorder=ZORDER, label=LABEL)
        posavg+=y
    return(negavg,posavg)

def stacked_barhplot(x,y,negavg,posavg, **kwargs):#,WIDTH,COLOR,ALPHA,ZORDER,LABEL):
    if y<0:
        p1 = plt.barh(x, y, left=negavg, **kwargs)#, width=WIDTH, color=COLOR, alpha=ALPHA, zorder=ZORDER, label=LABEL)
        negavg+=y
    else:        
        p1 = plt.barh(x, y, left=posavg, **kwargs)#, width=WIDTH, color=COLOR, alpha=ALPHA, zorder=ZORDER, label=LABEL)
        posavg+=y
    return(negavg,posavg)


# In[ ]:


modripfs=modripfs5+modripfs6

fig, axs = plt.subplots(1,2, figsize=(12, 6), sharey=True)

plt.sca(axs[0])
# unobscured low cloud Adjustment = original low cloud Adjustment + change in obscuration
orig = AVGS['LO']['NETcld_amt']
adj_tot = AVGS['trueLO']['NETcld_amt']
dobcs = adj_tot-orig
inds = np.argsort(dobcs)
# inds = np.argsort(adj_tot)
mo=[]
for i,m in enumerate(inds):
    mo.append(modripfs[m])
    posavg=0
    negavg=0   
    if i==0:
        negavg,posavg = stacked_barhplot(i,orig[m],negavg,posavg,height=0.7,color='k',alpha=1,zorder=10,label='Original')    
        negavg,posavg = stacked_barhplot(i,dobcs[m],negavg,posavg,height=0.7,color='gray',alpha=1,zorder=10,label='$\Delta$Obscuration')    
        plt.plot(adj_tot[m],i,'o',color='C3',ms=10,zorder=20,label='Obscuration-Adjusted')
    else:
        negavg,posavg = stacked_barhplot(i,orig[m],negavg,posavg,height=0.7,color='k',alpha=1,zorder=10,label='_nolegend_')
        negavg,posavg = stacked_barhplot(i,dobcs[m],negavg,posavg,height=0.7,color='gray',alpha=1,zorder=10,label='_nolegend_')
        plt.plot(adj_tot[m],i,'o',color='C3',ms=10,zorder=20,label='_nolegend_')
plt.axvline(x=0,ls='--',color='gray')    
_=plt.yticks(np.arange(len(mo)),mo)
plt.title('Low Cloud Amount Adjustment',loc='left')
plt.xlabel('W/m$^2$')
plt.xlim(-1,1)
plt.grid(axis='y')

plt.sca(axs[1])
# obscuration-adjusted high cloud amount Adjustment = original high cloud amount Adjustment + change in obscuration:
orig = AVGS['HI']['NETcld_amt']
adj_tot = AVGS['trueHI']['NETcld_amt']
dobcs = adj_tot - orig
mo=[]
for i,m in enumerate(inds):
    mo.append(modripfs[m])
    posavg=0
    negavg=0   
    if i==0:
        negavg,posavg = stacked_barhplot(i,orig[m],negavg,posavg,height=0.7,color='k',alpha=1,zorder=10,label='Original')    
        negavg,posavg = stacked_barhplot(i,dobcs[m],negavg,posavg,height=0.7,color='gray',alpha=1,zorder=10,label='$\Delta$Obscuration')    
        plt.plot(adj_tot[m],i,'o',color='C3',ms=10,zorder=20,label='Obscuration-Adjusted')
    else:
        negavg,posavg = stacked_barhplot(i,orig[m],negavg,posavg,height=0.7,color='k',alpha=1,zorder=10,label='_nolegend_')
        negavg,posavg = stacked_barhplot(i,dobcs[m],negavg,posavg,height=0.7,color='gray',alpha=1,zorder=10,label='_nolegend_')
        plt.plot(adj_tot[m],i,'o',color='C3',ms=10,zorder=20,label='_nolegend_')
plt.axvline(x=0,ls='--',color='gray')    
_=plt.yticks(np.arange(len(mo)),mo)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Nonlow Cloud Amount Adjustment',loc='left')
plt.xlabel('W/m$^2$')
plt.xlim(-1,1)
plt.grid(axis='y')


# In[ ]:


fig, axs = plt.subplots(1,2, figsize=(12, 6), sharey=True)

plt.sca(axs[0])
# unobscured low cloud Adjustment = original low cloud Adjustment + change in obscuration
orig = AVGS['LO']['NETcld_tot']
adj_tot = AVGS['trueLO']['NETcld_tot']
dobcs = adj_tot-orig
inds = np.argsort(dobcs)
# inds = np.argsort(adj_tot)
mo=[]
for i,m in enumerate(inds):
    mo.append(modripfs[m])
    posavg=0
    negavg=0   
    if i==0:
        negavg,posavg = stacked_barhplot(i,orig[m],negavg,posavg,height=0.7,color='k',alpha=1,zorder=10,label='Original')    
        negavg,posavg = stacked_barhplot(i,dobcs[m],negavg,posavg,height=0.7,color='gray',alpha=1,zorder=10,label='$\Delta$Obscuration')    
        plt.plot(adj_tot[m],i,'o',color='C3',ms=10,zorder=20,label='Obscuration-Adjusted')
    else:
        negavg,posavg = stacked_barhplot(i,orig[m],negavg,posavg,height=0.7,color='k',alpha=1,zorder=10,label='_nolegend_')
        negavg,posavg = stacked_barhplot(i,dobcs[m],negavg,posavg,height=0.7,color='gray',alpha=1,zorder=10,label='_nolegend_')
        plt.plot(adj_tot[m],i,'o',color='C3',ms=10,zorder=20,label='_nolegend_')
plt.axvline(x=0,ls='--',color='gray')    
_=plt.yticks(np.arange(len(mo)),mo)
plt.title('Low Cloud Adjustment',loc='left')
plt.xlabel('W/m$^2$')
plt.xlim(-1,1)
plt.grid(axis='y')

plt.sca(axs[1])
# obscuration-adjusted high cloud amount Adjustment = original high cloud amount Adjustment + change in obscuration:
orig = AVGS['HI']['NETcld_tot']
adj_tot = AVGS['trueHI']['NETcld_tot']
dobcs = adj_tot - orig
mo=[]
for i,m in enumerate(inds):
    mo.append(modripfs[m])
    posavg=0
    negavg=0   
    if i==0:
        negavg,posavg = stacked_barhplot(i,orig[m],negavg,posavg,height=0.7,color='k',alpha=1,zorder=10,label='Original')    
        negavg,posavg = stacked_barhplot(i,dobcs[m],negavg,posavg,height=0.7,color='gray',alpha=1,zorder=10,label='$\Delta$Obscuration')    
        plt.plot(adj_tot[m],i,'o',color='C3',ms=10,zorder=20,label='Obscuration-Adjusted')
    else:
        negavg,posavg = stacked_barhplot(i,orig[m],negavg,posavg,height=0.7,color='k',alpha=1,zorder=10,label='_nolegend_')
        negavg,posavg = stacked_barhplot(i,dobcs[m],negavg,posavg,height=0.7,color='gray',alpha=1,zorder=10,label='_nolegend_')
        plt.plot(adj_tot[m],i,'o',color='C3',ms=10,zorder=20,label='_nolegend_')
plt.axvline(x=0,ls='--',color='gray')    
_=plt.yticks(np.arange(len(mo)),mo)
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.title('Nonlow Cloud Adjustment',loc='left')
plt.xlabel('W/m$^2$')
plt.xlim(-1,1)
plt.grid(axis='y')


# In[ ]:


# The original high and low cloud amount Adjustments must be anti-correlated 
# such that the adjusted ones can both has less intermodel spread
from scipy.stats import linregress
hi_orig = AVGS['HI']['SWcld_amt']
lo_orig = AVGS['LO']['SWcld_amt']
hi_adj = AVGS['trueHI']['SWcld_amt']
lo_adj = AVGS['trueLO']['SWcld_amt']
x,y,lab = lo_orig,hi_orig,'Original'
m, b, r, p, std_err = linregress(x,y)
if p<0.05:
    LABEL = lab+'; r='+str(np.round(r,2))+'*'
else:
    LABEL = lab+'; r='+str(np.round(r,2))
dummy=np.linspace(np.nanmin(x),np.nanmax(x),100)
plt.plot(dummy,m*dummy+b,'k',label=LABEL)    
plt.plot(x,y,marker='o',ms=10,ls='',mec='k',mfc='k',alpha=0.25)
plt.plot(x,y,marker='o',ms=10,ls='',mec='k',mfc='None')

x,y,lab = lo_adj,hi_adj,'Adjusted'
m, b, r, p, std_err = linregress(x,y)
if p<0.05:
    LABEL = lab+'; r='+str(np.round(r,2))+'*'
else:
    LABEL = lab+'; r='+str(np.round(r,2))
dummy=np.linspace(np.nanmin(x),np.nanmax(x),100)
plt.plot(dummy,m*dummy+b,'C3',label=LABEL) 
plt.plot(x,y,marker='o',ms=10,ls='',mec='C3',mfc='C3',alpha=0.25)
plt.plot(x,y,marker='o',ms=10,ls='',mec='C3',mfc='None')   

plt.legend()
# plt.ylim(-0.15,0.2)
plt.ylabel('Nonlow [W/m$^2$]')
plt.xlabel('Low [W/m$^2$]')
plt.title('SW Cloud Amount Adjustment',loc='left')

# plt.grid()


# In[ ]:


# The original high and low cloud amount Adjustments must be anti-correlated 
# such that the adjusted ones can both has less intermodel spread
from scipy.stats import linregress
hi_orig = AVGS['HI']['NETcld_amt']
lo_orig = AVGS['LO']['NETcld_amt']
hi_adj = AVGS['trueHI']['NETcld_amt']
lo_adj = AVGS['trueLO']['NETcld_amt']
x,y,lab = lo_orig,hi_orig,'Original'
m, b, r, p, std_err = linregress(x,y)
if p<0.05:
    LABEL = lab+'; r='+str(np.round(r,2))+'*'
else:
    LABEL = lab+'; r='+str(np.round(r,2))
dummy=np.linspace(np.nanmin(x),np.nanmax(x),100)
plt.plot(dummy,m*dummy+b,'k',label=LABEL)    
plt.plot(x,y,marker='o',ms=10,ls='',mec='k',mfc='k',alpha=0.25)
plt.plot(x,y,marker='o',ms=10,ls='',mec='k',mfc='None')

x,y,lab = lo_adj,hi_adj,'Adjusted'
m, b, r, p, std_err = linregress(x,y)
if p<0.05:
    LABEL = lab+'; r='+str(np.round(r,2))+'*'
else:
    LABEL = lab+'; r='+str(np.round(r,2))
dummy=np.linspace(np.nanmin(x),np.nanmax(x),100)
plt.plot(dummy,m*dummy+b,'C3',label=LABEL) 
plt.plot(x,y,marker='o',ms=10,ls='',mec='C3',mfc='C3',alpha=0.25)
plt.plot(x,y,marker='o',ms=10,ls='',mec='C3',mfc='None')   

plt.legend()
plt.ylim(-0.15,0.2)
plt.ylabel('Nonlow [W/m$^2$]')
plt.xlabel('Low [W/m$^2$]')
plt.title('Net Cloud Amount Adjustment',loc='left')

# plt.grid()


# In[ ]:


orig = AVGS['LO']['NETcld_tot']
adj_tot = AVGS['trueLO']['NETcld_tot']
lo_dobcs = adj_tot-orig
orig = AVGS['HI']['NETcld_tot']
adj_tot = AVGS['trueHI']['NETcld_tot']
hi_dobcs = adj_tot - orig
plt.plot(hi_dobcs,lo_dobcs,'o')
dummy=np.linspace(-0.25,0.25,100)
plt.plot(dummy,-dummy,ls='--')


# In[ ]:





# In[ ]:




