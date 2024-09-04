#!/usr/bin/env python
# coding: utf-8

# ## Description
# My local version of the cloud feedbacks code, intended to investigate the implications of the obscuration adjustments

# ## Import useful functions

# In[1]:


import cal_CloudRadKernel_xr as CRK
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import xsearch as xs
import xarray as xr
import numpy as np
import warnings
import copy
import os
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


# ## Get filepaths based on user input

# In[2]:


freq='mon' 
realm = 'atmos'
eras = ['CMIP5','CMIP6']
xs_paths = {}
variables = ['tas','rsdscs','rsuscs','clisccp'] # necessary for cloud feedback calcs
for era in eras:
    xs_paths[era] = {}
    if era=='CMIP5':
        exps = ['amip','amip4K']
    elif era=='CMIP6':
        exps = ['amip','amip-p4K']
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


# In[3]:


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


# ## Do all the cloud feedback calculations

# In[4]:


def clean_up(output,obsc_output,era,model,member,exps):
    
    trueHI={}
    bands=['LW','SW','NET']
    fields = ['tot','amt','alt','tau','err']
    for field in fields:
        for band in bands:
            trueHI[band+'cld_'+field] = copy.deepcopy(output['HI680'][band+'cld_'+field])

    LOobsc = obsc_output['LO680']
    # update the high cloud amount feedback to include the change in obscuration term:
    trueHI['LWcld_amt'] += LOobsc['LWdobsc_fbk']
    trueHI['SWcld_amt'] += LOobsc['SWdobsc_fbk']
    trueHI['NETcld_amt'] = trueHI['LWcld_amt'] + trueHI['SWcld_amt']
    # update the total high cloud feedback:
    trueHI['LWcld_tot'] = trueHI['LWcld_amt'] + trueHI['LWcld_alt'] + trueHI['LWcld_tau'] + trueHI['LWcld_err']
    trueHI['SWcld_tot'] = trueHI['SWcld_amt'] + trueHI['SWcld_alt'] + trueHI['SWcld_tau'] + trueHI['SWcld_err']
    trueHI['NETcld_tot'] = trueHI['LWcld_tot'] + trueHI['SWcld_tot']

    LOobsc = obsc_output['LO680']
    # SWcld_tot is equivalent to SWdunobsc_fbk; ditto for LW compoent
    # SWdobsc_fbk is already incorporated into high cloud amount feedback; ditto for LW component

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
    # update the total low cloud feedback:
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
            path = '/p/user_pub/climate_work/zelinka1/cmip5/amip4K/'
        else:
            path = '/p/user_pub/climate_work/zelinka1/cmip6/amip-p4K/'
        savefile2 = path+model+'.'+member+'.'+name+'_fbks.nc'
        if os.path.exists(savefile2):
            os.remove(savefile2)
            
        DS['time'] = np.arange(12)
        if 'albcs' in DS.coords:
            DS=DS.reset_coords(names='albcs', drop=True)
        if 'height' in DS.coords:
            DS=DS.reset_coords(names='height', drop=True)
        DS.coords    
        DS.to_netcdf(savefile2)
        print('Saved '+savefile2)

    return(ALL,LO,HI,trueLO,trueHI)


# In[5]:


# NEED to deal CCSM4 and other models that have mis-match between rips of amip and amip4K
# also HadGEM2-A
AVGS={}
for era in eras[1:]:
    models = list(filepath[era].keys())
    for model in models:
        if model=='HadGEM2-A':
            continue
        members = list(filepath[era][model].keys())
        for member in members:
            if model=='IPSL-CM5A-LR' and member=='r2i1p1':
                continue
            if model=='CNRM-CM6-1' and member=='r1i1p1f2':
                continue   
            if model=='GISS-E2-1-G' and member=='r1i1p1f1':
                continue   
            if model=='HadGEM3-GC31-LL' and member=='r5i1p1f3':
                continue   
            if model=='IPSL-CM6A-LR': # plev isn't called plev
                continue   
                
                
            if era=='CMIP5':
                path = '/p/user_pub/climate_work/zelinka1/cmip5/amip4K/'
            else:
                path = '/p/user_pub/climate_work/zelinka1/cmip6/amip-p4K/'
            savefile2 = path+model+'.'+member+'.ALL_fbks.nc' 
            # if os.path.exists(savefile2): 
            #     print('Already saved '+model+'.'+member)
            #     continue              
            
            cnt=0
            exps=list(filepath[era][model][member].keys())
            for exp in exps:
                variables=list(filepath[era][model][member][exp].keys())
                for var in variables:
                    cnt+=1
            if cnt<8:
                print('Skipping '+model+'.'+member+' -- not enough fields')
                continue
                                
            print('Working on '+model+'.'+member)
            (output,obsc_output) = CRK.CloudRadKernel(filepath[era][model][member]) 
            
            # Finalize the calculations and save netcdfs
            (ALL,LO,HI,trueLO,trueHI) = clean_up(output,obsc_output,era,model,member,exps)

            print('Done with '+model+'.'+member)


# In[ ]:


model,member
# ds=output['ALL']['LWcld_tot']
# ds1=ds.reset_coords(names='albcs', drop=True)
# ds2=ds1.reset_coords(names='height', drop=True)
# ds2.coords


# In[ ]:


stop


# In[ ]:


ds=xr.open_mfdataset('/p/user_pub/climate_work/zelinka1/cmip*/ami*4K/*.trueHI_fbks.nc', concat_dim = 'modripf', combine = 'nested')
ds
# ds.mean('time')['LWcld_alt'].plot(col='modripf')


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


# In[10]:


# ds=xr.open_dataset('/p/user_pub/climate_work/zelinka1/cmip5/amip4K/MIROC5.r1i1p1.ALL_fbks.nc')
ds=xr.open_dataset('/p/user_pub/climate_work/zelinka1/cmip5/amip4K/CanAM4.r1i1p1.ALL_fbks.nc')
ds


# In[7]:


import glob
AVGS={}
MAPS={}
DSnames = ['ALL','LO','HI','trueLO','trueHI']
for d,DS in enumerate([ALL,LO,HI,trueLO,trueHI]):
    sec = DSnames[d]
    AVGS[sec]={}
    MAPS[sec]={}
    for era in eras:
        if era=='CMIP5':
            path = '/p/user_pub/climate_work/zelinka1/cmip5/amip4K/'
        else:
            path = '/p/user_pub/climate_work/zelinka1/cmip6/amip-p4K/'
        gg=glob.glob(path+'*.'+sec+'_fbks.nc')
        for g in gg:
            model = g.split('/')[-1].split('.')[0]
            member = g.split('/')[-1].split('.')[1]
            ds = xr.open_dataset(g)
            print(g)
            print(ds.coords)
            # # compute global averages:
            # AVGS[sec][model+'.'+member] = global_avgs(ds)
            # MAPS[sec][model+'.'+member] = ds.mean('time')


# In[ ]:


plt.figure(figsize=(12,18))
modripfs = list(MAPS[sec].keys())
cnt=0
for modripf in modripfs:
    cnt+=1
    plt.subplot(16,3,cnt)
    A=MAPS['LO'][modripf]['NETcld_amt']
    A.plot()
    plt.title(modripf+' low cloud amount')
    cnt+=1
    plt.subplot(16,3,cnt)
    B=MAPS['trueLO'][modripf]['NETcld_amt']
    B.plot()
    plt.title(modripf+' unobscured low cloud amount')
    cnt+=1
    plt.subplot(16,3,cnt)
    C = B-A
    C.plot()
    plt.title(modripf+' b-a')


# In[ ]:


pd.DataFrame(AVGS[sec])


# In[ ]:


plt.figure(figsize=(9,9))
plt.suptitle('Low Cloud Feedbacks',fontsize=16,x=0.11,y=0.95,ha='left')
modripfs=list(AVGS[sec].keys())
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
        orig,adj=[],[]
        for mo in modripfs:            
            orig.append(float(np.array(AVGS['LO'][mo][band+'cld_'+flav])))
            adj.append(float(np.array(AVGS['trueLO'][mo][band+'cld_'+flav])))
        handle_orig, = plt.plot(f*np.ones(len(orig)),orig,marker='o',ls='',mec='k',mfc='None')
        plt.plot(f,np.average(orig),marker='o',ls='',mfc='k',mec='k',ms=10)
        handle_adj, = plt.plot(0.15+f*np.ones(len(adj)),adj,marker='o',ls='',mec='C3',mfc='None')
        plt.plot(0.15+f,np.average(adj),marker='o',ls='',mfc='C3',mec='C3',ms=10)
    plt.title(band,loc='left')
    plt.ylim(-0.5,1)
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


plt.figure(figsize=(9,9))
plt.suptitle('High Cloud Feedbacks',fontsize=16,x=0.11,y=0.95,ha='left')
modripfs=list(AVGS[sec].keys())
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
        orig,adj=[],[]
        for mo in modripfs:            
            orig.append(float(np.array(AVGS['HI'][mo][band+'cld_'+flav])))
            adj.append(float(np.array(AVGS['trueHI'][mo][band+'cld_'+flav])))
        handle_orig, = plt.plot(f*np.ones(len(orig)),orig,marker='o',ls='',mec='k',mfc='None')
        plt.plot(f,np.average(orig),marker='o',ls='',mfc='k',mec='k',ms=10)
        handle_adj, = plt.plot(0.15+f*np.ones(len(adj)),adj,marker='o',ls='',mec='C3',mfc='None')
        plt.plot(0.15+f,np.average(adj),marker='o',ls='',mfc='C3',mec='C3',ms=10)
    plt.title(band,loc='left')
    if band=='LW':
        plt.ylim(-0.5,1)
    elif band=='NET':
        plt.ylim(-0.75,0.75)
    elif band=='SW':
        plt.ylim(-1,0.5)
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




