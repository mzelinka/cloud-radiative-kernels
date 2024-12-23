#!/usr/bin/env python
# coding: utf-8

# =============================================
# Performs the cloud feedback and cloud error
# metric calculations in preparation for comparing
# to expert-assessed values from Sherwood et al (2020)
# =============================================

# IMPORT STUFF:
import glob
import cftime
import xarray as xr
import xcdat as xc
import numpy as np
from datetime import date

# =============================================
# define necessary information
# =============================================

# datadir = "../../data/"
datadir = "/home/zelinka1/git/cloud-radiative-kernels/data/"

# Define a python dictionary containing the sections of the histogram to consider
# These are the same as in Zelinka et al, GRL, 2016
sections = ["ALL", "HI680", "LO680"]
Psections = [slice(0, 7), slice(2, 7), slice(0, 2)]
sec_dic = dict(zip(sections, Psections))

# Define the grid
LAT = np.arange(-89,91,2.0)
LON = np.arange(1.25,360,2.5)
lat_axis = xc.create_axis("lat", LAT)
lon_axis = xc.create_axis("lon", LON)
output_grid = xc.create_grid(x=lon_axis, y=lat_axis)

###########################################################################
def get_amip_data(filepath,exp,var, lev=None):
    # load in cmip data using the appropriate function for the experiment/mip
    print('  '+var)#, end=",")

    filename = filepath[exp]
    if 'amip' in exp:
        tslice = slice("1983-01", "2008-12")  # we only want this portion of the amip run (overlap with all AMIPs and ISCCP)
    elif 'piClim' in exp: # for the piClim-4xCO2 or sstClim4xCO2 exps:
        tslice = slice(None,None)
    elif 'sstClim' in exp:
        tslice = slice(None,None)

    if var=='clisccp' and 'GISS-E2-1-G' in filename[var] and 'amip' in filename[var]:
        print('Dealing with the bad clisccp metadata in GISS-E2-1-G')
        # Load in tas data with correct metadata from same exp:
        tas = xc.open_mfdataset(filename['tas'])
        # Manually load in each month's clisccp field, then concatenate
        gg=glob.glob(filename['clisccp']+'*nc')
        gg.sort()
        ds0 = []    
        for g in gg:
            ds0.append(xc.open_dataset(g))
        ds=xr.concat(ds0, 'time')
        # Give it the correct time stamps from the tas field above
        f = ds.assign_coords(time=tas.sel(time=slice('1979-01','2014-12')).time)
        
    else:
        f = xc.open_mfdataset(filename[var], chunks={"lat": -1, "lon": -1, "time": -1}).load()
    
    
    if lev:
        f = f.sel(time=tslice, plev=lev)
        f = f.drop_vars(["plev", "plev_bnds"])
    else:
        f = f.sel(time=tslice)

    if 'plev7' in f.coords: # CNRM-CM6-1
        f = f.rename({'plev7':'plev'})
    if 'pressure2' in f.coords: # IPSL
        f = f.rename({'pressure2':'plev'})
        
    # prevent xarray from thinking tau is the Z axis
    try:
        if 'tau' in f.coords:
            del f.tau.attrs['positive'] 
    except:
        pass
    
    # Compute climatological monthly means
    f=f.bounds.add_missing_bounds()
    avg = f.temporal.climatology(var, freq="month", weighted=True)
    # Regrid to cloud kernel grid
    output = avg.regridder.horizontal(
        var, output_grid, tool="xesmf", method="bilinear", extrap_method="inverse_dist"
    )

    return output


###########################################################################
def get_model_data(filepath):
    # Read in data, regrid
    
    # input filepath is a dictionary containing paths to the data, filepath[exp][var] 

    # Load in regridded monthly mean climatologies from control and perturbed simulation
    variables = ["tas", "rsdscs", "rsuscs", "clisccp"]
    exps = list(filepath.keys())
    exp = exps[0]
    print(exp)    
    ctl = []
    for var in variables:
        ctl.append(get_amip_data(filepath,exp,var))
    ctl = xr.merge(ctl)

    exp = exps[1]
    print(exp)
    fut = []
    for var in variables:
        fut.append(get_amip_data(filepath,exp,var))
    fut = xr.merge(fut)
    
    # set tau,plev to consistent field
    ctl["clisccp"] = ctl["clisccp"].transpose("time", "tau", "plev", "lat", "lon")
    fut["clisccp"] = fut["clisccp"].transpose("time", "tau", "plev", "lat", "lon")
    ctl["tau"] = np.arange(7)
    ctl["plev"] = np.arange(7)
    fut["tau"] = np.arange(7)
    fut["plev"] = np.arange(7)

    # Make sure clisccp is in percent
    sumclisccp1 = ctl["clisccp"].sum(dim=["tau", "plev"])
    if np.max(sumclisccp1) <= 1.0:
        ctl["clisccp"] = ctl["clisccp"] * 100.0
        fut["clisccp"] = fut["clisccp"] * 100.0

    return (ctl, fut)


###########################################################################
def get_CRK_data(filepath):
    # Read in data, regrid

    # Load in regridded monthly mean climatologies from control and perturbed simulation
    print("get data")
    ctl, fut = get_model_data(filepath)
    print("get LW and SW kernel")
    CRKs = get_kernel_regrid(ctl)

    # global mean and annual average delta tas
    avgdtas0 = fut["tas"] - ctl["tas"]
    avgdtas0 = xc_to_dataset(avgdtas0)
    avgdtas0 = avgdtas0.spatial.average("data", axis=["X", "Y"])["data"]
    dTs = avgdtas0.mean()

    return (ctl.clisccp,fut.clisccp,CRKs,dTs)


###########################################################################
def get_kernel_regrid(ctl):
    # Read in data and map kernels to lat/lon
    
    f = xc.open_mfdataset(datadir + "cloud_kernels2.nc", decode_times=False)
    f = f.rename({"mo": "time", "tau_midpt": "tau", "p_midpt": "plev"})
    # f["time"] = ctl["time"].copy() 
    f["tau"] = np.arange(7)
    f["plev"] = np.arange(7)  # set tau,plev to consistent field
    LWkernel = f["LWkernel"].isel(albcs=0).squeeze()  # two kernels file are different
    SWkernel = f["SWkernel"]
    del f

    # Compute clear-sky surface albedo
    ctl_albcs = ctl.rsuscs / ctl.rsdscs  # (12, 90, 144)
    ctl_albcs = ctl_albcs.fillna(0.0)
    ctl_albcs = ctl_albcs.where(~np.isinf(ctl_albcs), 0.0)
    ctl_albcs = xr.where(ctl_albcs > 1.0, 1, ctl_albcs)  # where(condition, x, y) is x where condition is true, y otherwise
    ctl_albcs = xr.where(ctl_albcs < 0.0, 0, ctl_albcs)

    # LW kernel does not depend on albcs, just repeat the final dimension over longitudes:
    LWK = LWkernel.expand_dims(dim=dict(lon=LON), axis=4)

    # Use control albcs to map SW kernel to appropriate longitudes
    SWK = map_SWkern_to_lon(SWkernel, LWK, ctl_albcs)

    # need to reshape and re-assign the times to match clisccp, in case clisccp.time is length n*12:
    mimic = ctl['clisccp']
    repyrs=int(len(mimic.time)/12)
    repLWK = np.tile(LWK.values, (repyrs, 1,1,1,1))
    repSWK = np.tile(SWK.values, (repyrs, 1,1,1,1))
    CRKs = xr.Dataset({
        'LWkernel':(('time','tau','plev','lat','lon'),repLWK),
        'SWkernel':(('time','tau','plev','lat','lon'),repSWK)
        },
        coords={'time': mimic.time,'tau': mimic.tau,'plev': mimic.plev,'lat': mimic.lat,'lon': mimic.lon})    
    CRKs = CRKs.bounds.add_missing_bounds()

    return CRKs


###########################################################################
def map_SWkern_to_lon(SWkernel, LWK, albcsmap):
    """revised from zelinka_analysis.py"""

    from scipy.interpolate import interp1d

    ## Map each location's clear-sky surface albedo to the correct albedo bin
    # Ksw is size 12,7,7,lats,3
    # albcsmap is size 12,lats,lons
    albcs = np.arange(0.0, 1.5, 0.5)
    nanarray = np.nan * np.ones(LWK.shape)
    SWkernel_map = LWK.copy(data=nanarray)

    for T in range(len(LWK.time)):
        for LAT in range(len(LWK.lat)):
            alon = albcsmap[T, LAT, :].copy()  # a longitude array
            if sum(~np.isnan(alon)) >= 1:  # at least 1 unmasked value
                if len(SWkernel[T, :, :, LAT, :] > 0) == 0:
                    SWkernel_map[T, :, :, LAT, :] = 0
                else:
                    f = interp1d(albcs, SWkernel[T, :, :, LAT, :], axis=2)
                    SWkernel_map[T, :, :, LAT, :] = f(alon.values)
            else:
                continue

    return SWkernel_map


###########################################################################
def compute_fbk(ctl, fut, DT):
    DR = fut - ctl
    fbk = DR / DT
    baseline = ctl
    return fbk, baseline


###########################################################################
def KT_decomposition_general(c1, c2, Klw, Ksw):
    """
    this function takes in a (month,TAU,CTP,lat,lon) matrix and performs the
    decomposition of Zelinka et al 2013 doi:10.1175/JCLI-D-12-00555.1
    """

    sum_c = c1.sum(dim=["tau", "plev"])  # Eq. B2
    dc = c2 - c1
    sum_dc = dc.sum(dim=["tau", "plev"])
    dc_prop = c1 * (sum_dc / sum_c)
    dc_star = dc - dc_prop  # Eq. B1
    C_ratio = c1 / sum_c

    # LW components
    Klw0 = (Klw * c1 / sum_c).sum(dim=["tau", "plev"])  # Eq. B4
    Klw_prime = Klw - Klw0  # Eq. B3
    Klw_p_prime = (Klw_prime * (C_ratio.sum(dim="plev"))).sum(dim="tau")  # Eq. B7
    Klw_t_prime = (Klw_prime * (C_ratio.sum(dim="tau"))).sum(dim="plev")  # Eq. B8
    Klw_resid_prime = Klw_prime - Klw_p_prime - Klw_t_prime  # Eq. B9
    dRlw_true = (Klw * dc).sum(dim=["tau", "plev"])  # LW total
    dRlw_prop = Klw0 * sum_dc  # LW amount component
    dRlw_dctp = (Klw_p_prime * (dc_star.sum(dim="tau"))).sum(dim="plev")  # LW altitude component
    dRlw_dtau = (Klw_t_prime * (dc_star.sum(dim="plev"))).sum(dim="tau")  # LW optical depth component
    dRlw_resid = (Klw_resid_prime * dc_star).sum(dim=["tau", "plev"])     # LW residual
    # dRlw_sum = dRlw_prop + dRlw_dctp + dRlw_dtau + dRlw_resid           # sum of LW components -- should equal LW total

    # SW components
    Ksw0 = (Ksw * c1 / sum_c).sum(dim=["tau", "plev"])  # Eq. B4
    Ksw_prime = Ksw - Ksw0  # Eq. B3
    Ksw_p_prime = (Ksw_prime * (C_ratio.sum(dim="plev"))).sum(dim="tau")  # Eq. B7
    Ksw_t_prime = (Ksw_prime * (C_ratio.sum(dim="tau"))).sum(dim="plev")  # Eq. B8
    Ksw_resid_prime = Ksw_prime - Ksw_p_prime - Ksw_t_prime  # Eq. B9
    dRsw_true = (Ksw * dc).sum(dim=["tau", "plev"])  # SW total
    dRsw_prop = Ksw0 * sum_dc  # SW amount component
    dRsw_dctp = (Ksw_p_prime * (dc_star.sum(dim="tau"))).sum(dim="plev")  # SW altitude component
    dRsw_dtau = (Ksw_t_prime * (dc_star.sum(dim="plev"))).sum(dim="tau")  # SW optical depth component
    dRsw_resid = (Ksw_resid_prime * dc_star).sum(dim=["tau", "plev"])     # SW residual
    # dRsw_sum = dRsw_prop + dRsw_dctp + dRsw_dtau + dRsw_resid

    # Set SW fields to zero where the sun is down
    dRsw_true = xr.where(Ksw0 == 0, 0, dRsw_true)
    dRsw_prop = xr.where(Ksw0 == 0, 0, dRsw_prop)
    dRsw_dctp = xr.where(Ksw0 == 0, 0, dRsw_dctp)
    dRsw_dtau = xr.where(Ksw0 == 0, 0, dRsw_dtau)
    dRsw_resid = xr.where(Ksw0 == 0, 0, dRsw_resid)

    output = {}
    output["LWcld_tot"] = dRlw_true.transpose("time", "lat", "lon")
    output["SWcld_tot"] = dRsw_true.transpose("time", "lat", "lon")
    output["NETcld_tot"] = output["LWcld_tot"]+output["SWcld_tot"]
    output["LWcld_amt"] = dRlw_prop.transpose("time", "lat", "lon")
    output["SWcld_amt"] = dRsw_prop.transpose("time", "lat", "lon")
    output["NETcld_amt"] = output["LWcld_amt"]+output["SWcld_amt"]
    output["LWcld_alt"] = dRlw_dctp.transpose("time", "lat", "lon")
    output["SWcld_alt"] = dRsw_dctp.transpose("time", "lat", "lon")
    output["NETcld_alt"] = output["LWcld_alt"]+output["SWcld_alt"]
    output["LWcld_tau"] = dRlw_dtau.transpose("time", "lat", "lon")
    output["SWcld_tau"] = dRsw_dtau.transpose("time", "lat", "lon")
    output["NETcld_tau"] = output["LWcld_tau"]+output["SWcld_tau"]
    output["LWcld_err"] = dRlw_resid.transpose("time", "lat", "lon")
    output["SWcld_err"] = dRsw_resid.transpose("time", "lat", "lon")
    output["NETcld_err"] = output["LWcld_err"]+output["SWcld_err"]
    
    return output


###########################################################################
def do_obscuration_calcs(CTL, FUT, Klw, Ksw, DT):
    (L_R_bar, dobsc, dunobsc, dobsc_cov) = obscuration_terms3(CTL, FUT)

    # Get unobscured low-cloud feedbacks and those caused by change in obscuration
    ZEROS = np.zeros(L_R_bar.shape)
    dummy, L_R_bar_base = compute_fbk(L_R_bar, L_R_bar, DT)
    dobsc_fbk, dummy = compute_fbk(ZEROS, dobsc, DT)
    dunobsc_fbk, dummy = compute_fbk(ZEROS, dunobsc, DT)
    dobsc_cov_fbk, dummy = compute_fbk(ZEROS, dobsc_cov, DT)
    obsc_output = obscuration_feedback_terms_general(
        L_R_bar_base, dobsc_fbk, dunobsc_fbk, dobsc_cov_fbk, Klw, Ksw
    )

    return obsc_output

###########################################################################
def obscuration_terms3(c1, c2):
    """
    USE THIS VERSION FOR DIFFERENCES OF 2 CLIMATOLOGIES (E.G. AMIP4K, 2xCO2 SLAB RUNS)

    Compute the components required for the obscuration-affected low cloud feedback
    These are the terms shown in Eq 4 of Scott et al (2020) DOI: 10.1175/JCLI-D-19-1028.1
    L_prime = dunobsc + dobsc + dobsc_cov, where
    dunobsc = L_R_prime * F_bar     (delta unobscured low clouds, i.e., true low cloud feedback)
    dobsc = L_R_bar * F_prime       (delta obscuration by upper level clouds)
    dobsc_cov = (L_R_prime * F_prime) - climo(L_R_prime * F_prime)  (covariance term)
    """
    # c is [mo,tau,ctp,lat,lon]
    # c is in percent

    # SPLICE c1 and c2:
    if c1.shape != c2.shape:
        raise RuntimeError("c1 and c2 are NOT the same size!!!")

    c12 = xr.concat([c1, c2], dim="time")
    c12["time"] = xr.cftime_range(start="1983-01", periods=len(c12.time), freq="MS")

    midpt = len(c1)

    U12 = (c12[:, :, 2:, :]).sum(dim=["tau", "plev"]) / 100.0  # [time, lat, lon]

    L12 = c12[:, :, :2, :] / 100.0

    F12 = 1.0 - U12
    F12 = xr.where(F12 >= 0, F12, np.nan)

    F12b = F12.expand_dims(dim=dict(tau=c12.tau), axis=1)
    F12b = F12b.expand_dims(dim=dict(plev=c12.plev), axis=2)

    L_R12 = L12 / F12  # L12/F12b
    sum_L_R12 = (L_R12).sum(dim=["tau", "plev"])
    sum_L_R12c = sum_L_R12.expand_dims(dim=dict(tau=L12.tau), axis=1)
    sum_L_R12c = sum_L_R12c.expand_dims(dim=dict(plev=L12.plev), axis=2)
    this = sum_L_R12c.where(sum_L_R12c <= 1, np.nan)
    this = this.where(this >= 0, np.nan)
    L_R12 = L_R12.where(~np.isnan(this), np.nan)
    L_R12 = L_R12.where(sum_L_R12c <= 1, np.nan)

    L_R_prime, L_R_bar = monthly_anomalies(L_R12)
    F_prime, F_bar = monthly_anomalies(F12b)
    L_prime, L_bar = monthly_anomalies(L12)

    # Cannot have negative cloud fractions:
    L_R_bar = L_R_bar.where(L_R_bar >= 0, 0.0)
    F_bar = F_bar.where(F_bar >= 0, 0.0)

    rep_L_bar = tile_uneven(L_bar, L12)
    rep_L_R_bar = tile_uneven(L_R_bar, L_R12)
    rep_F_bar = tile_uneven(F_bar, F12b)

    dobsc = rep_L_R_bar * F_prime
    dunobsc = L_R_prime * rep_F_bar
    prime_prime = L_R_prime * F_prime

    dobsc_cov, climo_prime_prime = monthly_anomalies(prime_prime)

    # Re-scale these anomalies by 2, since we have computed all anomalies w.r.t.
    # the ctl+pert average rather than w.r.t. the ctl average
    dobsc *= 2
    dunobsc *= 2
    dobsc_cov *= 2

    # change time dimension
    Time = c1.time.copy()
    rep_L_R_bar_new = rep_L_R_bar[midpt:]
    rep_L_R_bar_new["time"] = Time
    dobsc_new = dobsc[midpt:]
    dobsc_new["time"] = Time
    dunobsc_new = dunobsc[midpt:]
    dunobsc_new["time"] = Time
    dobsc_cov_new = dobsc_cov[midpt:]
    dobsc_cov_new["time"] = Time

    return (rep_L_R_bar_new, dobsc_new, dunobsc_new, dobsc_cov_new)


###########################################################################
def obscuration_feedback_terms_general(L_R_bar0, dobsc_fbk, dunobsc_fbk, dobsc_cov_fbk, Klw, Ksw):
    """
    Estimate unobscured low cloud feedback,
    the low cloud feedback arising solely from changes in obscuration by upper-level clouds,
    and the covariance term

    This function takes in a (month,tau,CTP,lat,lon) matrix

    Klw and Ksw contain just the low bins

    the following terms are generated in obscuration_terms():
    dobsc = L_R_bar0 * F_prime
    dunobsc = L_R_prime * F_bar
    dobsc_cov = (L_R_prime * F_prime) - climo(L_R_prime * F_prime)
    """

    Klw_low = Klw
    Ksw_low = Ksw
    L_R_bar0 = 100 * L_R_bar0
    dobsc_fbk = 100 * dobsc_fbk
    dunobsc_fbk = 100 * dunobsc_fbk
    dobsc_cov_fbk = 100 * dobsc_cov_fbk

    LWdobsc_fbk = (Klw_low * dobsc_fbk).sum(dim=["tau", "plev"])
    LWdunobsc_fbk = (Klw_low * dunobsc_fbk).sum(dim=["tau", "plev"])
    LWdobsc_cov_fbk = (Klw_low * dobsc_cov_fbk).sum(dim=["tau", "plev"])

    SWdobsc_fbk = (Ksw_low * dobsc_fbk).sum(dim=["tau", "plev"])
    SWdunobsc_fbk = (Ksw_low * dunobsc_fbk).sum(dim=["tau", "plev"])
    SWdobsc_cov_fbk = (Ksw_low * dobsc_cov_fbk).sum(dim=["tau", "plev"])

    ###########################################################################
    # Further break down the true and apparent low cloud-induced radiation anomalies into components
    ###########################################################################
    # No need to break down dobsc_fbk, as that is purely an amount component.

    # Break down dunobsc_fbk:
    C_ctl = L_R_bar0
    dC = dunobsc_fbk
    C_fut = C_ctl + dC

    obsc_fbk_output = KT_decomposition_general(C_ctl, C_fut, Klw_low, Ksw_low)

    obsc_fbk_output["LWdobsc_fbk"] = LWdobsc_fbk
    obsc_fbk_output["LWdunobsc_fbk"] = LWdunobsc_fbk
    obsc_fbk_output["LWdobsc_cov_fbk"] = LWdobsc_cov_fbk
    obsc_fbk_output["SWdobsc_fbk"] = SWdobsc_fbk
    obsc_fbk_output["SWdunobsc_fbk"] = SWdunobsc_fbk
    obsc_fbk_output["SWdobsc_cov_fbk"] = SWdobsc_cov_fbk

    return obsc_fbk_output

###########################################################################
def xc_to_dataset(idata):
    idata = idata.to_dataset(name="data")
    if "height" in idata.coords:
        idata = idata.drop("height")
    idata = idata.bounds.add_missing_bounds()
    return idata


###########################################################################
def monthly_anomalies(idata):
    """
    Compute departures from the climatological annual cycle
    usage:
    anom,avg = monthly_anomalies(data)
    """
    idata["time"].encoding["calendar"] = "noleap"
    idata = xc_to_dataset(idata)
    clim = idata.temporal.climatology("data", freq="month", weighted=True)
    anom = idata.temporal.departures("data", freq="month", weighted=True)

    return (anom["data"], clim["data"])


###########################################################################
def tile_uneven(data, data_to_match):
    """extend data to match size of data_to_match even if not a multiple of 12"""

    A12 = len(data_to_match) // 12
    ind = np.arange(12,)
    rep_ind = np.tile(ind, (A12 + 1))[
        : int(len(data_to_match))
    ]  # int() added for python3

    rep_data = (data).isel(time=rep_ind)
    rep_data["time"] = data_to_match.time.copy()

    return rep_data

###########################################################################
def CloudRadKernel(filepath,rapidAdj = False):
    
    (ctl_clisccp,fut_clisccp,CRKs,dTs) = get_CRK_data(filepath)

    if rapidAdj: # if doing rapid adjustments instead of feedbacks, do not normalize by ∆T
        dTs = 1

    ###########################################################################
    # Compute cloud feedbacks and their breakdown into components
    ###########################################################################
    print("Compute feedbacks")
    clisccp_fbk, clisccp_base = compute_fbk(ctl_clisccp, fut_clisccp, dTs)
    
    # The following performs the amount/altitude/optical depth decomposition of
    # Zelinka et al., J Climate (2012b), as modified in Zelinka et al., J. Climate (2013)
    output = {}
    for sec in sections:
        print("    for section " + sec)

        PP = sec_dic[sec]

        C1 = clisccp_base[:, :, PP, :]
        C2 = C1 + clisccp_fbk[:, :, PP, :]
        Klw = CRKs['LWkernel'][:, :, PP, :]
        Ksw = CRKs['SWkernel'][:, :, PP, :]

        output[sec] = KT_decomposition_general(C1, C2, Klw, Ksw)

    ###########################################################################
    # Compute obscuration feedback components
    ###########################################################################
    print("Get Obscuration Terms")
    sec = "LO680"  # this should already be true, but just in case...
    PP = sec_dic[sec]
    Klw = CRKs['LWkernel'][:, :, PP, :]
    Ksw = CRKs['SWkernel'][:, :, PP, :]
    obsc_output = {}
    obsc_output[sec] = do_obscuration_calcs(ctl_clisccp, fut_clisccp, Klw, Ksw, dTs)
       
    return (output,obsc_output)