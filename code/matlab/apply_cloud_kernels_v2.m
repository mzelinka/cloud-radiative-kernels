% This script demonstrates how to apply the cloud radiative kernels to
% estimate the radiative impact of ISCCP simulator-defined cloud fraction anomalies 
% Dividing these radiative impacts by the global mean surface temperature
% anomaly yields an estimate of the cloud feedback

% Reference: Zelinka, Mark D., Stephen A. Klein, Dennis L. Hartmann, 2012: Computing and Partitioning Cloud Feedbacks Using 
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

% Other scripts that are called by this script:
% 1. get_netcdf_data.m
% 2. count_wt_mean.m

% Lines that include the comment "USER: MODIFY THIS LINE" are those in which 
% the user must update the path to the relevant netcdf files

% This script written by Mark Zelinka (zelinka1@llnl.gov) on 23 June 2014
% v2 modifications done on 13 July 2017

%% Load in the cloud radiative kernels
% Kernels are in units of W/m2/%
% Kernels are size (mo,tau,CTP,lat,albcs)

variable='/g/g19/zelinka1/CMIP3_kernels/cloud_kernels2.nc'; % USER: MODIFY THIS LINE 
LWkernel = get_netcdf_data(variable,'LWkernel'); 
SWkernel = get_netcdf_data(variable,'SWkernel'); 
albcs_midpt = get_netcdf_data(variable,'albcs'); % 0, 0.5, 1.0
kern_lat = get_netcdf_data(variable,'lat');  
kern_lon=1.25:2.5:360;
kern_coslat=cos(pi*kern_lat./180);

%% Load model clisccp, rsuscs, rsutcs, tas from amip control run 
variable='/p/lscratche/zelinka1/cmip5/clisccp/clisccp_cfMon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc'; % USER: MODIFY THIS LINE 
ctl_clisccp = get_netcdf_data(variable,'clisccp'); % size (time,tau,CTP,lat,lon); units: percent
ctl_clisccp(ctl_clisccp>500)=NaN; 

variable='/p/lscratche/zelinka1/cmip5/tas/tas_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc'; % USER: MODIFY THIS LINE 
ctl_tas = get_netcdf_data(variable,'tas'); % size (time,lat,lon)
ctl_tas(ctl_tas>500)=NaN; 

variable='/p/lscratche/zelinka1/cmip5/rsdscs/rsdscs_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc'; % USER: MODIFY THIS LINE 
ctl_rsdscs = get_netcdf_data(variable,'rsdscs'); % size (time,lat,lon)
ctl_rsdscs(ctl_rsdscs>500)=NaN; 

variable='/p/lscratche/zelinka1/cmip5/rsuscs/rsuscs_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc'; % USER: MODIFY THIS LINE 
ctl_rsuscs = get_netcdf_data(variable,'rsuscs'); % size (time,lat,lon)
ctl_rsuscs(ctl_rsuscs>500)=NaN; 

lon = get_netcdf_data(variable,'lon'); % size (time,lat,lon)
lat = get_netcdf_data(variable,'lat'); % size (time,lat,lon)
coslat=cos(pi*lat./180);

% Compute clear-sky surface albedo
ctl_albcs=ctl_rsuscs./ctl_rsdscs; 
clear ctl_rsuscs ctl_rsdscs

%% Load model clisccp and tas from amipFuture run 
variable='/p/lscratche/zelinka1/cmip5/clisccp/clisccp_cfMon_MPI-ESM-LR_amipFuture_r1i1p1_197901-198112.nc'; % USER: MODIFY THIS LINE 
fut_clisccp = get_netcdf_data(variable,'clisccp'); % size (time,tau,CTP,lat,lon); units: percent
fut_clisccp(fut_clisccp>500)=NaN; 

variable='/p/lscratche/zelinka1/cmip5/tas/tas_Amon_MPI-ESM-LR_amipFuture_r1i1p1_197901-198112.nc'; % USER: MODIFY THIS LINE 
fut_tas = get_netcdf_data(variable,'tas'); % size (time,lat,lon)
fut_tas(fut_tas>500)=NaN; 

%% Compute anual cycles
[a,b,c,d,e]=size(ctl_clisccp);
avgctl_albcs=nan*ones(12,d,e);
avgctl_tas=nan*ones(12,d,e);
avgfut_tas=nan*ones(12,d,e);
avgctl_clisccp=nan*ones(12,b,c,d,e);
avgfut_clisccp=nan*ones(12,b,c,d,e);
% before blindly doing this for every model, make sure your model data start in January
for M=1:12
    avgctl_clisccp(M,:,:,:,:)=squeeze(nanmean(ctl_clisccp(M:12:end,:,:,:,:),1));
    avgfut_clisccp(M,:,:,:,:)=squeeze(nanmean(fut_clisccp(M:12:end,:,:,:,:),1));
    avgctl_tas(M,:,:)=squeeze(nanmean(ctl_tas(M:12:end,:,:),1));
    avgfut_tas(M,:,:)=squeeze(nanmean(fut_tas(M:12:end,:,:),1));
    avgctl_albcs(M,:,:)=squeeze(nanmean(ctl_albcs(M:12:end,:,:),1));
end
clear ctl_clisccp fut_clisccp ctl_tas fut_tas ctl_albcs

%% Difference the anual cycles 
anom_clisccp=avgfut_clisccp-avgctl_clisccp;
anom_tas=avgfut_tas-avgctl_tas;
clear avgfut_clisccp avgctl_clisccp avgfut_tas avgctl_tas

%% Compute global mean surface air temperature anomaly
avgdtas=count_wt_mean(squeeze(nanmean(nanmean(anom_tas,1),3)),coslat',2);
clear anom_tas

%% Interpolate to a common grid (use the same grid as the kernels)
% first, a kluge to deal with weirdness of Matlab's interp function
x=[lon;lon(1)+360];
cat_albcs=cat(3,avgctl_albcs,avgctl_albcs(:,:,1)); 
cat_clisccp=cat(5,anom_clisccp,anom_clisccp(:,:,:,:,1)); 
clear anom_clisccp avgctl_albcs
    
[X1,Y1] = meshgrid(x,lat);
[X2,Y2] = meshgrid(kern_lon,kern_lat);
anom_clisccp_int=nan*ones(12,b,c,90,144);
avgctl_albcs_int=nan*ones(12,90,144);
for M=1:12
    avgctl_albcs_int(M,:,:) = interp2(X1,Y1,squeeze(cat_albcs(M,:,:)),X2,Y2);  
    for T=1:b
        for P=1:c
            anom_clisccp_int(M,T,P,:,:) = interp2(X1,Y1,squeeze(cat_clisccp(M,T,P,:,:)),X2,Y2);
        end
    end
end
clear cat_albcs cat_clisccp

%% Map each location's clear-sky surface albedo to the correct albedo bin
[a,b,c,d,e]=size(anom_clisccp_int); % (month,tau,CTP,lat,lon)
SWkernel_map=NaN*ones(12,b,c,d,e);
[X1,Y1] = meshgrid(albcs_midpt,1:b);
for M=1:a
    for LA=1:d
        alon=squeeze(avgctl_albcs_int(M,LA,:)); 
        if numel(alon(isnan(alon)))>0; continue; end
        [X2,Y2] = meshgrid(alon,1:b);
        for P=1:c
            SWkernel_map(M,:,P,LA,:)=interp2(X1,Y1,squeeze(SWkernel(M,:,P,LA,:)),X2,Y2);
        end
    end
end
LWkernel_map = repmat(LWkernel(:,:,:,:,1),[1,1,1,1,e]); % has no dependence on underlying albedo
clear LWkernel SWkernel

%% Calculate cloud feedback
% each feedback is size (MO,TAU,CTP,LAT,LON)=(12,7,7,90,144)
SW_cld_fdbk=anom_clisccp_int.*SWkernel_map/avgdtas;
LW_cld_fdbk=anom_clisccp_int.*LWkernel_map/avgdtas;

% ensure that SW_cld_fdbk is zero rather than NaN if the SW kernel is zero (e.g., in the polar night)
SW_cld_fdbk(SWkernel_map==0)=0;

%% Quick sanity check:
% print the global, annual mean LW and SW cloud feedbacks:
sumLW = squeeze(nanmean(nanmean(nansum(nansum(LW_cld_fdbk,2),3),1),5));
avgLW_cld_fbk = count_wt_mean(sumLW,kern_coslat,1);
display(['avg LW cloud feedback = ',num2str(avgLW_cld_fbk)])
sumSW = squeeze(nanmean(nanmean(nansum(nansum(SW_cld_fdbk,2),3),1),5));
avgSW_cld_fbk = count_wt_mean(sumSW,kern_coslat,1);
display(['avg SW cloud feedback = ',num2str(avgSW_cld_fbk)])
