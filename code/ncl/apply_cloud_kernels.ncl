; This NCL script is transformed from original Zelinka's matlab file (.m)
; by Xiaolong Chen (chenxl@lasg.iap.ac.cn) 01/30/2018

begin

  predir1 = "/WORK/chenxl/task/cloud/data/cloud_kernel/"   ; kernel's location
  predir2 = "/WORK/chenxl/task/cloud/data/examp_file/"     ; model data's location

  fdir = predir1+"cloud_kernels2.nc"
  fid = addfile(fdir,"r")
  LWkernel = tofloat(fid->LWkernel)
  SWkernel = tofloat(fid->SWkernel)
  albcs_midpt = tofloat(fid->albcs)    ; 0, 0.5, 1.0
  tau_midpt = tofloat(fid->tau_midpt)
  LWkernel!0 = "mo"
  LWkernel!1 = "tau"
  LWkernel!2 = "plev"
  LWkernel!3 = "lat"
  LWkernel!4 = "alb"
  SWkernel!0 = "mo"
  SWkernel!1 = "tau"
  SWkernel!2 = "plev"
  SWkernel!3 = "lat"
  SWkernel!4 = "alb"

  printVarSummary(SWkernel)
  replace_ieeenan(LWkernel,0.,0)

  kern_lat = tofloat(fid->lat)
  nklat = dimsizes(kern_lat)
  nklon = 144
  kern_lon = fspan(1.25,358.75,nklon)
  rad = atan(1.)/45.
  kern_coslat=cos(rad*kern_lat)

  fdir = predir2+"clisccp_cfMon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc"
  fid = addfile(fdir,"r")
  ctl_clisccp = fid->clisccp
  ctl_clisccp = where(ctl_clisccp .gt. 500., ctl_clisccp@_FillValue, ctl_clisccp)
  tau_ctl = tofloat(ctl_clisccp&tau)
  plev_ctl = tofloat(ctl_clisccp&plev)

  fdir = predir2+"tas_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc"
  fid = addfile(fdir,"r")
  ctl_tas = fid->tas
  ctl_tas = where(ctl_tas .gt. 500., ctl_tas@_FillValue, ctl_tas)

  fdir = predir2+"rsdscs_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc"
  fid = addfile(fdir,"r")
  ctl_rsdscs = fid->rsdscs
  ctl_rsdscs = where(ctl_rsdscs .gt. 500., ctl_rsdscs@_FillValue, ctl_rsdscs)

  fdir = predir2+"rsuscs_Amon_MPI-ESM-LR_amip_r1i1p1_197901-198112.nc"
  fid = addfile(fdir,"r")
  ctl_rsuscs = fid->rsuscs
  ctl_rsuscs = where(ctl_rsuscs .gt. 500., ctl_rsuscs@_FillValue, ctl_rsuscs)

  lon = ctl_tas&lon
  lat = ctl_tas&lat
  coslat = cos(rad*lat)
  nlat = dimsizes(lat)
  nlon = dimsizes(lon)
  nlev = dimsizes(ctl_clisccp&plev)

; Compute clear-sky surface albedo
  ctl_rsdscs = where(ctl_rsdscs .eq. 0., ctl_rsdscs@_FillValue, ctl_rsdscs)   ; exclude polar night
  ctl_albcs = ctl_rsuscs
  ctl_albcs = ctl_rsuscs/ctl_rsdscs
  delete(ctl_rsuscs)
  delete(ctl_rsdscs)

; Load model clisccp and tas from amipFuture run
  fdir = predir2+"clisccp_cfMon_MPI-ESM-LR_amipFuture_r1i1p1_197901-198112.nc"
  fid = addfile(fdir,"r")
  fut_clisccp = fid->clisccp
  fut_clisccp = where(fut_clisccp .gt. 500., fut_clisccp@_FillValue, fut_clisccp)

  fdir = predir2+"tas_Amon_MPI-ESM-LR_amipFuture_r1i1p1_197901-198112.nc"
  fid = addfile(fdir,"r")
  fut_tas = fid->tas
  fut_tas = where(fut_tas .gt. 500., fut_tas@_FillValue, fut_tas)
  
; Compute anual cycles
  dim_clisccp = dimsizes(ctl_clisccp)
  ntau = dim_clisccp(1)     ; tau dimension
  
  avgctl_clisccp = new((/12,ntau,nlev,nlat,nlon/),float)
  avgfut_clisccp = new((/12,ntau,nlev,nlat,nlon/),float)
  do i = 0, ntau-1
  avgctl_clisccp(:,i,:,:,:) = clmMonTLLL(ctl_clisccp(:,i,:,:,:)) 
  avgfut_clisccp(:,i,:,:,:) = clmMonTLLL(fut_clisccp(:,i,:,:,:)) 
  end do  
  avgctl_tas = clmMonTLL(ctl_tas) 
  avgfut_tas = clmMonTLL(fut_tas) 
  avgctl_albcs = clmMonTLL(ctl_albcs) 

  delete(ctl_clisccp)
  delete(fut_clisccp)
  delete(ctl_tas)
  delete(fut_tas)
  delete(ctl_albcs)

; Difference the anual cycles
  anom_clisccp = avgctl_clisccp
  anom_clisccp = avgfut_clisccp - avgctl_clisccp
  anom_tas = avgctl_tas
  anom_tas = avgfut_tas - avgctl_tas
  delete(avgfut_clisccp)
  delete(avgctl_clisccp)  
  delete(avgfut_tas)
  delete(avgctl_tas)  
  
; Compute global mean surface air temperature anomaly
  avgdtas = wgt_areaave(anom_tas,coslat,1.0,0)
  delete(anom_tas)

; Interpolate to a common grid (use the same grid as the kernels)
  avgctl_albcs_int = linint2_Wrap(lon,lat,avgctl_albcs,True,kern_lon,kern_lat,0)
  anom_clisccp_int = linint2_Wrap(lon,lat,anom_clisccp,True,kern_lon,kern_lat,0)
;  avgctl_albcs_int =  where(ismissing(avgctl_albcs_int), -1., avgctl_albcs_int)   ; mark where is polar night
; Map each location's clear-sky surface albedo to the correct albedo bin

  xi1d = albcs_midpt
  yi1d = tau_midpt

  SWkernel_map = anom_clisccp_int 
  zi = SWkernel(mo|:,plev|:,lat|:,tau|:,alb|:)

  do i = 0, 11
  do j = 0, nklat-1
  xo2d = onedtond(avgctl_albcs_int(i,j,:),(/ntau,nklon/))
  yo2d = onedtond(tau_ctl,(/nklon,ntau/))
  yo2d!0 = "alb"
  yo2d!1 = "tau"
  xo1d = ndtooned(xo2d)
  yo1d = ndtooned(yo2d(tau|:,alb|:))

;  if all(.not. ismissing(xo1d)) then
  zo = linint2_points_Wrap(xi1d,yi1d,zi(i,:,j,:,:),False,xo1d,yo1d,0)
  zo2 = reshape(zo,(/nlev,ntau,nklon/))
  zo2!0 = "plev"
  zo2!1 = "tau"
  zo2!2 = "alb"
  SWkernel_map(i,:,:,j,:) = (/zo2(tau|:,plev|:,alb|:)/)
  delete(zo)
  delete(zo2)
;  else
;  SWkernel_map(i,:,:,j,:) = 0.
;  end if

  delete(xo2d)
  delete(yo2d)
  delete(xo1d)
  delete(yo1d)
  end do
  end do

  printMinMax(SWkernel_map,0)
 ; exit

;  SWkernel_map = where(ismissing(SWkernel_map), 0., SWkernel_map)
  LWkernel_map = conform(anom_clisccp_int,LWkernel(:,:,:,:,0),(/0,1,2,3/))

; Calculate cloud feedback
; each feedback is size (MO,TAU,CTP,LAT,LON)=(12,7,7,90,144) 
  SW_cld_fdbk = anom_clisccp_int*SWkernel_map/conform(anom_clisccp_int,avgdtas,0)
  LW_cld_fdbk = anom_clisccp_int*LWkernel_map/conform(anom_clisccp_int,avgdtas,0)
    
; ensure that SW_cld_fdbk is zero rather than NaN if the SW kernel is zero (e.g., in the polar night)
  SW_cld_fdbk = where(SWkernel_map .eq. 0., 0., SW_cld_fdbk)

; Quick sanity check:
; print the global, annual mean LW and SW cloud feedbacks:
  avgLW_cld_fbk = wgt_areaave(dim_avg_n_Wrap(dim_sum_n_Wrap(LW_cld_fdbk,(/1,2/)),0),kern_coslat,1.0,0)
  avgSW_cld_fbk = wgt_areaave(dim_avg_n_Wrap(dim_sum_n_Wrap(SW_cld_fdbk,(/1,2/)),0),kern_coslat,1.0,0)

  print("avg LW cloud feedback = "+avgLW_cld_fbk)
  print("avg SW cloud feedback = "+avgSW_cld_fbk)
;  exit 
; save output
  LW_cld_fdbk!0 = "month"
  LW_cld_fdbk!1 = "tau"
  LW_cld_fdbk!2 = "plev"
  LW_cld_fdbk!3 = "lat"
  LW_cld_fdbk!4 = "lon"
  LW_cld_fdbk&month = ispan(1,12,1)
  LW_cld_fdbk&tau = tau_ctl
  LW_cld_fdbk&plev = plev_ctl
  LW_cld_fdbk&lat = kern_lat
  LW_cld_fdbk&lon = kern_lon 
  LW_cld_fdbk&lat@units = "degrees_north"
  LW_cld_fdbk&lon@units = "degrees_east"
  LW_cld_fdbk@long_name = "Cloud longwave feedback"
  LW_cld_fdbk@units = "W m-2 K-1"
  copy_VarMeta(LW_cld_fdbk,SW_cld_fdbk)
  SW_cld_fdbk@long_name = "Cloud shortwave feedback"

  cdir = predir2+"cld_fdbk.nc"
  system("rm -f "+cdir)
  ncdf = addfile(cdir,"c")
  ncdf->csf = SW_cld_fdbk
  ncdf->clf = LW_cld_fdbk
end
