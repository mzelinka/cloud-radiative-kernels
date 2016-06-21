function [data,index] = get_netcdf_data(filename,variable)

% contact Mark Zelinka (zelinka1@llnl.gov)

data=[];
ncid=netcdf.open(filename,'NC_NOWRITE');
[Ndims,nvars,ngatts,unlimdimid] = netcdf.inq(ncid);
for index=0:nvars-1; 
    varname = netcdf.inqVar(ncid,index);
    if strcmp(varname,variable)==1; 
        varid = netcdf.inqVarID(ncid,varname);
        data = netcdf.getVar(ncid,varid);
        try
            MV=netcdf.getAtt(ncid,varid,'missing_value');
            data(data==MV)=NaN;
        end
        break;
    end
end
if numel(data)==0
    index=[]; 
else
    if min(size(data))>1 % not a vector
        data=permute(data,[ndims(data):-1:1]);
    end
end
data=double(data);
netcdf.close(ncid);
