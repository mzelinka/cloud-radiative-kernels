% function count_wt_mean.m
% Created 3/8/07 to compute a count-weighted mean of a matrix

% this code altered 4/3/07 to return a count matrix that is not count weighted
% this code altered 8/16/07 to return a count matrix of finite elements where counts > 0
% this code altered 8/23/07 to return the following four matrices:
    % 1.) the count-weighted mean along dimension dim
    % 2.) the count-weighted count matrix (the sum of the count matrix along dim)
    % 3.) the standard deviation along dimension dim
    % 4.) the number of finite observations along dimension dim

function [mean,sumcnt,std,num]=count_wt_mean(matrix,counts,dim)

% counts and matrix must be the same size
counts(isnan(matrix))=NaN;
sumcnt=nansum(counts,dim); 
mean =  nansum(matrix.*counts,dim)./sumcnt;

std=nanstd(matrix,0,dim);
matrix(counts==0)=NaN;
num=nansum(isfinite(matrix),dim); 

% if this is in a loop, you will most likely need to put the following statement after calling this function 
% to suppress the divide by zero warning:
% warning off last 