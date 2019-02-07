function [ rmin ] = pthetaEstimation( D,m)
%THETAESTIMATION Calculates the minimum width of the gaussian kernel based on the
%given a dissimilarity matrix D(n,n) with n rows/columns and m dimensions.
%-------------------------------------------------------------------------
%INPUT: Datamatrix
%OUTPUT: rmin - Minimum width

n = size(D, 2);

djmax = max(D);

rmin =min(bsxfun(@rdivide, djmax, (sqrt(m)*nthroot(n-1,m))));

end

