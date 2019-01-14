function [M,e] = ny_svd_one_step(X,landmarks)
%NY_SVD Summary of this function goes here
%   Detailed explanation goes here
X = full(X);
idx = randperm(size(X,1),landmarks);

A = X(idx,idx);
B = X(1:landmarks,landmarks+1:end);
F = X(landmarks+1:end,1:landmarks);
C = X(landmarks+1:end,landmarks+1:end);
sA = pinv(sqrtm(full(A)));
Gu = [A;F]*sA;
Gh = (sA*[A B])';
M = Gu*Gh';
[U,S,V] = svd(M);
e = norm(M-X);
end

