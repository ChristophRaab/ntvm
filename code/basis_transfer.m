function [Xs,Ys] = basis_transfer(Xs,Xt,Ys)
%BASIS_TRANSFER Summary of this function goes here
    
    [US,SZ,VS] = svd(Xs,"econ");
    [U,S,V] = svd(Xt,"econ");
     Xs = U*SZ*V';
end

