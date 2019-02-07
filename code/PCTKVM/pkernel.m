% With Fast Computation of the RBF kernel matrix
% To speed up the computation, we exploit a decomposition of the Euclidean distance (norm)
%
% Inputs:
%       ker:    'rbf','srbf','lap'
%       D:      dissimililarity matrix 
%       theta:  bandwidth of the kernel
% Output:
%       K: kernel matrix
%
% Gustavo Camps-Valls
% 2006(c)
% Jordi (jordi@uv.es), 2007
% 2007-11: if/then -> switch, and fixed RBF kernel
% Modified by Mingsheng Long
% 2013(c)
% Mingsheng Long (longmingsheng@gmail.com), 2013
% Extension: Works with dissimilarity matrix and standard Gaussian Kernel
% by Christoph Raab 

function K = pkernel(ker, D, theta)

if ~exist('ker', 'var')
    ker = 'rbf';
end
if ~exist('theta', 'var')
    theta = 1.0;
end

switch ker
    case 'rbf'
        theta = theta / mean(mean(D));
        K = exp(-theta * D);
    case 'srbf'
        coe = 1 / (2*theta*theta);
        K = exp(-coe * D);
    case 'lap'
        theta = theta / mean(mean(D));
        K = exp(-sqrt(theta * D));
    otherwise
        error(['Unsupported kernel ' ker])
end

if size(K, 1) == size(K, 2)
    K = (K + K') / 2;
end

end
