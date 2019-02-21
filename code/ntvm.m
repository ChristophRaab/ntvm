function [model] = ntvm(Xs,Ys,Xt,options)
% Implementations of Probabilistic Classification Transfer Kernel Vector Machines.
%
% The original PCVM Algorithm is presented in the following paper:
% Huanhuan Chen, Peter Tino and Xin Yao. Probabilistic Classification Vector Machines.
% IEEE Transactions on Neural Networks. vol.20, no.6, pp.901-914, June 2009.
%	Copyright (c) Huanhuan Chen
% The following improvments by Christoph Raab:
% Ability to Transfer Learning with SVD Rotation to algin source and target
% distributions.
% BETA VERSION
% Optional: theta estimation
% Multi-Class Label with One vs One
%--------------------------------------------------------------------------
%Parameters:
% [Xs] - (N,M) Matrix with training data. M refers to dims
% [Ys] - Corrosponding training label for the training
% [Xt] - Testdata to train the Transfer Kernel. (Optional)
% [options] - Struct which contains parameters:
%          .theta - Parameter for theta. Give -1 to make a theta estimation
%                   with theta ~= 0. The theta is fixed to this.
%          .eta - eigenspectrum damping factor for TKL
%          .ker - Kernel Type: 'linear' | 'rbf' | 'lap'
% Output:
% The trained model as struct. For multiclass problems struct array

C = unique(Ys,'stable');
sizeC = size(C,1);
%    Align of feature space examples´
[Xs,Ys] = augmentation(Xs,Xt,Ys);
 %[Xs,Ys] = basis_transfer(Xs,Xt,Ys);
  %     Xt = US*S*VS';
[Xt,Xs]=ny_svd(Xt,Xs,options.landmarks);

if sizeC == 2
  
    model = ntvm_train(Xs,Ys,Xt,options);
    
elseif sizeC > 2
    fprintf('\nMulticlass Problem detected! Splitting up label vector..\n');
    u = 1;
    
    %For Loops to calculate the One vs One Models
    for j = 1:sizeC
        for i=j+1:sizeC
            
            one = C(j,1);
            two = C(i,1);
            
            oneIndx = find(Ys == one);
            twoIndx = find(Ys == two);
            
            YsOR = [ones(size(oneIndx,1),1); ones(size(twoIndx,1),1)*-1];
            
            XsOR= [Xs(oneIndx,:); Xs(twoIndx,:)];
            
            singleM = ntvm_train(XsOR,YsOR,Xt,options);
            singleM.one = one; singleM.two = two;
            model(u) = singleM;
            
            u = u+1;
        end
    end
    fprintf('\nPCTKVM: Training finished\n');
else
    fprintf('\nNo suitable labels found! Please enter a valid class labels\n');
end

