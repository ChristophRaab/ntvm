function [model] = pctkvm_train(trainX,trainY,testX,options)
% Implementations of Probabilistic Classification Transfer Kernel Vector Machines.
%
% The original PCVM Algorithm is presented in the following paper:
% Huanhuan Chen, Peter Tino and Xin Yao. Probabilistic Classification Vector Machines.
% IEEE Transactions on Neural Networks. vol.20, no.6, pp.901-914, June 2009.
%    Copyright (c) Huanhuan Chen
% The following improvements by Christoph Raab:
% Ability to Transfer Learning with Domain Invariant Transfer Kernel Learning
% See the Matlab-method TKL(.) for to the original paper
% Optional theta estimation
% Multi-Class Label with One vs One
%--------------------------------------------------------------------------
%Parameters:
% trainX - (N,M) Matrix with training data. M refers to dims
% trainY - Corresponding training label for the training
% testX - Testdata to train the Transfer Kernel. (Optional)
% options - Struct which contains parameters:
%          .theta - Parameter for theta. Give -1 to make a theta estimation
%                   with theta ~= 0. The theta is fixed to this.
%          .eta - eigenspectrum damping factor for TKL
%          .ker - Kernel-Type: 'linear' | 'rbf' | 'lap'
% Output:
% The trained model as a struct. For multiclass problems, it is a struct
% array.

C = unique(trainY);
sizeC = size(C,1);

if sizeC == 2
    
    % the maximal iterations
    niters = 600;
    
    pmin=10^-5;
    errlog = zeros(1, niters);
    
    ndata= size(trainX,1);
    
    display = 0; % can be zero
    
    % Initial weight vector to let w to be large than zero
    w = rand(ndata,1)+ 0.2;
    
    % Initial bias b
    b = randn;
    
    % initialize the auxiliary variables Ht to follow the target labels of the training set
    Ht = 10*rand(ndata,1).*trainY + rand(ndata,1);
    
    % Threshold to determine whether this is small
    w_minimal = 1e-3;
    
    % Threshold for convergence
    threshold = 1e-3;
    
    % all one vector
    I = ones(ndata,1);
    
    y = trainY;
    
    % active vector indicator
    nonZero = ones(ndata,1);
    
    % non-zero wegith vector
    w_nz = w(logical(nonZero));
    
    wold = w;
    
    repy=repmat(trainY(:)', ndata, 1);
    
    if display
        number_of_RVs = zeros(niters,1);
    end
    
    %Calculate Disimilarity Matrix
    X = [trainX' testX'];
    n1sq = sum(X.^2, 1);
    n1 = size(X, 2);
    D = (ones(n1, 1) * n1sq)' + ones(n1, 1) * n1sq -2 * (X' * X);
    D = sqrt(D);
    
    
    if options.theta == -1
        options.theta  = pthetaEstimation(D,size(trainX,2));
    end
    
    K = pTKL(D,size(trainX, 1), options);
    
    % Take the left upper square of the K Matrix for the learning algorithm
    Kl = K(1:ndata,1:ndata);
    
    
    % Main loop of algorithm
    for n = 1:niters
        %     fprintf('\n%d. iteration.\n',n);
        
        
        % Note that theta^2
        % scale columns of kernel matrix with label trainY
        Ky = Kl.*repmat(trainY(:)', ndata, 1);
        
        % non-zero vector
        Ky_nz = Ky(:,logical(nonZero));
        
        if n==1
            Ht_nz = Ht;
        else
            Ht_nz = Ky_nz*w_nz + b*ones(ndata,1);
        end
        
        Z = Ht_nz + y.*normpdf(Ht_nz)./(normcdf(y.*Ht_nz)+ eps);
        
        % Adjust the new estimates for the parameters
        M = sqrt(2)*diag(w_nz);
        
        % new weight vector
        Hess = eye(size(M,1))+M*Ky_nz'*Ky_nz*M;
        Hess = Hess+1*eps*ones(size(Hess));
        U    = chol(Hess);
        Ui   = inv(U);
        
        w(logical(nonZero)) = M*Ui*Ui'*M*(Ky_nz'*Z - b*Ky_nz'*I);
        
        S = sqrt(2)*abs(b);
        b = S*(1+ S*ndata*S)^(-1)*S*(I'*Z - I'*Ky*w);
        
        
        % expectation
        A=diag(1./(2*w_nz.^2));
        beta=(0.5+pmin)/(b^2+pmin);
        
        
        nonZero	= (w > w_minimal);
        
        % determine used vectors
        used = find(nonZero==1);
        
        w(~nonZero)	= 0;
        
        % non-zero weight vector
        w_nz = w(nonZero);
        
        if display % && mod(n,10)==0
            number_of_RVs(n) = length(used);
            plot(1:n, number_of_RVs(1:n));
            title('non-zero vectors')
            drawnow;
        end
        
        if (n >1 && max(abs(w - wold))< threshold)
            
            
            
            break;
        else
            wold = w;
        end
        
    end
    
    if n<niters
        %         fprintf('PCVM terminates in %d iteration.\n',n);
        
    else
        %         fprintf('Exceed the maximal iterations (500). \nConsider to increase niters.\n')
    end
    model.w = w;
    model.b = b;
    model.used = used;
    model.theta = options.theta;
    model.errlog = errlog;
    model.K = K;
    
elseif sizeC > 2
    fprintf('\nMulticlass Problem detected! Splitting up label vector..\n');
    u = 1;
    
    %For Loops to calculate the One vs One Models
    for j = 1:sizeC
        for i=j+1:sizeC
            
            one = C(j,1);
            two = C(i,1);
            
            oneIndx = find(trainY == one);
            twoIndx = find(trainY == two);
            
            trainYOR = [ones(size(oneIndx,1),1); ones(size(twoIndx,1),1)*-1];
            
            trainXOR= [trainX(oneIndx,:); trainX(twoIndx,:)];
            
            singleM = pctkvm_train(trainXOR,trainYOR,testX,options);
            singleM.one = one; singleM.two = two;
            model(u) = singleM;
            
            u = u+1;
        end
    end
    fprintf('\nPCTKVM: Training finished\n');
else
    fprintf('\nNo suitable labels found! Please enter a valid class labels\n');
end

