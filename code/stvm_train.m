function [model] = stvm_train(Xs,Ys,Xt,options)
%STVM Summary of this function goes here
%   Detailed explanation goes here
    niters = 600;
    
    pmin=10^-5;
    errlog = zeros(1, niters);
    
    ndata= size(Xs,1);
    
    display = 0; % can be zero
    
    % Initial weight vector to let w to be large than zero
    w = rand(ndata,1)+ 0.2;
    
    % Initial bias b
    b = randn;
    
    % initialize the auxiliary variables Ht to follow the target labels of the training set
    Ht = 10*rand(ndata,1).*Ys + rand(ndata,1);
    
    % Threshold to determine whether this is small
    w_minimal = 1e-3;
    
    % Threshold for convergence
    threshold = 1e-3;
    
    % all one vector
    I = ones(ndata,1);
    
    y = Ys;
    
    % active vector indicator
    nonZero = ones(ndata,1);
    
    % non-zero wegith vector
    w_nz = w(logical(nonZero));
    
    wold = w;
    
    repy=repmat(Ys(:)', ndata, 1);
    
    if display
        number_of_RVs = zeros(niters,1);
    end
    
    
    theta = options.theta;
    K = kernel(options.ker, [Xs', Xt'], [],theta);
    
    % Take the left upper square of the K Matrix for the learning algorithm
    Kl = K(1:ndata,1:ndata);
    
    
    % Main loop of algorithm
    for n = 1:niters
        %     fprintf('\n%d. iteration.\n',n);
        
        
        % Note that theta^2
        % scale columns of kernel matrix with label Ys
        Ky = Kl.*repmat(Ys(:)', ndata, 1);
        
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
        Hess = Hess+eps*ones(size(Hess));
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
    model.theta = theta;
    model.errlog = errlog;
    model.K = K;
    model.Ys = Ys;
end

