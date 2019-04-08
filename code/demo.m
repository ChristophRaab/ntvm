%% Matlab Demo file For Nyström Transfer Vector Machine

addpath(genpath('/libsvm'));
addpath(genpath('../data'));
addpath(genpath('../code'));

clear all;

%% Reuters Dataset
options.ker = 'rbf';        % kernel: 'linear' | 'rbf' | 'lap'
options.eta = 2.0;           % TKL: eigenspectrum damping factor
options.gamma = 1.0;         % TKL: width of gaussian kernel
options.k = 100;              % JDA: subspaces bases
options.lambda = 1.0;        % JDA: regularization parameter
options.svmc = 10.0;         % SVM: complexity regularizer in LibSVM
options.g = 40;              % GFK: subspace dimension
options.tcaNv = 50;          % TCA: numbers of Vectors after reduction
options.theta = 1;           %PCVM: Width of gaussian kernel
options.subspace_dim_d = 5;  %SA: Subspace Dimensions
options.landmarks = 600;     %NTVM: Number of Landmarks
options.ntvm_ker = 'rbf';    %NTVM: Kernel Type
for strData = {'org_vs_people','org_vs_place', 'people_vs_place'} %
    
    for iData = 1:2
        data = char(strData);
        data = strcat(data, '_', num2str(iData));
        load(strcat('../data/Reuters/', data));
        
        fprintf('data=%s\n', data);
        
        Xs=full(bsxfun(@rdivide, bsxfun(@minus,Xs,mean(Xs)), std(Xs)));
        Xt=full(bsxfun(@rdivide, bsxfun(@minus,Xt,mean(Xt)), std(Xt)));
        
        m = size(Xs, 2);
        n = size(Xt, 2);
        %% SVM
        K = kernel(options.ker, [Xs, Xt], [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('SVM = %.2f%%\n', acc(1));
        
        %% PCVM
        
        model = pcvm_train(Xs',Ys,options.gamma);
        [erate, nvec, label, y_prob] = pcvm_predict(Xs',Ys,Xt',Yt,model);
        fprintf('\nPCVM %.2f%% \n', 100-erate*100)
        
        %% TCA
        nt = length(Ys);
        mt = length(Yt);
        K = tca(Xs',Xt',options.tcaNv,options.gamma,options.ker);
        model = svmtrain(full(Ys),[(1:nt)',K(1:nt,1:nt)],['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label,acc,scores] = svmpredict(full(Yt),[(1:mt)',K(nt+1:end,1:nt)],model);
        
        fprintf('\nTCA %.2f%% \n',acc(1));
        
        %% JDA
        
        Cls = [];
        [Z,A] = JDA(Xs,Xt,Ys,Cls,options);
        Z = Z*diag(sparse(1./sqrt(sum(Z.^2))));
        Zs = Z(:,1:size(Xs,2));
        Zt = Z(:,size(Xs,2)+1:end);
        K = kernel(options.ker, Z, [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\nJDA %.2f%% \n',acc(1));
        
        
        %% TKL SVM
        tic;
        K = TKL(Xs, Xt, options);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(options.svmc), ' -t 4 -q 1']);
        [labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\nTKL %.2f%%\n',acc(1));
        
        
        %% PCTKVM No Theta Est
        
        options.theta =2;
        model = pctkvm_train(Xs',Ys,Xt',options);
        [erate, nvec, label, y_prob] = pctkvm_predict(Ys,Yt,model);
        
        fprintf('\nPCTKVM_Theta %.2f%% \n', 100-erate*100);
        
        %% SA
        [Xss,~,~] = pca(Xs');
        [Xtt,~,~] = pca(Xt');
        Xss = Xss(:,1:options.subspace_dim_d);
        Xss = Xtt(:,1:options.subspace_dim_d);
        [predicted_label, acc, decision_values,model] =  Subspace_Alignment(Xs',Xt',Ys,Yt,Xss,Xtt);
        fprintf('\nSA %.2f%%\n',acc(1));
        
        
        %% NTVM
        model = ntvm(full(Xs'),full(Ys),full(Xt'),options);
        [erate, nvec, label, y_prob] = ntvm_predict(Yt,model);
        fprintf('\nNTVM %.2f%% \n', 100-erate*100);
    end
end
