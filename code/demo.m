% This script tests the performance of the STVM for the Reuters, 20
% newsgroup dataset.
% Kernel Script from M. Long, J. Wang, J. Sun and P. S. Yu, "Domain
% Invariant Transfer Kernel Learning," in IEEE Transactions on Knowledge
% and Data Engineering
close all;
clear all;
addpath(genpath('../libsvm/matlab'));
addpath(genpath('../data'));
addpath(genpath('../code'));

options.ker = 'linear';      % TKL: kernel: | 'rbf' |'srbf | 'lap'
options.eta = 2.0;        % TKL: eigenspectrum damping factor
options.gamma = 1;        % TKL: width of gaussian kernel
options.svmc = 10.0;      % SVM: complexity regularizer in LibSVM
options.theta = 2;        % PCVM: width of kernel

testSize= 5;
for name =  {'org_vs_people','org_vs_place', 'people_vs_place'}

    errResult = [];
    nvecResult = [];

    for iData = 1:2
        data = char(name);
        data = strcat(data, '_', num2str(iData));
        load(strcat('../data/Reuters/', data));

        fprintf('data=%s\n', data);

        Xs =full(Xs);
        Xt = full(Xt);

        % Xs-SCORE and Sampling
        Xs=bsxfun(@rdivide, bsxfun(@minus,Xs,mean(Xs)), std(Xs));
        Xt=bsxfun(@rdivide, bsxfun(@minus,Xt,mean(Xt)), std(Xt));
        Xs = Xs';Xt = Xt';
%         idx = kmeans(Xt,2,'distance','correlation');
%         Xt = [ Xt(idx==1,:); Xt(idx==2,:)];
%         Yt = [ Yt(idx==1,:);Yt(idx==2,:)];
%         
            
        model = stvm(full(Xs),full(Ys),full(Xt),options);

        [erate, nvec, label, y_prob] = stvm_predict(Yt,model);
        erate = erate*100;
        fprintf('\nSTVM %.2f%% \n', erate);

        %Clustering:
        %idx = kmeans(Xs,2,'distance','cosine');
        %result = full(idx == Ys);
        %count = sum(result(:) == 1)
        % acc = count - size(Ys,1)

%         idxs = randperm(size(Yt,1));
%         Xs = Xs(idxs,:);
%         Ys = Ys(idxs);
%         idxs = randperm(size(Yt,1));
%         Xt = Xt(idxs,:);
%         Yt = Yt(idxs);
%         idx = kmeans(Xt,2,'distance','correlation');
%         Xt = [ Xt(idx==1,:); Xt(idx==2,:)];
%         Yt = [ Yt(idx==1,:);Yt(idx==2,:)];
        [Xs,Ys] = augmentation(Xs,Xt,Ys);
        [Xt,Xs]=ny_svd(Xt,Xs,600);
        m = size(Xs, 1);
        n = size(Xt, 1);
        K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);

        fprintf('\n Error: SVM %.2f%% \n', 100-acc(1));

        %             model = pcvm_train(Xs,Ys,options.theta);
        %             [erate, nvec, label, y_prob] = pcvm_predict(Xs,Ys,Xt,Yt,model);
        %             erate = erate*100;
        %             fprintf('\nPCVM %.2f%% \n', erate);

    end
end
% srcStr = {'Caltech10', 'Caltech10', 'Caltech10', 'amazon', 'amazon', 'amazon', 'webcam', 'webcam', 'webcam', 'dslr', 'dslr', 'dslr'};
% tgtStr = {'amazon', 'webcam', 'dslr', 'Caltech10', 'webcam', 'dslr', 'Caltech10', 'amazon', 'dslr', 'Caltech10', 'amazon', 'webcam'};
% options.theta = 2;
% options.ker = 'linear';      % TKL: kernel: | 'rbf' |'srbf | 'lap'
% for iData = 1:12
%     src = char(srcStr{iData});
%     tgt = char(tgtStr{iData});
%     data = strcat(src, '_vs_', tgt);
%     
%     load(['../data/OfficeCaltech/' src '_SURF_L10.mat']);
%     fts = fts ./ repmat(sum(fts, 2), 1, size(fts, 2));
%     Xs = zscore(fts, 1);
%     Ys = labels;
%     
%     load(['../data/OfficeCaltech/' tgt '_SURF_L10.mat']);
%     fts = fts ./ repmat(sum(fts, 2), 1, size(fts,2));
%     Xtt = zscore(fts, 1);
%     Yt = labels;
%     
%     fprintf('data=%s\n', data);
%     Xs = bsxfun(@rdivide, Xs, sqrt(sum(Xs.^2, 1)) + eps);
%     Xtt = bsxfun(@rdivide, Xtt, sqrt(sum(Xtt.^2, 1)) + eps);
%     Xs = Xs';
%     Xtt = Xtt';
%     Xs = bsxfun(@rdivide, Xs, sqrt(sum(Xs.^2, 1)) + eps);
%     Xtt = bsxfun(@rdivide, Xtt, sqrt(sum(Xtt.^2, 1)) + eps);
%     Xs = Xs';
%     Xt = Xtt';
%     
%         model = stvm(full(Xs),full(Ys),full(Xt),options);
%     
%         [erate, nvec, label, y_prob] = stvm_predict(Yt,model);
%         erate = erate*100;
%         fprintf('\nSTVM %.2f%% \n', erate);
%     
%     [Xs,Ys] = basis_transfer(Xs,Xt,Ys);
%     m = size(Xs, 1);
%     n = size(Xt, 1);
%     K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
%     model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
%     [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
%     fprintf('\n Error: SVM %.2f%% \n', 100-acc(1));
% end
options.theta = 2;
options.ker = 'rbf';      % TKL: kernel: | 'rbf' |'srbf | 'lap'
for name = {'comp_vs_rec','comp_vs_sci','comp_vs_talk','rec_vs_sci','rec_vs_talk','sci_vs_talk'}%
    for j=1:36
        
        data = char(name);
        data = strcat(data, '_', num2str(j));
        load(strcat('../data/20Newsgroup/', data));
        fprintf('data=%s\n', data);
        
        %% Xs-SCORE and Sampling
        Xs=bsxfun(@rdivide, bsxfun(@minus,Xs,mean(Xs)), std(Xs));
        Xt=bsxfun(@rdivide, bsxfun(@minus,Xt,mean(Xt)), std(Xt));
        
        Xs = full(Xs)';
        Xt = full(Xt)';
        soureIndx = crossvalind('Kfold', Ys, 2);
        targetIndx = crossvalind('Kfold', Yt,2);
        
        Xs = Xs(find(soureIndx==1),:);
        Ys = Ys(find(soureIndx==1),:);
        
        
        Xt = Xt(find(targetIndx==1),:);
        Yt = Yt(find(targetIndx==1),:);
        
        
        m = size(Xs, 1);
        n = size(Xt, 1);
        
        %% SVM
        %         K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
        %
        %         model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        %         [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        %
        %         fprintf('SVM = %0.4f\n', acc(1));
        %
        %         %% PCVM
        %         model = pcvm_train(Xs,Ys,options.theta);
        %         [erate, nvec, label, y_prob] = pcvm_predict(Xs,Ys,Xt,Yt,model);
        %         erate = erate*100;
        %         fprintf('\nPCVM %.2f%% \n', erate);
        
        model = stvm(full(Xs),full(Ys),full(Xt),options);
        
        [erate, nvec, label, y_prob] = stvm_predict(Yt,model);
        erate = erate*100;
        fprintf('\nSTVM %.2f%% \n', erate);
        
        [Xs,Ys] = basis_transfer(Xs,Xt,Ys);
        m = size(Xs, 1);
        n = size(Xt, 1);
        K = kernel(options.ker, [Xs', Xt'], [],options.gamma);
        model = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-c ', num2str(options.svmc), ' -t 4 -q 1']);
        [label, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], model);
        fprintf('\n Error: SVM %.2f%% \n', 100-acc(1));
    end
end
