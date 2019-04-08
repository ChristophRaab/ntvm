%% Skript to plot the model of stvm and tkl-svm
% Load orgs vs people dataset
load ../data/Reuters/org_vs_people_1;

% Draw a 2-fold sample
Xs =full(Xs);
Xt = full(Xt);
Xs=bsxfun(@rdivide, bsxfun(@minus,Xs,mean(Xs)), std(Xs));
Xt=bsxfun(@rdivide, bsxfun(@minus,Xt,mean(Xt)), std(Xt));
Xs = Xs';Xt = Xt';
soureIndx = crossvalind('Kfold', Ys, 2);
targetIndx = crossvalind('Kfold', Yt, 2);

Xs = Xs(find(soureIndx==1),:);
Ys = Ys(find(soureIndx==1),:);


Xt = Xt(find(targetIndx==1),:);
Yt = Yt(find(targetIndx==1),:);

%Parameters for stvm
options.theta = 1;
options.ker = 'rbf';
%Train and predict
model_pctkvm = pctkvm_train(Xs,Ys,Xt,options);
[erate, nvec, label, y_prob] = pctkvm_predict(Ys,Yt,model_pctkvm);

% Parameters for tkl-svm
m = size(Xs, 1);
n = size(Xt, 1);
options.ker = 'rbf';      % TKL: kernel: | 'rbf' |'srbf | 'lap'
options.eta = 2.0;           % TKL: eigenspectrum damping factor
options.gamma = 1;         % TKL: width of gaussian kernel
%Train and predict
K = TKL(Xs', Xt', options);
modelsvm = svmtrain(full(Ys), [(1:m)', K(1:m, 1:m)], ['-s 0 -c ', num2str(10), ' -t 4 -q 1']);
[labels, acc,scores] = svmpredict(full(Yt), [(1:n)', K(m+1:end, 1:m)], modelsvm);

% Find subspace embedding with minimal loss
i = 50; j = 30;
[Xs,loss]  = tsne(full(Xs),'Algorithm','barneshut','NumDimensions',2,'NumPCAComponents',j,'Perplexity',i);
       
%% Plot it
figure
plot(Xs(Ys==-1,1),Xs(Ys==-1,2),'r.');
hold on
plot(Xs(Ys==1,1),Xs(Ys==1,2),'bx');
plot(Xs(model_pctkvm.used,1),Xs(model_pctkvm.used,2),'ko','LineWidth',1);
title(strcat('PCTKVM, Number Vectors',{' '},num2str(nvec),{', '},'Error',{' '},num2str(erate*100)));
xlabel(strcat('Kullback-Leibler divergence original vs reduced space',{' '},num2str(loss)));
print("PCTKVM_Model","-depsc","-r1000")
hold off


figure
plot(Xs(Ys==-1,1),Xs(Ys==-1,2),'r.');
hold on
plot(Xs(Ys==1,1),Xs(Ys==1,2),'bx');
plot(Xs(modelsvm.SVs,1),Xs(modelsvm.SVs,2),'ko','LineWidth',1);
title(strcat('TKL-SVM, Number Vectors',{' '},num2str(modelsvm.totalSV),{', '},'Error',{' '},num2str(100-acc(1))));
xlabel(strcat('Kullback-Leibler divergence original vs reduced space',{' '},num2str(loss)));
print("TKLSVM_Model","-depsc","-r1000")
hold off
