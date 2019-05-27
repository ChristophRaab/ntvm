% % This script visualizes the problem of transfer learning with and synthetic
% % gaussian dataset
% Orginal space
close all;
clear all; 
Z= [randn(100,3); randn(100,3)+2];
Ys = [ones(100,1); ones(100,1)*2];

X= [randn(100,3)-5; randn(100,3)-7];
Yt = [ones(100,1)*3; ones(100,1)*4];

D = [Z;X];
Y = [Ys;Yt];
color_1 = [0 1 0]; % yellow: source label1
color_2 = [0 0 1];  %magenta: source label2
color_3 = [0 1 0]; %green: target label3
color_4 = [0 0 1]; %blue: target label4

cmap = [color_1; color_2; color_3;color_4];
label = cmap(Y,:);
 
figure;
hold on;
scatter3(D(find(Y==1),1),D(find(Y==1),2),D(find(Y==1),3),'o','r');
scatter3(D(find(Y==2),1),D(find(Y==2),2),D(find(Y==2),3),'o','b');
scatter3(D(find(Y==3),1),D(find(Y==3),2),D(find(Y==3),3),'x','b');
scatter3(D(find(Y==4),1),D(find(Y==4),2),D(find(Y==4),3),'x','r');
xlabel("x_1");
ylabel("x_2");
print("Homogenoues_Transfer_Problem","-depsc","-r1000")

Z = zscore(Z,1); X = zscore(X,1);

D = [Z;X];
Y = [Ys;Yt];
figure;
hold on;
scatter3(D(find(Y==1),1),D(find(Y==1),2),D(find(Y==1),3),'o','b');
scatter3(D(find(Y==2),1),D(find(Y==2),2),D(find(Y==2),3),'o','r');
scatter3(D(find(Y==3),1),D(find(Y==3),2),D(find(Y==3),3),'x','b');
scatter3(D(find(Y==4),1),D(find(Y==4),2),D(find(Y==4),3),'x','r');
xlabel("x_1");
ylabel("x_2")
print("DataNormalized","-depsc","-r1000")


[~,ZS,~] = svd(Z);
[U,~,V] = svd(X);
Z = (U*ZS*V')+0.1*randn
D = [Z;X];

Y = [Ys;Yt];
figure;
hold on;
scatter3(D(find(Y==1),1),D(find(Y==1),2),D(find(Y==1),3),'o','b');
scatter3(D(find(Y==2),1),D(find(Y==2),2),D(find(Y==2),3),'o','r');
scatter3(D(find(Y==3),1),D(find(Y==3),2),D(find(Y==3),3),'x','b');
scatter3(D(find(Y==4),1),D(find(Y==4),2),D(find(Y==4),3),'x','r');
xlabel("x_1");
ylabel("x_2")
print("BasisTransfer","-depsc","-r1000")


% 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Z= randn(200,3)+2;
Ys = ones(200,1);
X= randn(200,3);
Yt = ones(200,1)*2;

D = [Z;X];
Y = [Ys;Yt];
cmap = [color_1; color_2;];
label = cmap(Y,:);

figure;
hold on;
scatter3(D(find(Y==1),1),D(find(Y==1),2),D(find(Y==1),3),'o','r');
scatter3(D(find(Y==2),1),D(find(Y==2),2),D(find(Y==2),3),'o','b');
scatter3(D(find(Y==3),1),D(find(Y==3),2),D(find(Y==3),3),'o','r');
scatter3(D(find(Y==4),1),D(find(Y==4),2),D(find(Y==4),3),'o','b');
xlabel("x_1");
ylabel("x_2")
print("Traditional_Problem","-depsc","-r1000")
% 
% 
