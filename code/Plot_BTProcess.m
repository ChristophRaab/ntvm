%% TWO Labels
% 
Z= [randn(100,3)+3; randn(100,3)+5];
Ys = [ones(100,1); ones(100,1)*2];

X= [randn(100,3)-5; randn(100,3)-3];
Yt = [ones(100,1)*3; ones(100,1)*4];

D = [Z;X];
Y = [Ys;Yt];
color_1 = [1 1 0]; % yellow: source label1
color_2 = [1 0 1];  %magenta: source label2
color_3 = [0 1 0]; %green: target label1
color_4 = [0 0 1]; %blue: target label2

cmap = [color_1;color_2;color_3; color_4;];
label = cmap(Y,:);

figure;
hold on;
set(gca,'color','none') 
scatter3(D(find(Y==1),1),D(find(Y==1),2),D(find(Y==1),3),'x','b');
scatter3(D(find(Y==2),1),D(find(Y==2),2),D(find(Y==2),3),'*','g');
scatter3(D(find(Y==3),1),D(find(Y==3),2),D(find(Y==3),3),'*','r');
scatter3(D(find(Y==4),1),D(find(Y==4),2),D(find(Y==4),3),'x','m');
xlabel('x')
ylabel('y')
zlabel('z')
view(-40,30)
grid on
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
print("DataUnormalized","-depsc","-r1000")
hold off;

Z = zscore(Z); X = zscore(X);
D = [Z;X];
Y = [Ys;Yt];
figure;
hold on;
set(gca,'color','none') 
s1=scatter3(D(find(Y==1),1),D(find(Y==1),2),D(find(Y==1),3),'x','b');
s2=scatter3(D(find(Y==2),1),D(find(Y==2),2),D(find(Y==2),3),'*','g');
s3=scatter3(D(find(Y==3),1),D(find(Y==3),2),D(find(Y==3),3),'*','r');
s4=scatter3(D(find(Y==4),1),D(find(Y==4),2),D(find(Y==4),3),'x','m');
xlabel('x')
ylabel('y')
zlabel('z')
view(-40,30)
grid on
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
print("DataNormalized","-depsc","-r1000")
hold off;

[U,S,V] = svd(Z);
[~,ZS,~] = svd(X);
Z = U*ZS*V';

D = [Z;X];
figure;
hold on;
set(gca,'color','none') 
scatter3(D(find(Y==1),1),D(find(Y==1),2),D(find(Y==1),3),'*','g');
scatter3(D(find(Y==2),1),D(find(Y==2),2),D(find(Y==2),3),'x','b');
scatter3(D(find(Y==3),1),D(find(Y==3),2),D(find(Y==3),3),'*','r');
scatter3(D(find(Y==4),1),D(find(Y==4),2),D(find(Y==4),3),'x','m');
xlabel('x')
ylabel('y')
zlabel('z')
view(-40,30)
grid on
ax = gca;
outerpos = ax.OuterPosition;
ti = ax.TightInset; 
left = outerpos(1) + ti(1);
bottom = outerpos(2) + ti(2);
ax_width = outerpos(3) - ti(1) - ti(3);
ax_height = outerpos(4) - ti(2) - ti(4);
ax.Position = [left bottom ax_width ax_height];
print("BasisTransfer","-depsc","-r1000")
hold off; 
