clc;
clear;
close all;

load('oriData_mat.mat');

Input = NUM(:,1:18)';
Output = NUM(:,19:20);
[m,n] = size(NUM);
indices = crossvalind('Kfold',m,10);  % Use K-fold for Cross-validation ,k=10 
min_std_1=99999;
min_std_2=99999;

for i = 1:10
    disp(['This is the ',num2str(i),'th cross-validation']) 
    num_test = (indices == i);
    num_train = ~num_test;
    data_train = NUM(num_train,:);
    data_test = NUM(num_test,:);
    P = data_train(:,1:18);%extract the inputs from the dataset
    P_test = data_test(:,1:18);
    T_test = data_test(:,19:20);
    T1=data_train(:,19);%extract the outputs from the dataset
    T2=data_train(:,20);
    T1=T1-18;
    T2=T2-8;
    % Normalization
    P = P';
    for j = 1:18
        Max(1,j) = max(P(j,:));
        Min(1,j) = min(P(j,:));
        P(j,:)=2*((P(j,:)-Min(1,j))/(Max(1,j)-Min(1,j)));
    end
    n1=max(T1);% Maximum of Output 1
    n2=max(T2);% Maximum of Output 2
    T1=T1'/n1;
    T2=T2'/n2;
    a = 18;
    b = 1;
    n_h_1 = 20;
    n_h_2 = 40;
    % 4 differnt formulas to calculate the nodes of hidden layers
    n_h = fix(sqrt(a*(a+b))+b); 
    n_h = fix(sqrt(a+b)+8);
    n_h = fix(sqrt(0.43*a*b+0.12*a*a+2.54*b+0.77*a+0.35)+0.51);
    n_h = fix(log2(a));
    n_h_1 = n_h;
    n_h_2 = n_h;
    net_1 = newff(minmax(P),T1,[n_h_1 n_h_1 n_h_1], {'tansig' 'tansig' 'tansig' 'purelin'},'trainlm');%build the neural network 1 of 3 hidden layers
    net_2 = newff(minmax(P),T2,[n_h_2 n_h_2 n_h_2], {'tansig' 'tansig' 'tansig' 'purelin'},'trainlm');%build the neural network 2 of 3 hidden layers
     
    %Parameter settings
    lr = maxlinlr(P);
    net_1.trainParam.show = 1; %the period of showing the results
    net_1.trainParam.lr = maxlinlr(P);%Learning rates
    net_1.trainParam.mc = 0.95; %Momentum constant
    net_1.trainParam.epochs = 100000;%Maximun of iterations
    net_1.trainParam.goal = 0.000001;%the minimum error expected to get
    net_2.trainParam.show = 1; 
    net_2.trainParam.lr = maxlinlr(P);
    net_2.trainParam.mc = 0.95;
    net_2.trainParam.epochs = 100000;
    net_2.trainParam.goal = 0.000001;

    % Network training
    [net_1,tr_1]=train(net_1,P,T1);
    [net_2,tr_2]=train(net_2,P,T2);

    %Simulation
    for j = 1:18
        input(j,:) = 2*((Input(j,:)-Min(1,j))/(Max(1,j)-Min(1,j)));
    end
    output_1 = sim(net_1,input);
    output_2 = sim(net_2,input);
    R_1 = Output(:,1)';
    R_2 = Output(:,2)';
    %Anti-normalization and Calculate the error 
    E_1=output_1*n1+18-R_1;
    E_1=E_1/n1;
    E_2=output_2*n2+8-R_2;
    E_2=E_2/n2;
    E_1_std=std(E_1,0,2);
    E_2_std=std(E_2,0,2);
    %preserve the best result by calculating Standard Deviation
    if E_1_std < min_std_1
        min_std_1=E_1_std;
        E_m1=E_1;
        output_m1=output_1;
        i_m_1 = i;
    end
    if E_2_std < min_std_2
       min_std_2=E_2_std;
       E_m2=E_2;
       output_m2=output_2;
       i_m_2 = i;
    end

end   
%Compare the Estimated results and the Original results
figure(1);
subplot(2,1,1)
plot(Output(:,1)','r');
hold on;
plot(output_m1*n1+18,'b');
hold on;
subplot(2,1,2)
plot(Output(:,2)','r');
hold on;
grid on;
axis([0 450 5 15]);
plot(output_m2*n2+8,'b');
figure(2)
plot(E_m1,'r');
hold on;
plot(E_m2,'b');
axis([0 450 -1 1]);