DataFile=csvread('/Users/Tero529/Desktop/AI/Data.csv');
inputs=transpose(DataFile(:,2:7));
inputs=inputs(:,1:1748);
targets = transpose(DataFile(:,8));
targets= targets(:,2:1749);
net=fitnet([5 5]);
net.trainFcn='trainlm';
x=1:70;
predictFile=csvread('/Users/Tero529/Desktop/AI/inputs.csv');
toPredict= transpose(predictFile(:,1:6));
actual=transpose(predictFile(2:71,7));
net.divideFcn = 'divideind';
net.layers{1}.TransferFcn = 'purelin';
net.layers{2}.TransferFcn = 'purelin';
net.divideParam.trainInd = 1 : 1119;
net.divideParam.valInd = 1120 : 1399;
net.divideParam.testInd = 1400 : 1748;
[net,tr] = train(net,inputs,targets);
outputs= net(toPredict);
plot(x,outputs(:,1:70),x,actual,'b--');
fid=fopen('myfile.txt','w');
fprintf(fid,outputs);
fclose(fid);
disp(outputs);