DataFile=csvread('/Users/Tero529/Desktop/AI/Data.csv');
inputs=transpose(DataFile(:,2:7));
inputs=inputs(:,1:1748);
targets = transpose(DataFile(:,8));
targets= targets(:,2:1749);
algorithms = [ {'trainlm' } {'traingdm'} {'traingda'} { 'trainscg' } { 'trainoss'} ];
activations = [{'tansig'} {'purelin' } {'logsig'} ];
layers=2;
x=1:349;
TwoArch = [ 5 5 ;
            5 10 ;
            10 10 ;
            10 20 ;
            40 40 ;
            50 50;
            50 80;
            50 100 ; 
            ];
        
prefix ='/Users/Tero529/Desktop/AI/Results/TwoLayer/';
for m = 1 : length(activations)
   for k = 1: length(algorithms)
     for i = 1:8
         net=fitnet(TwoArch(i,:));
         net.trainFcn=char(algorithms(k));
         net.divideFcn = 'divideind';
         for j= 1: layers
            net.layers{j}.TransferFcn = char(activations(m));
         end
        net.divideParam.trainInd = 1 : 1119;
        net.divideParam.valInd = 1120 : 1399;
        net.divideParam.testInd = 1400 : 1748;
        [net,tr] = train(net,inputs,targets);
        outputs=net(inputs);
        errors  = gsubtract(targets,outputs);
        performance  = perform(net,targets,outputs)  ;
        trainTargets = targets .* tr.trainMask{1};
        valTargets   = targets .* tr.valMask{1};
        testTargets  = targets .* tr.testMask{1};
        trainPerformance = perform(net,trainTargets,outputs);
        valPerformance   = perform(net,valTargets,outputs);
        testPerformance  = perform(net,testTargets,outputs);
        f=figure();
        plotTitle =['Training Algorithm : ' , char(algorithms(k)),' Activation Function : ',char(activations(m))];
        plot(x,outputs(1400:1748),x,targets(1400:1748),'c--');
        title(plotTitle); 
        plotTitle=[char(algorithms(k)),'_',char(activations(m))];
        saveas(f,[prefix,num2str(i),'/','Relative/',plotTitle,'.png']);
        f=figure();
        plotregression(targets(1:1119),outputs(1:1119),'Training',targets(1120:1399),outputs(1120:1399),'Validation',targets(1400:1748),outputs(1400:1748),'Testing');
        saveas(f,[prefix,num2str(i),'/','Regression/',plotTitle,'.png']);
        f=figure();
        plotperform(tr);
        saveas(f,[prefix,num2str(i),'/','Performance/',plotTitle,'.png']);
        unconfigure(net);
     end
    end
end