
gridP300 = ['A','B','C','D','E','F';
        'G','H','I','J','K','L';
        'M','N','O','P','Q','R';
        'S','T','U','V','W','X'
        'Y','Z','1','2','3','4';
        '5','6','7','8','9',' ']

% Butterworth filter
[B,A] = butter(2,[.1,10]/128);
bwfiltered = filtfilt(B,A,A01.X);                    % zero-phase filtering
%plot(A01.X,'k-.'); grid on ; hold on
%plot(bwfiltered,'LineWidth',1.5); hold on

% Notch filter
[num,den]=iirnotch(50/128,50/128); % notch filter implementation 
bwnotchfiltered = filter(num,den, bwfiltered);
%plot(bwnotchfiltered,'LineWidth',1.5);

numtrain = 2400;
numtest = 1800;
total = 4200;
numchannels = 8;
epochsize = 200;

%For train data set

a = find(A01.y>0);
b = find(A01.y>1);

Xtrain = zeros(numtrain,epochsize,numchannels);
Ytrain = zeros(numtrain,1);
YtrainRCnum = zeros(numtrain,1);

j = 1;

for i = 1:numtrain
    if(A01.y(a(j)) == 1)
        Ytrain(i) = 1;  %Non Target
    else
        Ytrain(i) = 2;  %Target
    end
    
    YtrainRCnum(i) = A01.y_stim(a(j));
    
    %Xtrain(i,:,:) = A01.X(a(j):a(j)+199,:);
    Xtrain(i,:,:) = bwnotchfiltered(a(j):a(j)+199,:);

    % Baseline correction
    
    for r = 1:numchannels
        tofindmed1 = Xtrain(i,1:10,r);
        sortedarray1 = sort(tofindmed1);
        median1 = sortedarray1(5);
        for s = 1:epochsize
            Xtrain(i,s,r) = Xtrain(i,s,r) - median1; 
        end
    end

    j = j + 32;

end

% For test data set

Xtest = zeros(numtest,epochsize,numchannels);
Ytest = zeros(numtest,1);
YtestRCnum = zeros(numtest,1);

for i =1:numtest
    if(A01.y(a(j)) == 1)
        Ytest(i) = 1;  %Non Target
    else
        Ytest(i) = 2;  %Target
    end
    
    YtestRCnum(i) = A01.y_stim(a(j));
    
    %Xtest(i,:,:) = A01.X(a(j):a(j)+199,:);
    Xtest(i,:,:) = bwnotchfiltered(a(j):a(j)+199,:);
    
    % Baseline correction
    
    for r = 1:numchannels
        tofindmed2 = Xtest(i,1:10,r);
        sortedarray2 = sort(tofindmed2);
        median2 = sortedarray2(5);
        for s = 1:epochsize
            Xtest(i,s,r) = Xtest(i,s,r) - median2; 
        end
    end
    
    j = j + 32;
end


% Down Sampling

epochsizeDS = 20;

XtrainDS = zeros(numtrain, epochsizeDS, numchannels);

for u = 1:numtrain
    obtain1 = Xtrain(u,:,:);
    obtain1 = squeeze(obtain1);
    for v = 1:numchannels
        count = 1;
        for w = 1:(epochsize/epochsizeDS):epochsize
            tempres1 = sort(squeeze(obtain1(w:w+(epochsize/epochsizeDS)-1,v)));
            XtrainDS(u,count,v) = tempres1((epochsize/(2*epochsizeDS)));
            count = count + 1;
        end
    end
end

XtestDS = zeros(numtest, epochsizeDS, numchannels);

for u = 1:numtest
    obtain2 = Xtest(u,:,:);
    obtain2 = squeeze(obtain2);
    for v = 1:numchannels
        count = 1;
        for w = 1:(epochsize/epochsizeDS):epochsize
            tempres2 = sort(squeeze(obtain2(w:w+(epochsize/epochsizeDS)-1,v)));
            XtestDS(u,count,v) = tempres2((epochsize/(2*epochsizeDS)));
            count = count + 1;
        end
    end
end


% Convert 3D to 2D

%Xtrain_new = zeros(numtrain, numchannels*epochsize);
Xtrain_new = zeros(numtrain, numchannels*epochsizeDS);

for j = 1:numtrain
    %mat1 = Xtrain(j,:,:);
    mat1 = XtrainDS(j,:,:);
    mat1 = mat1(:);
    mat1 = mat1';
    Xtrain_new(j,:) = mat1;
end

%Xtest_new = zeros(numtest, numchannels*epochsize);
Xtest_new = zeros(numtest, numchannels*epochsizeDS);

for j = 1:numtest
    %mat2 = Xtest(j,:,:);
    mat2 = XtestDS(j,:,:);
    mat2 = mat2(:);
    mat2 = mat2';
    Xtest_new(j,:) = mat2;
end

% SVMStruct = fitcsvm(Xtrain_new, Ytrain, 'KernelFunction', 'linear');
% YSVM = predict(SVMStruct, Xtest_new);
% Accuracy = (nnz((YSVM==Ytest))/numel(YSVM))*100

DAStruct = fitcdiscr(Xtrain_new, Ytrain, 'discrimType', 'pseudoLinear');
YDA = predict(DAStruct, Xtest_new);
Accuracy = (nnz((YDA==Ytest))/numel(YDA))*100

% SDAStruct = stepwisefit(Xtrain_new, Ytrain);
% YSDA = predict(SDAStruct, Xtest_new);
% Accuracy = (nnz((SYDA==Ytest))/numel(SYDA))*100

% NVStruct = fitcnb(Xtrain_new, Ytrain);
% YNB = predict(NVStruct, Xtest_new);
% Accuracy = (nnz((YNB==Ytest))/numel(YNB))*100

% DTStruct = fitctree(Xtrain_new, Ytrain);
% YDT = predict(DTStruct, Xtest_new);
% Accuracy = (nnz((YDT==Ytest))/numel(YDT))*100

% KNNStruct = fitcknn(Xtrain_new, Ytrain);
% YKNN = predict(KNNStruct, Xtest_new);
% Accuracy = (nnz((YKNN==Ytest))/numel(YKNN))*100

[confmat, order] = confusionmat(Ytest, YDA);
confmat


% Form units of 120 y's

unitsof120Yactual = zeros(numtest/120,120);  % made using Ytest
unitsof120Ypredicted = zeros(numtest/120,120);   % made using YDA/YSVM/whichever classifier
unitsof120YtestRCnum = zeros(1,120);  % stores 1-12 values in 15x120 format - all 15 are identical thus use only 1 row


count = 1;

for c = 1:120:numtest
    temp1 = Ytest(c:(c+120-1));
    temp2 = YDA(c:(c+120-1));
    
    if count==1
        temp3 = YtestRCnum(c:(c+120-1));
        temp3 = temp3';
        unitsof120YtestRCnum(count,:) = temp3;
    end
    
    temp1 = temp1';
    temp2 = temp2';
    
    unitsof120Yactual(count,:) = temp1;
    unitsof120Ypredicted(count,:) = temp2;
    
    count = count + 1;
end


countof2sactual = zeros(numtest/120,1);
characteractual = zeros(numtest/120,1);

for d = 1:numtest/120

    store1 = find(unitsof120Yactual(d,:)==2);
    get1 = nnz(unitsof120Yactual(d,:)==2);   % 20 (as ideal)
    countof2sactual(d,:) = get1;
    RCnumvalueactual = unitsof120YtestRCnum(store1);
    
    [n,bin] = hist(RCnumvalueactual,unique(RCnumvalueactual));
    [~,idx] = sort(-n);
    freq1 = n(idx); % count instances
    val1 = bin(idx); % corresponding values
    
    % Shouldnt be ties i=here, since ideal case
    flag1 = 1;
    flag2 = 1;
    for e = 1:length(val1)
        if(flag1==2 && flag2==2)
            break
        end
        
        if(val1(e)<=6 && flag1==1)
            colnum = val1(e);
            flag1 = 2;
        elseif(val1(e)>6 && flag2==1)
            rownum = val1(e) - 6;
            flag2 = 2;
        end
    end
    
    gridP300(rownum,colnum) % Display the character
    characteractual(d,1) = gridP300(rownum,colnum);
end


countof2spredicted = zeros(numtest/120,1);
characterpredicted = zeros(numtest/120,1);

for d = 1:numtest/120

    store2 = find(unitsof120Ypredicted(d,:)==2);
    get2 = nnz(unitsof120Ypredicted(d,:)==2);   % will not all be 20 due to prediction errors
    countof2spredicted(d,:) = get2;
    RCnumvaluepredicted = unitsof120YtestRCnum(store2);
     
    [n,bin] = hist(RCnumvaluepredicted,unique(RCnumvaluepredicted));
    [~,idx] = sort(-n);
    freq2 = n(idx); % count instances
    val2 = bin(idx); % corresponding values     

    % Shouldnt be ties i=here, since ideal case
    flag1 = 1;
    flag2 = 1;
    for e = 1:length(val2)
        if(flag1==2 && flag2==2)
            break
        end
        
        if(val2(e)<=6 && flag1==1)
            colnum = val2(e);
            flag1 = 2;
        elseif(val2(e)>6 && flag2==1)
            rownum = val2(e) - 6;
            flag2 = 2;
        end
    end
     
    gridP300(rownum,colnum) % Display the character
    characterpredicted(d,1) = gridP300(rownum,colnum);

end






% MY OWN CLASSIFIER BASED ON PLOT - Didnt work, since too much overlap

% one = zeros(numtest*numchannels,1);
% two = zeros(numtest*numchannels,1);
% for k = 1:numtest
%     matrix1 = Xtest(k, :, :);
%     matrix1 = squeeze(matrix1);
%     matrix1 = matrix1';
%     summat = sum(matrix1,2);
%     for l = 1:numchannels
%         one(numchannels*(k-1)+l) = l;
%         two(numchannels*(k-1)+l) = summat(l)/epochsize;
%     end
% end
% 
% three = repmat(Ytest', numchannels, 1);
% three = three(:);
% 
% Ymy = zeros(numtest,1);
% for m = 1:numtest
%     matrix2 = Xtest(m, :, :);
%     matrix2 = squeeze(matrix2);
%     matrix2 = matrix2';
%     summat = (sum(matrix2,2))/epochsize;
%     sumofsummat = sum(summat,1)
%     var = sumofsummat/numchannels
%     if(var>-0.75 && var<0.75)
%         Ymy(m) = 2;
%     else
%         Ymy(m) = 1;
%     end 
% end
% 
% nnz(Ymy==Ytest)
% numel(Ymy)
% Accuracy = (nnz(Ymy==Ytest)/numel(Ymy))*100
% figure;
% gscatter(one, two, three);
  
% END OF MY OWN CLASSIFIER


% PLOT AS SIR TOLD AND AS GIVEN IN PUBLICATION

% Collect all epochsizexnumchannels with y = 1 and collect all epochsizexnumchannels with y = 2. Find average ie one epochsizexnumchannels for both
% and plot numchannels graphs for each (total 16) for all electrodes separately.

% totalx = zeros(total,epochsize,numchannels);
% totalx(1:numtrain,:,:) = Xtrain;
% totalx(numtrain+1:total,:,:) = Xtest;
% totaly = zeros(total,1);
% totaly(1:numtrain,:) = Ytrain;
% totaly(numtrain+1:total,:) = Ytest;
% 
% indices1 = find(totaly==1);
% indices2 = find(totaly==2);
% 
% yequalsone = totalx(indices1,:,:);
% yequalstwo = totalx(indices2,:,:);
% 
% sumyequalsone = zeros(epochsize,numchannels);
% for n = 1:(5*total/6)
%     temp1 = yequalsone(n,:,:);
%     temp1 = squeeze(temp1);
%     sumyequalsone = sumyequalsone + temp1;
% end
% 
% sumyequalsone = sumyequalsone/(5*total/6);
% 
% sumyequalstwo = zeros(epochsize,numchannels);
% for o = 1:(total/6)
%     temp2 = yequalstwo(o,:,:);
%     temp2 = squeeze(temp2);
%     sumyequalstwo = sumyequalstwo + temp2;
% end
% 
% sumyequalstwo = sumyequalstwo/(total/6);
% 
% for p = 1:numchannels
%     figure;
%     plot(sumyequalsone(:,p));
%     figure;
%     plot(sumyequalstwo(:,p));
% end

% END OF PLOT AS SIR TOLD   