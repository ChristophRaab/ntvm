function [erate, nvec, y_sign, y_prob] = pctkvm_predict(trainY,testY,model)

sizeM = size(model,2);

if sizeM == 1
    m = size(trainY,1);
   
    K = model.K(m+1:end, 1:m);
    w = model.w;
    b = model.b;
    used = model.used;
    
    weights = w(used).*trainY(used);
    
    % Compute RVM over test data and calculate error
    PHI	= K(:,used);
    test_num = size(K,1);
    
    Y_regress = PHI*weights+b*ones(test_num,1);
    y_sign	= sign(Y_regress);
    
    % the probablistic output
    y_prob = normcdf(Y_regress);
    
    errs	= sum(y_sign(testY== -1)~=-1) + sum(y_sign(testY==1)~=1);
    erate = errs/test_num;
    nvec = length(used);
elseif sizeM > 2
    fprintf('\nMulticlass Prediction detected! Splitting up test data..\n')
    usedVectors = [];
    multiLabels = [];
    multiProb = [];
    %     multiProb = zeros(size(testY,1),sizeM);
    
    % For-Loop to calculate One vs One prediction
    for i = 1:size(model,2)
        
        % Taking the corrosponding labels from original train label vector
        oneIndx = find(trainY == model(i).one);
        twoIndx = find(trainY == model(i).two);
        
        % Merge the label vectors into one training vector
        trainYOR = [ones(size(oneIndx,1),1); ones(size(twoIndx,1),1)*-1];
        
        m = size(trainYOR,1);
        
        % Taking the lower left square for prediction
    
        [erate, nvec, label, y_prob] = pctkvm_predict(trainYOR,testY,model(i));
        
        label(find(label==1)) = model(i).one;
        label(find(label==-1)) = model(i).two;
        
        multiLabels = [multiLabels label];
        multiProb = [multiProb y_prob];
        
        vectors =model(i).used;
        
        usedVectors = [usedVectors; vectors];
        
    end
    
    nvec = size(unique(usedVectors),1);
    [y_sign,F,C] = mode(multiLabels,2);
    
    % Calculate mean over the prob estimation for the probs with the
    % voted label
    usedLabels  = multiLabels - y_sign;
    multiProb(find(usedLabels ~= 0)) = NaN;
    y_prob = nanmean(multiProb,2);
    
    resulterror = abs(testY-y_sign);
    erate = size(resulterror(resulterror ~=0),1) / size(testY,1);
    
    fprintf('\nPCTKVM Acc: %f\n',1-erate);
else
    fprintf('\nWrong Input of Models\n');
end


