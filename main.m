%%%****************************************************************************************
%%%* This is the source code for the paper "Multi-step-ahead Stock Price Prediction Using *
%%%* Recurrent Fuzzy Neural Network and Variational Mode Decomposition" (VMD-MFRFFN)      *  
%%%* Authors: Hamid Nasiri, Mohammad Mehdi Ebadzadeh                                      *
%%%****************************************************************************************

rng(1); % For reproducibility of results

global nStates;
global nData;
global nRules_Output;
global rbarOutput;
global rbarState;
global currentIndividual;
global targetOutput;


%% Load HSI Benchmark
fprintf("Loading Hang Seng Index Dataset...\n");
maxNumberOfIMFS = 9;

TargetDimension = 1;
load('Benchmarks\\HSI_Index.mat');
% Test Data
testOutputTemp = data(end-375:end,TargetDimension);

% Train Data
data = data(1:end-377,TargetDimension);

finalPrediction = [];
fprintf("Decomposing Train Data Using VMD...\n");
[imf,residual] = vmd(data,'NumIMF',maxNumberOfIMFS);
fprintf("Decomposing Test Data Using VMD...\n");
[imf_test, residual_test] = vmd(testOutputTemp,'NumIMF',maxNumberOfIMFS);

numberOfRules = [6 6 6 5 5 5 4 4 4];

for numberOfIMFS = 1:maxNumberOfIMFS

    data = imf(:,numberOfIMFS);
    data_test = imf_test(:,numberOfIMFS);
    
    tempdata = data;
    tempdata_test = data_test;
    data = (data - min(data)) ./ (max(data)-min(data)) ;
    data_test = (data_test - min(data_test)) ./ (max(data_test)-min(data_test)) ;

    trainInput = data(1:end-1,TargetDimension);
    targetOutput = data(2:end,TargetDimension);
    testInput = data_test(1:end-1,TargetDimension);
    testOutput = data_test(2:end,TargetDimension);


    % ****************************************************
    % *                  Parameters                      *
    % ****************************************************

    coverageRules = false; % Generating Fuzzy Rules Using Coverage?
    ruleSigma = 0.3;
    coverageThreshold = 0.2;
    nRules_Output = numberOfRules(numberOfIMFS);
    nRules_State = 2;
    nStates = 2;
    mfType = "trimf"; % trimf gaussmf gauss2mf gbellmf 
    nFuzzySetsOutput = numberOfRules(numberOfIMFS);
    nFuzzySetsState = 2;
    PSO_SwarmSize = 25;
    PSO_MaxIteration = 2;
    nData = size(trainInput,1);
    nDimensions = size(trainInput,2);

    %% Create Fuzzy Rules
    if ~coverageRules
        if mfType == "gaussmf" 
            mfOutput = createGaussianMembershipFunction(nFuzzySetsOutput);
            mfState = createGaussianMembershipFunction(nFuzzySetsState);
        elseif mfType == "trimf"
            mfOutput = createMembershipFunction(nFuzzySetsOutput);
            mfState = createMembershipFunction(nFuzzySetsState);
        elseif mfType == "gauss2mf"
            mfOutput = createGaussian2MembershipFunction(nFuzzySetsOutput);
            mfState = createGaussian2MembershipFunction(nFuzzySetsState);
        elseif mfType == "gbellmf"
            mfOutput = createGeneralizedBellshapeMembershipfunction(nFuzzySetsOutput);
            mfState = createGeneralizedBellshapeMembershipfunction(nFuzzySetsState);    
        end
        mfOutput = repmat(mfOutput,1,nDimensions);
        mfState = repmat(mfState,1,nDimensions);
    else
        if ~exist('mf','var')
            if mfType == "gaussmf"
                generateGaussianFuzzyRulesUsingCoverage;
            elseif mfType == "trimf"
                generateTriangleFuzzyRulesUsingCoverage;
            end
        end
        mfOutput = mf;
        mfState = mf;
        nFuzzySetsOutput = length(mf);
        nFuzzySetsState = length(mf);

        nRules_Output = nFuzzySetsOutput;
        nRules_State = nFuzzySetsState;
    end

    fprintf("Calculating Membership Values...\n");
    calculateMembershipValues;


    %% Initialization

    % W -> Weight of Output Network
    % V -> Weight of State Network
    currentIndividual.V = rand(nStates,nRules_State);
    currentIndividual.W = rand(nStates,nRules_Output);
    costCalculation;

    %% Training Network

    fprintf("Training Network Using PSO For IMF%d ...",numberOfIMFS); 

    fun = @(x) objectiveFunction(x);
    lb = zeros(1,nRules_State*nStates);
    ub = ones(1,nRules_State*nStates);
    options = optimoptions('particleswarm','SwarmSize',PSO_SwarmSize,'Display','iter','MaxIterations',PSO_MaxIteration);     %patternsearch fmincon  
    [x, fval] = particleswarm(fun,nRules_State*nStates,lb,ub,options);    

    currentIndividual.V = x;   
    costCalculation   
    calculatingTestError;   

    finalPrediction = [finalPrediction ; scaledPredicted'];

end

predictedOutput = sum(finalPrediction);

predictedOutput = predictedOutput';
testRMSE = sqrt(immse(predictedOutput,testOutputTemp(2:end)));
testMAPE = mean(abs((predictedOutput-testOutputTemp(2:end))./testOutputTemp(2:end))*100);

fprintf("One-step-ahead Prediction Results:\n");
fprintf("RMSE = %e\n",testRMSE);
fprintf("MAPE = %e\n",testMAPE);
