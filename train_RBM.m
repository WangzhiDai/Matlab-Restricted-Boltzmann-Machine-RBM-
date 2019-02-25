% train_RBM: train the RBM model with contrastive divergence method on
% training set
%
% rbm = train_RBM(rbm, V, opts)
%
%
%Output parameters:
% rbm: the restricted boltzmann machine (RBM) model
%
%
%Input parameters:
% rbm: the initial boltzmann machine (RBM) model
% V: visible (input) variables, where # of row is number of data and # of col is # of visible (input) nodes
% opts (optional): options
%
% options (defualt value):
%  opts.MaxIter: Maxium iteration number (100)
%  opts.InitialMomentum: Initial momentum until InitialMomentumIter (0.5)
%  opts.InitialMomentumIter: Iteration number for initial momentum (5)
%  opts.FinalMomentum: Final momentum after InitialMomentumIter (0.9)
%  opts.WeightCost: Weight cost (0.0002)
%  opts.StepRatio: Learning step size (0.01)
%  opts.BatchSize: # of mini-batch data (# of all data)
%  opts.Verbose: verbose or not (false)
%  opts.CD_K: K steps of contrast divergence (10)
%  opts.PCD: if persistent contrast divergence (0)
%  opts.data_test: test data during training, to monitor test set
%  likelihood or imputation accuracy.
%
%
%Example:
% datanum = 1024;
% outputnum = 16;
% inputnum = 4;
% 
% inputdata = rand(datanum, inputnum);
% outputdata = rand(datanum, outputnum);
% 
% rbm = rand_RBM(inputnum, outputnum);
% rbm = rain_RBM( rbm, inputdata );


function [rbm, pause_iter,train_loglikelihood,test_loglikelihood, acc,acc_lowb]= train_RBM(rbm, V, opts )
% Important parameters
InitialMomentum = 0.5;     % momentum for first five iterations
FinalMomentum = 0.9;       % momentum for remaining iterations
WeightCost = 0;       % costs of weight update
InitialMomentumIter = 5;
MaxIter = 200;
StepRatio = 0.2;
BatchSize = 0;
Verbose = true;
CD_K=10;
PCD=0;

if( exist('opts' ) )
    if( isfield(opts,'MaxIter') )
    MaxIter = opts.MaxIter;
    end
    if( isfield(opts,'InitialMomentum') )
    InitialMomentum = opts.InitialMomentum;
    end
    if( isfield(opts,'InitialMomentumIter') )
    InitialMomentumIter = opts.InitialMomentumIter;
    end
    if( isfield(opts,'FinalMomentum') )
    FinalMomentum = opts.FinalMomentum;
    end
    if( isfield(opts,'WeightCost') )
    WeightCost = opts.WeightCost;
    end
    if( isfield(opts,'StepRatio') )
    StepRatio = opts.StepRatio;
    end
    if( isfield(opts,'BatchSize') )
    BatchSize = opts.BatchSize;
    end
    if( isfield(opts,'Verbose') )
    Verbose = opts.Verbose;
    end
    if( isfield(opts,'CD_K') )
    CD_K = opts.CD_K;
    end
    if( isfield(opts,'PCD') )
    PCD = opts.PCD;
    end
    if( isfield(opts,'data_test') )
    data_test = opts.data_test;
    end
end

num_p = size(V,1);
dimH = size(rbm.hidbias, 2);
dimV = size(rbm.visbias, 2);

if( BatchSize <= 0 )
  BatchSize = num_p;
end

deltaW = zeros(dimV, dimH);
deltaHid = zeros(1, dimH);
deltaVis = zeros(1, dimV);

if( Verbose ) 
    timer = tic;
end

%Main training loop
fprintf('Training Started \nMax Iteration: %3d \nLearning Rate: %.4f\nBatchSize: %3d\nCD_K: %3d\nPCD %3d\n',... 
        MaxIter, StepRatio, BatchSize, CD_K,PCD);
pause_step=0;
for iter=1:MaxIter   
    % Set momentum
    if( iter <= InitialMomentumIter )
        momentum = InitialMomentum;
    else
        momentum = FinalMomentum;
    end
    
    % Randomize order of training set
    ind = randperm(num_p);
    
    for batch=1:BatchSize:num_p
		bind = ind(batch:min([batch + BatchSize - 1, num_p]));
        
        % Gibbs sampling step 0  %compute v,h of data
        vis0 = double(V(bind,:)); % Set values of visible nodes (input)
        hid0 = v2h( rbm, vis0 );  % Compute hidden nodes (expected hidden given input)
        vh_data = (vis0' * hid0)./BatchSize;
        v_data = sum(vis0,1)./BatchSize;
        h_data = sum(hid0,1)./BatchSize;
        
        % Get model means by sampling
        if strcmp(Sample_Method,'gibbs')
            chain_num=BatchSize;
            [vh_model,~,v_model,h_model]=gibbs_sampling_srbm(rbm,vis0(1:chain_num,:),hid0(1:chain_num,:),CD_K);
            vh_model=vh_model./chain_num;
            v_model=v_model./chain_num;
            h_model=h_model./chain_num;
        else
            chain_num=5;
            maxIter=1e6;
            [vh_model,~,v_model,h_model]=metropolis_sampling_srbm(rbm,vis0(1:chain_num,:),hid0(1:chain_num,:),maxIter); 
            vh_model=vh_model./chain_num;
            vv_model=vv_model./chain_num;
            v_model=v_model./chain_num;
            h_model=h_model./chain_num;
        end

		% Compute the weights update by contrastive divergence
        dW = (vh_data - vh_model);
        dHid = (h_data - h_model);
        dVis = (v_data - v_model);
        
		deltaW = momentum * deltaW + StepRatio * dW;
        deltaL = momentum * deltaL + StepRatioL * dL;
		deltaHid = momentum * deltaHid + StepRatio * dHid;
		deltaVis = momentum * deltaVis + StepRatio * dVis;

		% Update the network weights
		rbm.W = rbm.W + deltaW - WeightCost * rbm.W;
		rbm.hidbias = rbm.hidbias + deltaHid;
		rbm.visbias = rbm.visbias + deltaVis;
    end
    
    if( Verbose )
        
        if mod(iter,100)==0
            pause_step=pause_step+1;
            pause_iter(pause_step)=iter;
            train_loglikelihood(pause_step)=rbm_likelihood(rbm,V);
            H = v2h( rbm, V );
            Vr = h2v( rbm, H );
            err = power( V - Vr, 2 );
            rmse = sqrt( sum(err(:)) / numel(err) );      
            totalti = toc(timer);
            aveti = totalti / iter;
            estti = (MaxIter-iter) * aveti;
            eststr = datestr(datenum(0,0,0,0,0,estti),'DD:HH:MM:SS');  
            if( isfield(opts,'data_test') )
                test_loglikelihood(pause_step)=rbm_likelihood(rbm,data_test);
                if ( isfield(opts,'data_test_missing') )
                    acc(pause_step)=test_one_rbm(rbm,opts.data_test_missing,data_test,'direct');
                    acc_lowb(pause_step)=test_one_rbm(rbm,opts.data_test_missing_lowb,data_test,'direct');
                    fprintf( '%3d : %9.4f %9.4f %9.4f %9.4f %9.4f %9.4f %s\n', iter, train_loglikelihood(pause_step), test_loglikelihood(pause_step),acc(pause_step), acc_lowb(pause_step), mean(H(:)), aveti, eststr );
                else
                    fprintf( '%3d : %9.4f  %9.4f %9.4f %9.4f %s\n', iter, train_loglikelihood(pause_step), test_loglikelihood(pause_step),mean(H(:)), aveti, eststr );
                end
            else
                fprintf( '%3d : %9.4f %9.4f %9.4f %s\n', iter, train_loglikelihood(pause_step), mean(H(:)), aveti, eststr );
            end
        end
    end
end
end

