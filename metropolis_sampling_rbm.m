function [v_sample,running_avg] =  metropolis_sampling_rbm(v,model,maxIter,tolerance)
%Metropolis sampling for an RBM.
%   [v_sample, running_avg] =  metropolis_sampling_rbm(v,model,maxIter) generates
%   conditional samples from an RBM model with metropolis samlping method.
%   v is a vector with observed and missing entries. v_sample returns
%   samples for the missing entries with the observed entries fixed.
%   running_avg is the running average of the samples generated for the
%   missing entries.
%
%   Author: Wangzhi Dai
%   Last Update: 08-20-2018

if nargin<3
    maxIter=1e6;
end
if nargin<4
    tolerance=eps;
end
% if nargin<5
%     verbose=false;
% end

%Get index of the missing entries
missing_idx=isnan(v);
num_missing=sum(missing_idx);
%Pre-allocate matrices for samples and running averages
v_sample=zeros(maxIter,num_missing);
running_avg=zeros(maxIter,num_missing);

%Initialize the first sample
v_sample=round(rand(1,num_missing));
running_avg=v_sample;
sample_full=v;
sample_full(missing_idx)=v_sample;
old_E=freeEnergy(model,sample_full);
%Main Loop
for iter=2:maxIter
    %Generate a new sample according to the previous one
    new_sample=v_sample;
    %Randomly flip one entry among the missing entries
    k=randi([1,num_missing],1);
    new_sample(k)=1-new_sample(k);
    %If the energy of the new sample is less than the previous one, keep
    %this new sample
    sample_full(missing_idx)=new_sample;
    new_E=freeEnergy(model,sample_full);
    if new_E<=old_E
        v_sample=new_sample;
    else
       %If the energy of the new sample is greater than the previous one, then
       %keep this new sample or keep the previous sample according to a
       %random criterion x
       x=rand;
       if exp(old_E-new_E)>x
           v_sample=new_sample;
       end
    end
    %Compute the running average
    running_avg_old=running_avg;
    running_avg=(running_avg_old*(iter-1)+v_sample)/iter;
    %Check Convergence
    if (all(abs(running_avg-running_avg_old)<tolerance)) && (iter>100)
        break
    end
    old_E=new_E;
end

% if verbose
%     v_expect=rbm_missing_expect(v,model);
%     figure;hold;
%     for ii=1:sum(missing_idx)
%         plot(1:iter,running_avg(:,ii)-v_expect(ii),'-','LineWidth',1.5,'MarkerSize',4);
%     end
%     disp(running_avg(end,:));
% end
end
