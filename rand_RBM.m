% randRBM: get randomized restricted boltzmann machine (RBM) model
%
% rbm = rand_RBM( dimV, dimH, type )
%
%
%Output parameters:
% rbm: the randomized restricted boltzmann machine (RBM) model
%
%
%Input parameters:
% dimV: number of visible (input) nodes
% dimH: number of hidden (output) nodes
% type (optional): (default: 'BBRBM' )
%                 'BBRBM': the Bernoulli-Bernoulli RBM
%                 'GBRBM': the Gaussian-Bernoulli RBM
% range (optional): (default: 1)
%   Range of initial random weights is (-1,1)*range
%


function rbm = rand_RBM( dimV, dimH, type ,range)

if( ~exist('type', 'var') || isempty(type) )
	type = 'BBRBM';
end

if( ~exist('range', 'var') || isempty(range) )
	range = 1;
end

if( strcmpi( 'GB', type(1:2) ) )%For Gaussian RBM only
    rbm.type = 'GBRBM';
    rbm.W = randn(dimV, dimH) * 0.1;
    rbm.hidbias = zeros(1, dimH);
    rbm.visbias = zeros(1, dimV);
    rbm.sig = ones(1, dimV);
else %For Binary RBM
    rbm.type = type;
    rbm.W = (2*rand(dimV, dimH)-1)*range;
    rbm.hidbias = rand(1, dimH)*range;
    rbm.visbias = rand(1, dimV)*range;
end

