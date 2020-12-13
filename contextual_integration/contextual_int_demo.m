clearvars;

global N Ntilde P m m_in g muf gs gf u btilde Jtilde
global utilde_in u_in dt x0

N = 800; %number of neurons in the spiking network
dt = 1e-3; %integration timestep
             
m_in = 4; %number of inputs
m = 1; %number of outputs

load('./backprop_trained_rate_net_params.mat'); 

Ntilde = numel(trained_params.predict) - 1; %number of neurons in the rate network

%x0: input state of the rate model
%utilde_in: input weights for the external input to the rate model
%Jtilde: recurrent weights for the rate model
%btilde: bias weights for the rate model

fields = fieldnames(trained_params);
x0 = trained_params.(fields{2})';
utilde_in = trained_params.change(1:m_in,:)';
Jtilde = trained_params.change(m_in+1:m_in+100,:);
btilde = trained_params.change(end,:);
Jtildeprime = Jtilde';

%% Compute factors from the rate model
 
TPCA = 300;
XX = zeros(Ntilde);

for i = 1:TPCA
    xprimes = backprop_rate_model;
    XX = XX + xprimes * xprimes';
    clc; fprintf('PCA, %g trials of %g \n', i, TPCA);
end

[V,d] = eig(XX);
V = fliplr(V);
d = flipud(diag(d));      

fracPCA = 0.95;  
P = find(cumsum(d)/sum(d) > fracPCA, 1, 'first');
V = V(:,1:P);

%% Parameters for spiking model, and some matrices to the spiking model

g = 3; %feedback gain in spiking network
muf = -0.3; %recurrent inhibition
gs = 0.11; %gain of slow random synapses
gf = 0.13; %gain of fast random synpases

%U: orthogonal matrix of norm g that maps from space of Ntilde to N
%u_in: input weights for the external input to the spiking model
%u: mapping from P-space to N-space

rng(3); %set ran seed, for ran matrcies
U = g * (orth((-1 + 2 * rand(N,Ntilde))));
u_in = U * utilde_in;  
u = U * Jtildeprime * V;   

%% demean spiking inputs

Tdemean = 300; %number of trials for computing the mean of the recurrent input
vJsbar = spiking_network('CI','demean',Tdemean,V);

%% Train with RLS

TRLS = 300; %number of trials for doing RLS
[w,W] = spiking_network('CI','train',TRLS,V,[],vJsbar);

%% Test

Ttest = 100; %number of trials for testing
ERR_RLS = spiking_network('CI','test',Ttest,V,[],vJsbar,w,W);  
