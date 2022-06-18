clearvars;

global N Ntilde P m m_in g muf gs gf u utilde Jtilde
global utilde_in u_in dt

N = 1200; %number of neurons in the spiking network
Ntilde = 800; %number of neurons in the rate network
dt = 1e-3; %integration timestep

load('../reaching/reaching_data.mat','emg');

m_in = 3; %number of inputs
m = 3; %number of outputs

%utilde_in: input weights for the external input

rng(1); %set ran seed, for ran matrcies
utilde_in = -1 + 2 * rand(Ntilde,m_in); 

%% Compute factors from the rate model

gtilde = 1.4;
Jtilde = gtilde * 1/sqrt(Ntilde) * randn(Ntilde);
utilde = -1 + 2 * rand(Ntilde,m);   
Jtildeprime = [Jtilde, utilde];

TPCA = 300;
XX = zeros(Ntilde+m);

for i = 1:TPCA
    if i == 1
        [xprimes,x] = rate_model(emg);
    else
        [xprimes,x] = rate_model(emg,x);
    end
    XX = XX + xprimes * xprimes';
    clc; fprintf('PCA, %g trials of %g \n', i, TPCA);
end

[V,d] = eig(XX);
V = fliplr(V);
d = flipud(diag(d)); 

fracPCA = 0.99;  
P = find(cumsum(d)/sum(d) > fracPCA, 1, 'first');
V = V(:,1:P);

%% Load cycling factors

cycling_emg = load('../cycling/cycling_data.mat','emg', 'factors');
factors = cycling_emg.factors;

m_in_cycling = 1; %number of inputs
P = size(factors,2); %number of factors

%utilde_in: input weights for the external input
%V: mapping from factors to N space. In this case, it's just an orthnormal mapping

rng(3); %set ran seed, for ran matrcies
utilde_in_cycling = -1 + 2 * rand(Ntilde, m_in_cycling);
V_cycling = orth((-1 + 2 * rand(Ntilde, P)));

%%

m_in = m_in + m_in_cycling; %number of inputs
emg = {emg, cycling_emg.emg};
factors_V = {V, factors};

%% Parameters for spiking model, and some matrices to the spiking model

g = 6; %feedback gain in spiking network
muf = -0.3; %recurrent inhibition
gs = 0.11; %gain of slow random synapses
gf = 0.13; %gain of fast random synpases

%U: orthogonal matrix of norm g that maps from space of Ntilde to N
%u_in: input weights for the external input to the spiking model
%u: mapping from P-space to N-space

U = g * orth((-1 + 2 * rand(N, Ntilde)));
u_in = U * [utilde_in, utilde_in_cycling]; 
u = U * [Jtildeprime * V, V_cycling];   

%% demean spiking inputs

Tdemean = 300; %number of trials for computing the mean of the recurrent input
vJsbar = spiking_network('reaching-cycling','demean',Tdemean,factors_V,emg);

%% Train with RLS

TRLS = 300; %number of trials for doing RLS
[w,W] = spiking_network('reaching-cycling','train',TRLS,factors_V,emg,vJsbar);

%% Test

Ttest = 100; %number of trials for testing
ERR_RLS = spiking_network('reaching-cycling','test',Ttest,factors_V,emg,vJsbar,w,W);  