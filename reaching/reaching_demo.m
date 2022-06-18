clearvars;

global N Ntilde P m m_in g muf gs gf u utilde Jtilde
global utilde_in u_in dt

N = 1200; %number of neurons in the spiking network
Ntilde = 800; %number of neurons in the rate network
dt = 1e-3; %integration timestep

load('./reaching_data.mat','emg');

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

%% Parameters for spiking model, and some matrices to the spiking model

g = 3; %feedback gain in spiking network
muf = -0.3; %recurrent inhibition
gs = 0.11; %gain of slow random synapses
gf = 0.13; %gain of fast random synpases

%U: orthogonal matrix of norm g that maps from space of Ntilde to N
%u_in: input weights for the external input to the spiking model
%u: mapping from P-space to N-space

U = g * orth((-1 + 2 * rand(N,Ntilde)));
u_in = U * utilde_in; 
u = U * Jtildeprime * V;   

%% demean spiking inputs

Tdemean = 300; %number of trials for computing the mean of the recurrent input
vJsbar = spiking_network('reaching','demean',Tdemean,V,emg);

%% Train with RLS

TRLS = 300; %number of trials for doing RLS
[w,W] = spiking_network('reaching','train',TRLS,V,emg,vJsbar);

%% Test

Ttest = 100; %number of trials for testing
ERR_RLS = spiking_network('reaching','test',Ttest,V,emg,vJsbar,w,W);  
