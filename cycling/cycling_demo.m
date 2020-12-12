clearvars;

global N Ntilde P m u u_in muf gs gf dt

N = 800; %number of neurons in the spiking network
Ntilde = 800;
dt = 1e-3;

load('/cycling_data.mat','emg', 'factors');

m_in = 1; %number of inputs
m = size(emg,2); %number of outputs
P = size(factors,2); %number of factors
g = 4; %feedback gain in spiking network
muf = -0.3; %recurrent inhibition
gs = 0.11; %gain of slow random synapses
gf = 0.13; %gain of fast random synpases

%utilde_in: input weights for the external input
%U: orthogonal matrix of norm g that maps from space of Ntilde to N

rng(3); %set ran seed, for ran matrcies
utilde_in = (-1 + 2 * rand(Ntilde,m_in));
V = orth((-1 + 2 * rand(Ntilde,P)));
U = g * (orth((-1 + 2 * rand(N,Ntilde))));
u_in = U * utilde_in;  
u = U * V;   

%% demean spiking inputs

Tdemean = 300; %number of trials for computing the mean of the recurrent input
vJsbar = spiking_network('cycling','demean',Tdemean,factors,emg);

%% Train with RLS

TRLS = 300; %number of trials for doing RLS
[w,W] = spiking_network('cycling','train',TRLS,factors,emg,vJsbar);

%% Test

Ttest = 100; %number of trials for testing
ERR_RLS = spiking_network('cycling','test',Ttest,factors,emg,vJsbar,w,W);  
