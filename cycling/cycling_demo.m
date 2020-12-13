clearvars;

global N P m u u_in muf gs gf dt

N = 800; %number of neurons in the spiking network
dt = 1e-3; %integration timestep

load('/cycling_data.mat','emg', 'factors');

m_in = 1; %number of inputs
m = size(emg,2); %number of outputs
P = size(factors,2); %number of factors

%utilde_in: input weights for the external input
%V: mapping from factors to N space. In this case, it's just an orthnormal mapping

rng(3); %set ran seed, for ran matrcies
utilde_in = -1 + 2 * rand(N,m_in);
V = orth((-1 + 2 * rand(N,P)));

%% Parameters for spiking model, and some matrices to the spiking model

g = 4; %feedback gain in spiking network
muf = -0.3; %recurrent inhibition
gs = 0.11; %gain of slow random synapses
gf = 0.13; %gain of fast random synpases

%U: orthogonal matrix of norm g that maps from space of N to N
%u_in: input weights for the external input to the spiking model
%u: mapping from P-space to N-space

U = g * eye(N);
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
