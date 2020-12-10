clearvars;

global N Ntilde P m m_in g muf gs gf

N = 800; %number of neurons in the spiking network
Ntilde = 800;

load('/cycling_data.mat','emg', 'factors');

m_in = 1; %number of inputs
m = size(emg,2); %number of outputs
P = size(factors,2); %number of factors
g = 4; %feedback gain in spiking network
muf = -0.3; %recurrent inhibition
gs = 0.11; %gain of slow random synapses
gf = 0.13; %gain of fast random synpases

%% demean spiking inputs

Tdemean = 300; %number of trials for computing the mean of the recurrent input
vJsbar = cycling_network('demean',Tdemean,factors,emg);

%% Train with RLS

TRLS = 300; %number of trials for doing RLS
[w,W] = cycling_network('train',TRLS,factors,emg,vJsbar);

%% Test

Ttest = 100; %number of trials for testing
ERR_RLS = cycling_network('test',Ttest,factors,emg,vJsbar,w,W);  
