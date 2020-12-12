clearvars;

global N Ntilde P m m_in g muf gs gf u utilde Jtilde
global utilde_in u_in dt

N = 1200; %number of neurons in the spiking network
Ntilde = 800;
dt = 1e-3;

load('/reaching_data.mat','emg');

m_in = 3; %number of inputs
m = 3; %number of outputs
g = 3; %feedback gain in spiking network
muf = -0.3; %recurrent inhibition
gs = 0.11; %gain of slow random synapses
gf = 0.13; %gain of fast random synpases

%utilde_in: input weights for the external input
%U: orthogonal matrix of norm g that maps from space of Ntilde to N
%Jmu: mean recurrent matrix in factor approximating network for spiking
    %neurons
%J0f: random f connections for facdtor approximating network for spiking
    %neurons
%J0s: random s connections for facdtor approximating network for spiking
    %neurons

rng(3); %set ran seed, for ran matrcies
utilde_in = (-1 + 2 * rand(Ntilde,m_in));
U = g * (orth((-1 + 2 * rand(N,Ntilde))));
u_in = U * utilde_in;  

%%

gtilde = 1.4;
Jtilde = gtilde * 1/sqrt(Ntilde) * randn(Ntilde);
utilde = -1 + 2 * rand(Ntilde,m);    

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

fracPCA = 0.99;  
%compute PCs from cov matrix, and arrange them from
%largest to smallest
[V,d] = eig(XX);
V = fliplr(V);
d = flipud(diag(d));                  
%pick the number of dimensions that accounts for fracPCA of variance
P = find(cumsum(d)/sum(d) > fracPCA, 1, 'first');
V = V(:,1:P);
Jtildeprime = [Jtilde, utilde];
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
