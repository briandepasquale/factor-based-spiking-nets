function demo_factor_approx_v2_4(task,save_pth)

%NOTE THAT SPIKING NETWORK DOES NOT GET BIAS TERM NOW FOR CI TASK. DOES
%THIS MATTER? 
%ANSWER: NO.

%DEMO WILL RUN THE FOLLOWING SUBROUTINES, OR SOME CAN BE SHUT OFF.
%1. If no factors are provided, PCs of the firing rate dyanmics are computed, 
    %so that only the most dominate signals are learned in the factor-approximating network.
%2. The mean input into each neuron is computed and subtracted, to
    %control for aphysiological firing rates.
%3. RLS is used to train the inputs into each neuron.
%4. RLS performance is tested.
%5. RLS data and shuffled data is gathered to do batch least square with
    %constraints, if desired. 
%6. BLS is solved.
%7. The BLS solution is tested. Median error across trials is reported.

%VARIOUS GLOBAL PARAMETERS:
%N: number of neurons in the factor approximating network
%Ntilde: number of neurons in the factor generating population when using a
    %neural network
%n: number of factors
%m: number of outputs
%m_in: number of external ninputs
%fracPCA: what fraction of the total variance should the factors retain
%DTRLS: timesteps between RLS steps
%u_tilde_f: input weights for output signal when using an autoencoding
    %model
%u_tilde_fin: input weights for the external input
%Jtilde: recurrent connections for target generating network, set to I when
    %the factors come from data
%u: orthogonal matrix of norm g that maps from space of Ntilde to N
%Jmu: mean recurrent matrix in factor approximating network for spiking
    %neurons
%Jf: random f connections for facdtor approximating network for spiking
    %neurons
%Js: random s connections for facdtor approximating network for spiking
    %neurons
%etautilde: timescale for continuous time factor generating network
%etau: timescale for firing rate, factor approximating network
%etaus: timescale of s synaptic currents in spiking model
%etauf: timescale of f synaptic currents in spiking model
%etauV: timescale of voltage in spiking model
%Vth: threshold in spiking model
%Vr: reset in spiking model
%Jsp_J: sparsity mask for J when solvig BLS
%JEI: EI mask for J and J0 when solving for BLS
%p_sparse: desired sparsity of J and J0
%v0_tilde: initial state of factor generating network when using a backprop
    %trained RNN
%b_tilde: bias of factor generating network when using a backprop trained
    %RNN
%DT: discrete time steps for backprop trained RNN
%F: nonlinearity for rate models

global N Ntilde n m m_in ...
    dt Tinit ...
    fracPCA DTRLS ...
    u_tilde_f u_tilde_fin Jtilde u Jmu Jf Js ...
    etautilde etau etauf etaus etauV Vth Vr ...
    Jsp_J JEI p_sparse ...
    v0_tilde b_tilde DT ...
    F

if nargin < 2 || isempty(save_pth); save_pth = '~'; end

%% How long to run each routine and which ones to run

%init: number of trials for dynamics to settle from initial condition
%PCA: how many trials to compute PCs
%demean: how many trials to compute mean
%RLS: how many trials to do RLS training for
%test: how many trials to test for, after training
%data: how much data to collect
%LS: number of trials to use for BLS

Tinit = 50;
Ttest = 100;

do_PCA = true; TPCA = 300;
do_demean = true; Tdemean = 300;
%do_RLS = true; TRLS = 500;
do_RLS = true; TRLS = 300;
do_data = true; Tdata = 100;
do_batch = false; do_shuff = false; TLS = 100;

%% tasks, and parameters to specifiy for certain tasks

%task: Pick the task
%out: dimension of output
%in: dimension of input

switch task   
    case 'osc4'        
        m = 1; %number of output EMG-PCs
        m_in = 1; %number of inputs
        if strcmp(model,'rate'); g = 1; end
    
    case 'oscfactor'       
        have_factors = true;
        m = 3; %number of output EMG-PCs
        m_in = 1; %number of inputs
        n = 12; %number of factors
        g = 2.75; %good gain for N = 800
        if strcmp(model,'rate'); g = 0.1; end; %rate model
        
    case 'oscfactor_start_stop'
        have_factors = true;
        m = 3; %number of output EMG-PCs
        m_in = 1; %number of inputs
        n = 12; %number of factors
        g = 3.5; %good gain for N = 800
        
    case 'xor'       
        m = 1;
        m_in = 2;
        N = 1200;
        Ntilde = 800;
        gtilde = 1.5;
        g = 5;
        fracPCA = 0.999;
        model = 'spikes';
        have_factors = false;
        have_RNN = false;
        
    case 'reaching'  
        m = 3;
        %m_in = 2;
        m_in = 3;
        N = 1200;
        Ntilde = 800;
        Tdata = 300;
        g = 3;
        gtilde = 1.4;
        %fracPCA = 0.9990;
        fracPCA = 0.9900;
        model = 'spikes';
        have_factors = false;
        have_RNN = false;
        
    case 'CI'
        m = 1;
        m_in = 4;
        N = 800;
        g = 3;
        fracPCA = 0.95;
        model = 'spikes';
        have_factors = false;
        have_RNN = true;
              
        load('~/Dropbox/factors_networks/demo/data/contextual_integrator/params.mat');
        Ntilde = numel(trained_params.predict) - 1;
        blah = fieldnames(trained_params);
        v0_tilde = trained_params.(blah{2})';
        u_tilde_fin = trained_params.change(1:m_in,:)';
        Jtilde = trained_params.change(m_in+1:m_in+100,:);
        b_tilde = trained_params.change(end,:);
                
end

%% Gain parameters

%g: learned feedback gain of spiking net
%gtilde: gain of recurrent connectivity
%muf: global inhibition
%gs: gain of slow random synapses
%gf: gain of fast random synpases

%J0 mean and variance when doing RLS
switch model
    case 'spikes'
        muf = -0.3; gs = 0.11; gf = 0.13;
    case 'rate'
        gs = 0; gf = 0; muf = 0;
end

%BLS parameters and J0 mean and variance when doing BLS
if do_batch      
    %EI: ratio of E to I
    %p_sparse: sparsity of solution after doing BLS
    p_EI = 0.5;
    p_sparse = 0.4;
    %for sparse, with J0 exhibiting E/I separation
    muf = -0.3 * (1/p_EI) * (1/p_sparse); %global inhibition
    gs = 0.11 * 1/sqrt(p_sparse); %slow variance
    gf = 0.13 * 1/sqrt(p_sparse); %fast variance
end

%% Parameter related to time

%DTRLS: number of time steps between RLS updates
%dt: timestep
%tautilde: decay time for factor generating network
%tauf: decay for fast synapses
%taus: decay for slow synapses
%tauV: decay for voltage in spiking net
%Vth: spike threshold
%Vr: reset voltage

DTRLS = 2; %DTRLS = 20; make it longer than tau, then use RLS data for BLS for rate networks? 
dt = 1e-3;
tautilde = 1e-2;
tau = 1e-2;
taus = 1e-1;
tauf = 5e-3;
tauV = 1e-2;
Vth = 1e-16;
Vr = -10;
DT = tau;

%precompute for doing Euler integration
etautilde = exp(-dt/tautilde);
etau = exp(-dt/tau);
etaus = exp(-dt/taus);
etauf = exp(-dt/tauf);
etauV = exp(-dt/tauV);

%% Random connections

%Jtilde: recurrent firing rate connections
%ufout: input connections into rate network for fout
%ufin: input connecdtions into rate network for fin
%Jsp_J0: %parse mask for random connections, this won't be the same as for the learned connections
%JEI: EI mask
%u: connections from Ntilde to N
%Jmu: recurrent inhibition
%Jf: fast random synapses
%Js: slow random synapses
%s_train: which synapses will be trained in the spiking network
%Jsp_J: sparse mask for the learned connections

%randseed = 3; %CI,reaching, xor
%randseed = 4; %xor
randseed = 5; %xor
rng(randseed); %set ran seed, for ran matrcies

%fracPCA: used to compute number of PCs to use
if ~exist('fracPCA'); fracPCA = 0.999; end

%for generating targets from an autoencoding network
if ~have_factors && ~have_RNN
    Jtilde = gtilde * 1/sqrt(Ntilde) * randn(Ntilde);
   % %if strcmp(task,'reaching')
   %     u_tilde_f = -cos(linspace(0,2*pi,Ntilde));
   %     u_tilde_f = cat(1, u_tilde_f, -sin(linspace(0,2*pi,Ntilde)));
   %     u_tilde_f = u_tilde_f';
   % else
        u_tilde_f  =  (-1 + 2 * rand(Ntilde,m));    
   % end
end

%input vector for fin
if ~have_RNN
    %if strcmp(task,'reaching')
    %    u_tilde_fin = sign(randn(Ntilde,1))';
    %    u_tilde_fin = cat(1,u_tilde_fin,cos(linspace(0,2*pi,Ntilde)));
    %    u_tilde_fin = cat(1,u_tilde_fin,sin(linspace(0,2*pi,Ntilde)));
    %    u_tilde_fin = u_tilde_fin';
    %else
        u_tilde_fin = (-1 + 2 * rand(Ntilde,m_in));
    %end
end

% Ntilde to N matrix
if N == Ntilde && strcmp(model,'rate')
    u = g * eye(N);
else
    u = g * (orth((-1 + 2 * rand(N,Ntilde))));
end

%J0 for spiking models
if strcmp(model,'spikes')
    Jmu = muf * (1/tauf) * 1/(N) * ones(N);
    Jf = gf * (1/tauf) * 1/sqrt(N) * randn(N);
    Js = gs * (1/taus) * 1/sqrt(N) * randn(N);
end

%sparsity and J0 for BLS
if do_batch
   
    %pattern of EI cells in J0 and J
    JEI = ones(N,1) * [ones(1,round(p_EI*N)), -1*ones(1,round((1-p_EI)*N))];
    
    %sparsity of J
    Jsp_J = zeros(N);
    for i = 1:N
        Jsp_J(i,randsample(N, 2*p_sparse*N)) = 1;
        %Jsp_J(i,randsample(1:p_EI*N, 2*p_sparse*N*p_EI)) = 1;
        %Jsp_J(i,randsample(p_EI*N+1:N, 2*p_sparse*N*(1-p_EI))) = 1;
    end
    
    %J0
    if strcmp(model,'spikes')
        Jsp_J0 = zeros(N);
        for i = 1:N
            Jsp_J0(i,randsample(N, p_sparse * N)) = 1;
            %Jsp_J0(i,randsample(1:p_EI*N, p_sparse*N*p_EI)) = 1;
            %Jsp_J0(i,randsample(p_EI*N+1:N, p_sparse*N*(1-p_EI))) = 1;
        end
        Jmu = muf * (1/tauf) * 1/(N) * ((ones(N,1) * [zeros(1,round(p_EI*N)), ones(1,round((1-p_EI)*N))]) .* Jsp_J0);
        Jf = gf * (1/tauf) * 1/sqrt(N) * (abs(randn(N)) .* JEI .* Jsp_J0);
        Js = gs * (1/taus) * 1/sqrt(N) * (abs(randn(N)) .* JEI .* Jsp_J0);
    end
    
end

F = @(x)tanh(x); %nonlinearity for rate neurons

%save all of the network parameters
save(fullfile(save_pth,sprintf('fac_approx_%s_%s.mat',model,task)),'-v7.3');

%% Compute PCs of rate model

if do_PCA && ~have_factors
    [wy_tilde,n,COV_vtilde] = run_fac_approx_model_v2_4('PCA',model,have_factors,have_RNN,TPCA,task);
    %[wy_tilde,d] = eig(COV_vtilde);
    %wy_tilde = fliplr(wy_tilde);
    %d = flipud(diag(d));
    %fracPCA = 0.95;
    %n = find(cumsum(d)/sum(d) > fracPCA, 1, 'first');
    %wy_tilde = wy_tilde(:,1:n);                 
else
    wy_tilde = orth((-1 + 2 * rand(Ntilde,n)));
end

%% demean spiking inputs

if do_demean
    VmuJ = run_fac_approx_model_v2_4('demean',model,have_factors,have_RNN,Tdemean,task,wy_tilde);
else
    VmuJ = zeros(N,1);
end

%% Train with RLS

if do_RLS
    
    [wy,w] = run_fac_approx_model_v2_4('RLStrain',model,have_factors,have_RNN,TRLS,task,wy_tilde,VmuJ);
    save(fullfile(save_pth,sprintf('fac_approx_%s_%s.mat',model,task)),'-append');
    % Test RLS solution
    ERR_RLS = run_fac_approx_model_v2_4('RLStest',model,have_factors,have_RNN,Ttest,task,wy_tilde,VmuJ,wy,w);
    if do_data
        %for RLS data
        data_RLS = run_fac_approx_model_v2_4('data',model,have_factors,have_RNN,Tdata,task,wy_tilde,VmuJ,wy,w,'RLS');
    end    
else        
    wy = zeros(size(wy_tilde,2),2*N); %learned spike net matrix
    w = zeros(m,2*N); %output matrix      
end

save(fullfile(save_pth,sprintf('fac_approx_%s_%s.mat',model,task)),'-append');

%%

%load('/Users/briandepasquale/Projects/inProgress/factors_manifesto/data/current_data/post_hoc/fac_approx_spikes_oscfactor_2_060119.mat');

%% Compute PC distribution of currents (this only works for the cycling task

uy = u * eye(Ntilde) * wy_tilde;

tar = zeros(length(data_RLS(1).y(:,1)), length(data_RLS(1).y(1,:)), length(data_RLS));
eta = zeros(length(data_RLS(1).VJ0s(:,1)), length(data_RLS(1).VJ0s(1,:)), length(data_RLS));
facs = data_RLS(1).y_tilde;
fins = data_RLS(1).fin;

for i = 1:length(data_RLS)
    tar(:,:,i) = data_RLS(i).y;
    eta(:,:,i) = data_RLS(i).VJ0s + data_RLS(i).VJ0f;
end

tar = reshape(uy * reshape(tar,size(tar,1),[]), size(uy,1), size(tar,2), size(tar,3));

%noise_input = bsxfun(@minus, eta, mean(eta,3)); %substract mean from noise input

%tot_mean_input = bsxfun(@plus, tmp5, mean(eta,3)); %add the across trial mean from the noise input (since this is actually signal)

%remove across trial and time mean
%tot_mean_input = bsxfun(@minus, tot_mean_input, mean(mean(tot_mean_input,3),2));
%noise_input = bsxfun(@minus, noise_input, mean(mean(noise_input,3),2));

%%

tot_input = reshape(tar,N,[]) + reshape(eta, N, []);

%% Filter by tauV

etauV = exp(-dt/tauV);
tot_input_V = tot_input;

for i = 2:length(tot_input)
    tot_input_V(:,i) = tot_input(:,i) + (tot_input_V(:,i-1) - tot_input(:,i)) * etauV;
end

%%

%[V_tot_input, d_tot_input] = eig(tot_input_V * tot_input_V');
[V_tot_input, d_tot_input] = eig(tot_input * tot_input');
V_tot_input = fliplr(V_tot_input);
d_tot_input_unnorm = flip(diag(d_tot_input));
d_tot_input = flip(diag(d_tot_input))/sum(diag(d_tot_input));

%% single trial PCA

remaining = zeros(length(d_tot_input_unnorm)-1,1);
upto = zeros(length(d_tot_input_unnorm)-1,1);

for i = 1:length(d_tot_input_unnorm)-1
    remaining(i) = sum(d_tot_input_unnorm(i+1:end));
    upto(i) = sum(d_tot_input_unnorm(1:i));
end

%%

foo = permute(reshape(V_tot_input' * tot_input, N, size(tar,2), size(tar,3)),[2,1,3]);
%foo = (V_tot_input(:,1:4)' * tot_input)';

%foo = (V_tot_input' * tot_input)';

foo_fft = fft(foo);

%foo2 = reshape((V_tot_input(:,101:end)' * tot_input)',1,[])';
%foo2_fft = fft(foo2);

L = size(foo,1);
P2 = abs(foo_fft/L);
P1 = P2(1:L/2+1,:,:);
P1(2:end-1,:,:) = 2*P1(2:end-1,:,:);
%P1 = P2(1:L/2+1,:);
%P1(2:end-1,:,:) = 2*P1(2:end-1,:);
%P1 = bsxfun(@rdivide,P1,sum(P1,1));
Fs = 1/1e-3;
f = Fs*(0:(L/2))/L;

%L2 = size(foo2,1);
%P22 = abs(foo2_fft/L2);
%P12 = P22(1:L2/2+1,:);
%P12(2:end-1,:,:) = 2*P12(2:end-1,:);
%f2 = Fs*(0:(L2/2))/L2;

%% Error between mean input current and factors

[~,m_tar_eta] = run_fac_approx_model_v2_4('RLStest_weakPCs',model,have_factors,have_RNN,1000,task,wy_tilde,VmuJ,wy,w,eye(N));
m_tar_eta = bsxfun(@minus,m_tar_eta,mean(m_tar_eta,2));

%%

blah = ([facs;fins;ones(1,length(fins))]' * ([facs;fins;ones(1,length(fins))]'\m_tar_eta'))';
mERR = (m_tar_eta - ([facs;fins;ones(1,length(fins))]' * ([facs;fins;ones(1,length(fins))]'\m_tar_eta'))');
frac_var_exp = sum(bsxfun(@minus,blah,mean(blah,2)).^2)/sum(bsxfun(@minus,m_tar_eta,mean(m_tar_eta,2)).^2);
mERR = diag(mERR * mERR') ./ diag(m_tar_eta * m_tar_eta');

%%

[mERR_sort, mERR_idx] = sort(mERR);

%% Filter out weak PCs from spiking network

%fracs = [0.0, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5];
fracs = 0:0.025:0.5;
ERR_RLS_strongPCs = zeros(length(fracs),1);
removed_var = zeros(length(fracs),1);

for i = 1:length(fracs)
    removed_PCs = randsample(101:800,round(N*fracs(i)));
    removed_var(i) = sum(d_tot_input(removed_PCs))/sum(d_tot_input);
    w_strongPCs = V_tot_input(:,[1:100,setdiff(101:800,removed_PCs)]);
    w_strongPCs_filt = w_strongPCs * w_strongPCs';
    ERR_RLS_strongPCs(i) = median(run_fac_approx_model_v2_4('RLStest_weakPCs',model,have_factors,have_RNN,Ttest,task,wy_tilde,VmuJ,wy,w, w_strongPCs_filt));
end

%%

myfig2 = figure('Units','inches','Position',[0 0 8 4]);
myfig2.PaperUnits = 'inches';
myfig2.PaperPosition = [0 0 8 4];
set(gcf,'color','w');

trial = 302;
blue = [102,160,255]/255;

subplot(2,4,1);
plot(blah(mERR_idx(trial),:), 'LineWidth', 2, 'color', blue); hold on;
plot(m_tar_eta(mERR_idx(trial),:),'k:', 'LineWidth', 1)
ylabel('current (mA)');
xlabel('time (ms)');
set(gca,'box','off', 'ylim', [-8 8], 'fontsize', 8);

%plot(cumsum(d_tot_input), 'LineWidth', 2); xlabel('PC#'); ylabel('cum. sum of variance'); title('single trial PCA on input currents');
subplot(2,4,5);
hold on;
%plot(upto ./ remaining, 'LineWidth', 2,'Marker','.', 'LineStyle','none', 'MarkerSize', 14, ...
%    'color',[0.5,0.5,0.5]); 

%remaining2 = zeros(length(d_tot_input)-1,1);
%upto2 = zeros(length(d_tot_input)-1,1);

%for i = 1:length(d_tot_input)-1
%    remaining2(i) = sum(d_tot_input(i+1:end));
%    upto2(i) = d_tot_input(i);
%end

%plot(upto2 ./ remaining2, 'LineWidth', 2,'Marker','.', 'LineStyle','none', 'MarkerSize', 14, ...
%    'color',[0.5,0.5,0.5]);

plot(1 - cumsum(d_tot_input), 'LineWidth', 2,'Marker','.', 'LineStyle','none', 'MarkerSize', 14, ...
    'color',[0.5,0.5,0.5]); 
xlabel('PC#'); 
%ylabel('var_{i}/var_{i+1:N}'); 
ylabel('remaining variance');
%title('single trial PCA on input currents');
%plot(d_tot_input_unnorm/sum(d_tot_input_unnorm), 'LineWidth', 2, 'Marker','x', 'LineStyle','none'); 
%xlabel('PC#'); ylabel('cum. sum of variance'); 
%title('single trial PCA on input currents');
set(gca,'xlim',[0 30],'box','off', 'fontsize', 8);
set(gca,'ylim',[0 1],'box','off', 'fontsize', 8);

%print(gcf,'~/Projects/inProgress/factors_manifesto/figures/matlab_exports/mark_analysis_1.pdf','-dpdf');
%saveas(gcf,'~/Projects/inProgress/factors_manifesto/figures/matlab_exports/mark_analysis_1.png');

plt1 = reshape(V_tot_input(:,1)' * tot_input, size(tar,2), size(tar,3));
plt2 = reshape(V_tot_input(:,100)' * tot_input, size(tar,2), size(tar,3));

%plt1 = reshape(V_tot_input(:,1)' * tot_input_V, size(tar,2), size(tar,3));
%plt2 = reshape(V_tot_input(:,100)' * tot_input_V, size(tar,2), size(tar,3));

plum = [221,160,221]/255;

subplot(2,4,2); plot(plt1(:,1), 'color', blue, 'LineWidth',2); hold on;
p1 = plot(plt1(:,2),'color', plum, 'LineWidth',2);
p1.Color(4) = 0.5;
%title('PC1'); 
xlabel('time (ms)');ylabel('PC (arb)');
set(gca,'box','off','ylim', [-140 140], 'fontsize', 8);

orange = [255,165,0]/255;
tomato = [255,99,71]/255;

subplot(2,4,6); hold on; plot(plt2(:,1), 'color', orange, 'LineWidth',1);
p2 = plot(plt2(:,2), 'color', tomato, 'LineWidth',1); 
p2.Color(4) = 0.75;
%title('PC100'); 
xlabel('time (ms)');ylabel('PC (arb)');
set(gca,'box','off','ylim', [-30 30], 'fontsize', 8);

subplot(2,4,4); plot(fracs*N, ERR_RLS_strongPCs, 'LineWidth', 2, 'Marker','.', 'LineStyle','none', 'MarkerSize', 14,...
    'color',[0.5,0.5,0.5]); 
xlabel('# of removed PCs');
ylabel('nMSE'); 
set(gca,'ylim', [-0.5 3.5] ,'xlim', [-20 400], 'box','off', 'fontsize', 8); 

%title('nMSE when removing PCs > 100');

subplot(2,4,8); plot(fracs*N, removed_var, 'LineWidth', 2, 'Marker','.', 'LineStyle','none', 'MarkerSize', 14,...
    'color',[0.5,0.5,0.5]); 
xlabel('# of removed PCs');ylabel('frac. var.');
set(gca,'xlim', [-20 400], 'box', 'off', 'fontsize', 8); 
%title('frac of variance caputred by PCs > 100');
    
%print(gcf,'~/Projects/inProgress/factors_manifesto/figures/matlab_exports/mark_analysis_3.pdf','-dpdf');
%saveas(gcf,'~/Projects/inProgress/factors_manifesto/figures/matlab_exports/mark_analysis_3.png');

%subplot(3,2,6); plot(mean(std(reshape(V_tot_input' * tot_input, N, size(tar,2), size(tar,3)),0,3),2))

%figure; plot(f,mean(P1,3));
subplot(2,4,3);
plot(f,sum(mean(P1(:,1:12,:),3),2),'Marker','.','LineStyle','-', 'color', blue, 'MarkerSize', 14);
ylabel('|P(f)|'); xlabel('freq (Hz)');
set(gca,'xlim',[0 50], 'box', 'off', 'ylim', [0 600], 'fontsize', 8); 
subplot(2,4,7);
plot(f,sum(mean(P1(:,13:end,:),3),2),'Marker','.','LineStyle','-', 'color', tomato, 'MarkerSize', 14)
xlabel('freq (Hz)'); ylabel('|P(f)|');
set(gca,'xlim',[0 50], 'ylim', [0 600], 'box', 'off', 'fontsize', 8);
%figure;plot(f2,P12);

print(gcf,'~/Projects/inProgress/factors_manifesto/figures/matlab_exports/post_hoc_supp_mat1_v2.pdf','-dpdf');

%% BLS

if do_batch
    
    %sample RLS data and train LS solution
    [noise_RLS,ys_LS_RLS,vLSs_RLS] = run_fac_approx_model_v2_4('LSdata',model,have_factors,have_RNN,TLS,task,wy_tilde,VmuJ,wy,w,'RLS');
    %do BLS training
    J_LS_RLS = do_BLS_v2(ys_LS_RLS,vLSs_RLS,model,wy_tilde);
    % test RLS noise BLS solution
    ERR_LS_RLS = run_fac_approx_model_v2_4('LStest',model,have_factors,have_RNN,Ttest,task,wy_tilde,VmuJ,wy,w,J_LS_RLS);
    if do_data
        %for BLS data from RLS noise solution
        data_BLS_RLS = run_fac_approx_model_v2_4('data',model,have_factors,have_RNN,Tdata,task,wy_tilde,VmuJ,wy,w,'BLS',J_LS_RLS);      
    end
    save(fullfile(save_pth,sprintf('fac_approx_%s_%s.mat',model,task)),'-append');
    
    if do_shuff
        %shuffle the noise from the RLS noise collected to compute the BLS solution
        noise_shuff = reshape(noise_RLS(randperm(numel(noise_RLS))),...
            size(noise_RLS,1),size(noise_RLS,2));
        %run with the shuffled noise
        [~,ys_LS_shuff,vLSs_shuff] = run_fac_approx_model_v2_4('LSdata',model,have_factors,have_RNN,TLS,task,wy_tilde,VmuJ,wy,w,'shuff',noise_shuff);
        %train
        J_LS_shuff = do_BLS_v2(ys_LS_shuff,vLSs_shuff,model,wy_tilde);
        % test shuffled BLS solution
        ERR_LS_shuff = run_fac_approx_model_v2_4('LStest',model,have_factors,have_RNN,Ttest,task,wy_tilde,VmuJ,wy,w,J_LS_shuff);
        save(fullfile(save_pth,sprintf('fac_approx_%s_%s.mat',model,task)),'-append');
        if do_data
            %for BLS data from shuffled noise solution
            data_BLS_shuff = run_fac_approx_model_v2_4('data',model,have_factors,have_RNN,Tdata,task,wy_tilde,VmuJ,wy,w,'BLS',J_LS_shuff);
        end
    end
    
end

%% save results

save(fullfile(save_pth,sprintf('fac_approx_%s_%s.mat',model,task)),'-append');
