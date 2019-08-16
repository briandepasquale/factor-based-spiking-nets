function varargout = run_fac_approx_model_v2_4(mode,model,have_factors,have_RNN,T,task,varargin)

%change factor definition and test on xor and reaching
%define factors w/o bias for CI and recompute

global N Ntilde n m m_in ...
    dt Tinit ...
    fracPCA DTRLS ...
    u_tilde_f u_tilde_fin Jtilde u Jmu Jf Js ...
    etautilde etauf etaus etau etauV Vth Vr ...
    p_sparse ...
    v0_tilde b_tilde DT F

%% Figure, for plotting various things as simulation runs

%figure handle
fh = figure('Color','w', 'menubar', 'none', 'Name', sprintf('task: %s',task),...
    'NumberTitle','off','ToolBar','none');

%axes handles
ah1 = axes(fh,'LineWidth',2,'FontSize',12,'Position',[0.1 0.75 0.8 0.2]);
ah2 = axes(fh,'LineWidth',2,'FontSize',12,'Position',[0.1 0.05 0.8 0.3]);
ah3 = axes(fh,'LineWidth',2,'FontSize',12,'Position',[0.1 0.40 0.8 0.3]);

xlabel(ah2,'time (v)');

%delete any old lines in the axes handles, from previous executions of code
delete(findobj(ah1,'Type','Line')); delete(findobj(ah2,'Type','Line'));

%line handles for input and target output
lh_f = line(ah1,'Color','b','LineWidth',2,'Marker','none','LineStyle',':');
lh_fin = line(ah1,'Color','k','LineWidth',1,'Marker','none','LineStyle',':');

%depending on which step of the algorithm is being executed, different
%things will be plotted

nplt = 3; %number of factors and rates/voltages to plot

switch mode
    
    case 'PCA'
        
        %line handles for factor targets
        for i = 1:nplt
            lh_ytilde(i) = line(ah2,'Color','r','LineWidth',2,'Marker','none','LineStyle','-');
        end
        
    case {'demean','RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}
        
        %line handle for generated output
        lh_z = line(ah1,'Color','r','LineWidth',1,'Marker','none','LineStyle','-');
        
        %line handles for factors, factor targets and rates/voltages
        for i = 1:nplt
            lh_y(i) = line(ah2,'Color','k','LineWidth',2,'Marker','none','LineStyle','-');
            lh_ytilde(i) = line(ah2,'Color','r','LineWidth',2,'Marker','none','LineStyle',':');
            lh_V(i) = line(ah3,'Color','b','LineWidth',2,'Marker','none','LineStyle','-');
        end
        
end

%% Load relevant data, if needed

if strcmp(task,'reaching')
    %load('data/reaching/EMG_v3.mat','EMG');
    load('data/reaching/EMG_v4.mat','EMG');
end

if any(strcmp(task,{'oscfactor'}))
    load('data/cycling/Cousteau','proj');
    FAC = 10 * repmat(proj,4,1);
    fac = FAC;
    load('data/cycling/Cousteau_emg','proj')
    EMG = repmat(proj,4,1);
end

if any(strcmp(task,{'oscfactor_start_stop'}))
    load('data/cycling/Cousteau','proj');
    FAC = 10 * proj;
    load('data/cycling/Cousteau_emg','proj')
    EMG = proj;
end

%% Unpack varargin, depending on which subroutine is being run

if any(strcmp(mode,{'demean','RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}))
    %Principal vectors, maping from Ntilde-space to n-space
    wy_tilde = varargin{1};
    
    if any(strcmp(mode,{'RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}))       
        %mean recurrent input to subtract from each factor approximating
        %neuron
        VmuJ = varargin{2};
        
        if (strcmp(mode,{'RLStrain'}) && numel(varargin) > 2) || ...
                any(strcmp(mode,{'RLStest','LSdata','LStest','data','RLStest_weakPCs'}))
            
            wy = varargin{3}; %learned connection, N-space to n-space
            w = varargin{4}; %learned output connections
            
            if strcmp(mode,'RLStest_weakPCs')
                
                w_strongPCs_filt = varargin{5};
                
            end
            
        end
        
        switch mode
            case 'RLStrain'
                %if values are provided from a previous RLS solution,
                %unpack them
                if numel(varargin) > 2
                    Pv = varargin{5}; %inverse covariance of v
                    Py = varargin{6}; %inverse covariance of y
                    
                end
                
            case 'LSdata'
                LSmode = varargin{5}; %use shuffled RLS noise or RLS noise
                
                if strcmp(LSmode,'shuff')
                    noise = varargin{6}; %if using shuffled noise, unpack the provided shuffled noise
                end
                
            case 'LStest'
                J_BLS = varargin{5}; %learned connections from batch least squares solution
                
            case 'data'
                solution = varargin{5};
                if numel(varargin) > 5 %want to use BLS solution to generate data
                    J_BLS = varargin{6}; %learned connections from batch least squares solution
                end               
        end        
    end    
end

%% Initialize matrices for saving data, depending on which subroutine is being run

if strcmp(mode,'PCA')  
    %for saving covariance of generator network inputs
    if have_RNN %RNN provided
        COV_vtilde = zeros(Ntilde);
    else
        COV_vtilde = zeros(Ntilde+m);
    end
    
elseif strcmp(mode,'demean')
    %for saving computed mean of recurrent input into each factor approx. neuron  
    sum_VmuJ = zeros(N,1);
    VmuJ = zeros(N,1);
    
elseif any(strcmp(mode,{'RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}))    
    
    nMSE = NaN(T,1); %normalized mean square error, saved for each trial
    
    if any(strcmp(mode,{'RLStrain','LSdata'}))     
        %if learned recurrent and output matrices were not provided....
        if strcmp(mode,'RLStrain') && numel(varargin) < 5
                      
            w = zeros(m,n+1); %for computing learned output matrix
            Py = 1 * eye(n+1); %inverse covariance matrix
            
            %for computing learned recurrent connections
            if strcmp(model,'rate')               
                wy = zeros(n,N); %n readout weights, half of J
                Pv = 1 * eye(N); %inverse covariance matrix
                
            else           
                wy = zeros(n,2*N); %n readout weights, half of J
                Pv = 1 * eye(2*N); %inverse covariance matrix
                
            end  
            
        elseif strcmp(mode,'LSdata')
            %compute total number of BLS samples to save, T trials x length of
            %trial x 1/dt steps per trial
            
            switch task
                case 'osc4'
                    [~,~,TTrial] = trial_osc4(dt);
                case 'xor'
                    [~,~,TTrial] = trial_xor(dt);
                case 'reaching'
                    [~,~,TTrial] = trial_reaching(dt,EMG);
                case 'oscfactor'
                    [~,~,TTrial] = trial_oscfactor(dt,FAC,EMG);
                case 'oscfactor_start_stop'
                    [~,~,TTrial] = trial_oscfactor_start_stop(dt,FAC,EMG);
            end            
                        
            Tdata_tot = T * TTrial;
            
            %for saving the ys, for solving BLS
            ys = nan(n,Tdata_tot);
            
            %if LS data collection is in RLS "mode" we want to collect data
            %for solving BLS, but with RLS noise
            if strcmp(LSmode,'RLS')                
                %for saving the RLS noise
                noise = nan(n,Tdata_tot);
            end
            
            %saved presynaptic currents/rates, for solving BLS
            if strcmp(model,'rate')
                vs_s = nan(N,Tdata_tot);
            else
                vs_s = nan(2*N,Tdata_tot);
            end
            
        end
        
    elseif strcmp(mode,'LStest') || (strcmp(mode,'data') && strcmp(solution,'BLS'))        
        %if testing BLS solution, or gathering data when using the BLS solution
        %remove weakest synapses so that sparsity is exactly p_sparse
        for i = 1:N
            [~,idx] = sort(abs(J_BLS(i,:)),'descend');
            J_BLS(i,idx(round(p_sparse*numel(J_BLS(i,:)))+1:end)) = 0;
        end
        
    elseif strcmp(mode,'data')       
        switch model   
            %how much of each variable to collect
            case 'spikes'       
                ndata = struct('f',m,'fin',m_in,'y_tilde',n,'y',n,...
                    'V',min([0,N]),'S',N,'v',0,'z0',0);
                
            case 'rate'               
                ndata = struct('f',m,'fin',m_in,'y_tilde',n,...
                    'V',min([200,N]),'S',0,'v',0,'z0',0,'y',n);
                
        end
        
    elseif strcmp(mode,'RLStest_weakPCs')
              
        [~,~,TTrial] = trial_oscfactor(dt,FAC,EMG);
        Vinfs = zeros(N,TTrial);
                        
    end   
end

%% Initalize state variables

%initialize random seed, to set network into same initial state
%go to different initial state it testing or gathering data
rng(3);

%compute what the effective recurrent connectivity is for computing uy
if have_factors
    Jtilde_prime = eye(Ntilde);
elseif have_RNN
    Jtilde_prime = Jtilde';
else
    Jtilde_prime = [Jtilde,u_tilde_f];
end

%if performing CI task, initial network state is determined, otherwise set
%randomly
if ~strcmp(task,'CI')
    v_tilde = 1e0 * randn(Ntilde,1); %teaching network state
end

if any(strcmp(mode,{'demean','RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}))
    
    %product of n to Ntilde, recurrent connectivity of factor generating network (if applicaabl)
    %and Ntilde to N. For going directory from n to N
    uy = u * Jtilde_prime * wy_tilde;
    %uy = u * wy_tilde;
    
    uy_y0 = zeros(N,1); uy_y = uy_y0; %learned input current
    vs0 = zeros(N,1); vs = vs0; %slow presynaptic current/firing rate when using RLS
    
    if strcmp(model,'spikes')
        V0 = 1e-3 * randn(N,1); V = V0; %spike net state
        VJ0s0 = zeros(N,1); VJ0s = VJ0s0; %J0 slow input current
        VJ0f0 = zeros(N,1); VJ0f = VJ0f0; %J0 fast input current
        Vmu0 = zeros(N,1); Vmu = Vmu0; %J0 mean current
        vf0 = zeros(N,1); vf = vf0; %fast presynaptic current when using RLS
    end
    
    if strcmp(task,'CI')     
        %uy_y = uy * wy_tilde' * (Jtilde' * v0_tilde + b_tilde');
        uy_y0 = uy * wy_tilde' * v0_tilde;
        
        %1 second of burn in for the spiking network
        for i = 1:1000
            switch model
                case 'spikes'
                    Vinf = Vmu - VmuJ + uy_y0 + VJ0s + VJ0f;
                    V = Vinf + (V - Vinf) * etauV;
                    S = V >= Vth;
                    V(S) = Vth + Vr;
                    VJ0s = VJ0s * etaus + sum(Js(:,S),2);
                    VJ0f = VJ0f * etauf + sum(Jf(:,S),2);
                    Vmu = Vmu * etauf + sum(Jmu(:,S),2);
                    if any(strcmp(mode,{'RLStrain','RLStest','data','demean','RLStest_weakPCs'}))
                        vs = etaus * vs + S;
                        vf = etauf * vf + S;
                    end
                    
                case 'rate'
                    vinf = F(uy_y);
                    vs = Vinf + (vs - vinf) * etau;
                    
            end
        end
        
        vs0 = vs; %slow presynaptic current/firing rate when using RLS     
        if strcmp(model,'spikes')
            V0 = V; %spike net state
            VJ0s0 = VJ0s; %J0 slow input current
            VJ0f0 = VJ0f; %J0 fast input current
            Vmu0 = Vmu; %J0 mean current
            vf0 = vf;
        end
    end
end

%set various counters
ttrial = inf; %time in current trial
TTrial = 0; %total time in current trial
t_trial_num = 0; %trial number
ttime = 1; %time across all trials
go = true; %flag to quit loop

if any(strcmp(mode,{'RLStest','data','RLStest_weakPCs'}))
    rng(4);
end

%% Run simulation

while go %run until stop condition is met
    
    %generate trial data
    if ttrial > TTrial       
        ttrial = 1; %reset trial time counter to 1
        t_trial_num = t_trial_num + 1; %increase trial count by 1
        
        %switch between whatever task is being run
        switch task
            case 'osc4'
                [fin,f,TTrial,cond,ttrialb,ITIb,ITI] = trial_osc4(dt);
            case 'xor'
                [fin,f,TTrial,cond,ttrialb,ITIb,ITI] = trial_xor(dt);
            case 'reaching'
                [fin,f,TTrial,cond,ttrialb,ITIb,ITI] = trial_reaching(dt,EMG);
            case 'oscfactor'
                [fin,f,TTrial,cond,ttrialb,ITIb,ITI] = trial_oscfactor(dt,FAC,EMG);
            case 'oscfactor_start_stop'
                [fin,f,TTrial,cond,ttrialb,ITIb,ITI,fac] = trial_oscfactor_start_stop(dt,FAC,EMG);
            case 'CI'
                [fin,f,TTrial,cond,ttrialb,ITIb,ITI,bias] = trial_CI(dt,DT,t_trial_num-Tinit,mode,T);
        end
        
        %return state variables of both networks back to their initial values
        if strcmp(task,'CI')                  
            v_tilde = v0_tilde; %return to initial value for discrete network        
            %return to initial value for spiking network
            if any(strcmp(mode,{'demean','RLStrain','RLStest','data','RLStest_weakPCs'}))
                uy_y = uy_y0; %return recurrent feedback to initial value
                %reset states specific to spiking model
                if strcmp(model,'spikes')
                    V = V0; %reset voltage state
                    VJ0s = VJ0s0; %J0 slow input current
                    VJ0f = VJ0f0; %J0 fast input current
                    Vmu = Vmu0;  %J0 mean current
                end
                
                %reset synaptic current states required when computing RLS
                if any(strcmp(mode,{'RLStrain','RLStest','data','RLStest_weakPCs'}))
                    vs = vs0; %slow presynaptic current/rate when using RLS
                    if strcmp(model,'spikes'); vf = vf0; end %fast presynaptic current when using RLS
                end             
            end           
        end
         
        %compute external inputs into factor generating and approximating
        %networks
        if (~have_factors && ~have_RNN); u_tilde_f_f = u_tilde_f * f; end  %f input into each factor generating unit       
        u_tilde_fin_fin = u_tilde_fin * fin; %fin input into each factor generating unit
        if any(strcmp(mode,{'demean','RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}))
            u_utilde_fin_fin = u * u_tilde_fin_fin; %fin into each factor approximating neuron
        end
        
        %after specified number of initialization trials....
        if t_trial_num > Tinit        
            %Make temporary empty arrays for saving things for a single trial
            switch mode           
                case 'PCA'      
                    %for collecting vtilde_prime of the rate network, for doing PCA
                    if have_RNN
                        vtilde_s = zeros(Ntilde,TTrial);
                    else
                        vtilde_s = zeros(Ntilde+m,TTrial);
                    end
                    
                case {'demean','RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}
                    
                    zs = zeros(m,TTrial); %for collecting z(t), for plotting
                    Vs = zeros(nplt,TTrial); %for collecting V(t), for plotting
                    ys_2 = zeros(n,TTrial);  %for collecdting y(t) for plotting
                    y_tilde_s = zeros(n,TTrial); %for collecting y_tilde(t) for plotting
                    
                    if strcmp(mode,'demean')
                        all_zs = zeros(N,TTrial); %for collecting recurrent inputs for mean computation
                        
                    elseif strcmp(mode,'data')                        
                        %empty struct for saving data
                        %data(t_trial_num-Tinit) = struct('fin',nan(ndata.fin,TTrial - ITI + ITIb),...
                        %    'z', nan(ndata.f,TTrial - ITI + ITIb),...
                        %    'f', nan(ndata.f,TTrial - ITI + ITIb),...
                        %    'y_tilde',nan(ndata.y_tilde,TTrial - ITI + ITIb),...
                        %    'y',nan(ndata.y,TTrial - ITI + ITIb),...
                        %    'V',nan(ndata.V,TTrial - ITI + ITIb),...
                        %    'S', spalloc(ndata.S,TTrial - ITI + ITIb, 0.05 * ndata.S * (TTrial - ITI + ITIb)),...
                        %    'VJ0s', nan(ndata.z0,TTrial - ITI + ITIb), ...
                        %    'VmuJ', nan(ndata.z0,TTrial - ITI + ITIb), ...
                        %    'Vmu', nan(ndata.z0,TTrial - ITI + ITIb), ...
                        %    'VJ0f', nan(ndata.z0,TTrial - ITI + ITIb), ...
                        %    'cond', nan(1,TTrial - ITI + ITIb));
                        
                        data(t_trial_num-Tinit) = struct('fin',nan(ndata.fin,TTrial - ITI + ITIb),...
                            'z', nan(ndata.f,TTrial - ITI + ITIb),...
                            'f', nan(ndata.f,TTrial - ITI + ITIb),...
                            'y_tilde',nan(ndata.y_tilde,TTrial - ITI + ITIb),...
                            'y',nan(ndata.y,TTrial - ITI + ITIb),...
                            'V',nan(ndata.V,TTrial - ITI + ITIb),...
                            'S', spalloc(ndata.S,TTrial - ITI + ITIb, 0.05 * ndata.S * (TTrial - ITI + ITIb)),...
                            'Vsum', nan(ndata.z0,TTrial - ITI + ITIb), ...
                            'cond', nan(1,TTrial - ITI + ITIb));
                        
                    end                 
            end            
        end        
    end
   
    %update state variables of factor generating population
    if have_factors
        %use the provided factors
        y_tilde = fac(ttrial,:)';
    else
        if have_RNN
            %vtilde_prime = Jtilde' * v_tilde + b_tilde'; %firing rates for computing PCs
             %vtilde_prime = Jtilde' * v_tilde; %firing rates for computing PCs
             vtilde_prime = v_tilde; %firing rates for computing PCs
            
            %discrete-time rate model
            if mod(ttrial,round(DT/dt)) == 1
                v_tilde = F(u_tilde_fin_fin(:,ttrial) + Jtilde' * v_tilde + b_tilde');
            end
            
        else
            %vtilde_prime = Jtilde_prime * [v_tilde;f(:,ttrial)]; %firing rates for computing PCs
            %vtilde_prime = Jtilde * v_tilde + u_tilde_f_f(:,ttrial);
            vtilde_prime = [v_tilde;f(:,ttrial)];
            
            %autoencoding rate model
            xinf = F(Jtilde * v_tilde + u_tilde_f_f(:,ttrial) + ...
                u_tilde_fin_fin(:,ttrial));
            v_tilde = xinf + (v_tilde - xinf) * etautilde;
            
        end
    end
    
    %update state variables of factor approx model
    if any(strcmp(mode,{'demean','RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}))
        
        %project firing rates down to the factor basis
        if ~have_factors
            y_tilde = wy_tilde' * vtilde_prime;
        end
        
        %integrate factor approximating model
        switch model           
            case 'spikes'
                
                if strcmp(mode, 'RLStest_weakPCs')
                    Vinfs(:,ttrial) = Vinfs(:,ttrial) + Vmu - VmuJ + uy_y + VJ0s + VJ0f + u_utilde_fin_fin(:,ttrial);
                    Vinf = Vmu - VmuJ + (w_strongPCs_filt) * (uy_y + VJ0s + VJ0f) + u_utilde_fin_fin(:,ttrial);
                else
                    Vinf = Vmu - VmuJ + uy_y + VJ0s + VJ0f + u_utilde_fin_fin(:,ttrial);
                end
                V = Vinf + (V - Vinf) * etauV; %Voltage
                
                S = V >= Vth; %spikes
                V(S) = Vth + Vr; %reset
                
                VJ0s = VJ0s * etaus + sum(Js(:,S),2); %slow J0 currents
                VJ0f = VJ0f * etauf + sum(Jf(:,S),2); %fast J0 currents
                Vmu = Vmu * etauf + sum(Jmu(:,S),2); %mean J0 currents
                
            case 'rate'               
                Vinf = F(uy_y + u_utilde_fin_fin(:,ttrial));
                vs = Vinf + (vs - Vinf) * etau;
                
        end
        
        %make presynaptic currents and concatenate
        if any(strcmp(mode,{'RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}))           
            if strcmp(model,'spikes')
                vs = etaus * vs + S;
                vf = etauf * vf + S;
            end
            
            %collect presynaptic current/rates
            if strcmp(model,'rate')
                %only 1 type of activity to collect
                v = vs;
            else
                %concatenate currents
                v = [vs;vf];
            end         
            
        end
        
        %generate output and learned feedback output
        %genearate real feedback after number of initialization trials
        if t_trial_num > Tinit        
            switch mode                
                case {'demean'}
                    y = y_tilde; %use factor targets as feedback 
                    uy_y = uy * y; %project from n to N             
                    
                case {'RLStrain','RLStest','RLStest_weakPCs'}                   
                    y = wy * v;  %feedback in n
                    %y(2) = 0; %added 2/1 to see effect of removing factors
                    uy_y = uy * y;  %project from n to N                
                    
                case {'LSdata'}    
                    %switch feedback type if using RLS or BLS solution
                    if strcmp(LSmode,'RLS')
                        y = wy * v; %feedback in n, from RLS solution               
                    elseif strcmp(LSmode,'shuff')
                        y = y_tilde + noise(:,ttime); %feedback in n, from ytilde plus shuffled RLS noise
                    end                    
                    uy_y = uy * y; %project from n to N                   
                    
                case 'LStest'                    
                    y = wy * v; %construct what would PC feedback, but is not fedback, simply readout
                    uy_y = J_BLS * v; %feedback in N basis, which is fedback               
                    
                case 'data'   
                     %switch feedback type if using RLS or BLS solution
                    if strcmp(solution,'RLS')
                        y = wy * v; %feedback currents in n basis
                        uy_y = uy * y; %project from n to N
                    elseif strcmp(solution,'BLS')
                        y = wy * v; %construct what would PC feedback, but is not fedback, simply readout
                        uy_y = J_BLS * v; %feedback in N basis, which is fedback
                    end                                        
            end
            
        else  %still in initial trial period, so use factor targets to set state of network      
            
            y = y_tilde; %feedback is targets in n basis
            uy_y = uy * y;  %project from n basis to N basis                   
        end
        
        %generate network output
        switch mode
            case 'demean'
                z = f(:,ttrial);
            otherwise
                %output
                z = w * [y;1];
        end
        
    end
    
    %after initial period
    if t_trial_num > Tinit      
        %after initial period, save certain results for different computations
        switch mode          
            case 'PCA'             
                vtilde_s(:,ttrial) = vtilde_prime;  %save for plotting and computing cov matrix
                
            case {'RLStrain','demean','LSdata','RLStest','LStest','data','RLStest_weakPCs'}             
                %save for plotting and computing nMSE    
                
                zs(:,ttrial) = z; %output
                ys_2(:,ttrial) = y; %factors
                y_tilde_s(:,ttrial) = y_tilde; %target factors
                
                %save voltage or firing rate
                switch model
                    case 'spikes'
                        Vs(:,ttrial) = V(1:nplt);
                        Vs(S(1:nplt),ttrial) = 2 * (Vth - Vr);
                    case 'rate'                       
                        Vs(:,ttrial) = vs(1:nplt);
                end                
                
                if strcmp(mode,'RLStrain')                   
                    if rand < 1/DTRLS %do RLS
                        
                        xP = Pv * v;
                        k = (1 + v' * xP)\xP';                        
                        Pv = Pv - xP * k;
                        wy = wy - (y - y_tilde) * k;
                        
                        xP = Py * [y;1];
                        k = (1 + [y;1]' * xP)\xP';                        
                        Py = Py - xP * k;
                        w = w - (z - f(:,ttrial)) * k;
                        
                    end
                    
                elseif strcmp(mode,'demean')               
                   switch model                
                       case 'spikes'                         
                           %save for mean computation, project targets from n
                           %basis to N basis, and add initial slow and fast
                           %currents
                           all_zs(:,ttrial) = uy * y_tilde + VJ0s + VJ0f;
                           
                       case 'rate'                         
                           %save for mean computation, project targets from n
                           %basis to N basis
                           all_zs(:,ttrial) = uy * y_tilde;
                           
                   end                 
                    
                elseif strcmp(mode,'LSdata')        
                    %save for BLS
                    ys(:,ttime) = y; %factor feedback, in n basis
                    vs_s(:,ttime) = v; %presynaptic currents or firing rates
                    %difference between generated feedback and target,
                    %which should be firing rate inputs projected down to
                    %PC basis minus desired mean projected down to PC basis
                    noise(:,ttime) = y - y_tilde;
                                        
                elseif strcmp(mode,'data')
                    
                    %if trial is outside the ITI, but gather some ITI data
                    if ttrial >= ttrialb(1) && ttrial <= ttrialb(2)
                        
                        data(t_trial_num-Tinit).fin(:,ttrial-ttrialb(1)+1) = fin(1:ndata.fin,ttrial);
                        data(t_trial_num-Tinit).f(:,ttrial-ttrialb(1)+1)  = f(1:ndata.f,ttrial);
                        data(t_trial_num-Tinit).z(:,ttrial-ttrialb(1)+1) = z(1:ndata.f,1);
                        data(t_trial_num-Tinit).cond(:,ttrial-ttrialb(1)+1)  = cond;                     
                        data(t_trial_num-Tinit).y_tilde(:,ttrial-ttrialb(1)+1) = y_tilde(1:ndata.y_tilde,1);
                        data(t_trial_num-Tinit).y(:,ttrial-ttrialb(1)+1) = y(1:ndata.y_tilde,1);
                                            
                        switch model
                            case 'spikes'
                                
                                data(t_trial_num-Tinit).S(:,ttrial-ttrialb(1)+1) = S(1:ndata.S,1);
                                data(t_trial_num-Tinit).Vsum(:,ttrial-ttrialb(1)+1) = VJ0s(1:ndata.z0,1) + VJ0f(1:ndata.z0,1);
                                %data(t_trial_num-Tinit).VJ0s(:,ttrial-ttrialb(1)+1) = VJ0s(1:ndata.z0,1);
                                %data(t_trial_num-Tinit).VJ0f(:,ttrial-ttrialb(1)+1) = VJ0f(1:ndata.z0,1);
                                %data(t_trial_num-Tinit).VmuJ(:,ttrial-ttrialb(1)+1) = VmuJ(1:ndata.z0,1);
                                %data(t_trial_num-Tinit).Vmu(:,ttrial-ttrialb(1)+1) = Vmu(1:ndata.z0,1);
                                data(t_trial_num-Tinit).V(:,ttrial-ttrialb(1)+1) = V(1:ndata.V,1);
                                
                            case 'rate'
                                
                                data(t_trial_num-Tinit).V(:,ttrial-ttrialb(1)+1) = v(1:ndata.V,1);
                         
                        end
                                                
                    end
                    
                end
                
        end
        
        % text output, plot things, perform computations
        if ttrial == TTrial
            
            %perform computations, which are subroutine specific
            switch mode                             
                case 'PCA'      
                    %update covariance matrix
                    COV_vtilde = COV_vtilde + vtilde_s * vtilde_s';
                    %compute PCs from cov matrix, and arrange them from
                    %largest to smallest
                    [wy_tilde,d] = eig(COV_vtilde);
                    wy_tilde = fliplr(wy_tilde);
                    d = flipud(diag(d));                  
                    %pick the number of dimensions that accounts for fracPCA of variance
                    n = find(cumsum(d)/sum(d) > fracPCA, 1, 'first');
                    wy_tilde = wy_tilde(:,1:n);
                    
                case 'demean'              
                    sum_VmuJ = sum_VmuJ + sum(all_zs,2); %add sum of current trial'v currents to running sum
                    VmuJ = sum_VmuJ/ttime; %divde by total elapsed time to compute mean
                                      
                case {'RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}
                    
                    %compute normalized output error on this trial
                    nMSE(t_trial_num-Tinit) = sum(diag((ys_2 - y_tilde_s) * (ys_2 - y_tilde_s)'))/...
                        sum(diag(y_tilde_s * y_tilde_s'));
                    
                    if strcmp(mode,'RLStest_weakPCs')
                        Vinf_bar = Vinfs/(t_trial_num - Tinit);
                    end
                    
            end
            
            %printing and plotting
            clc;
            
            %plot f and fin
            set(lh_f,'XData',dt*[ttrial-TTrial+1:ttrial],'YData',f(1,:));
            set(lh_fin,'XData',dt*[ttrial-TTrial+1:ttrial],'YData',fin(1,:));
            
            switch mode               
                case 'PCA'                    
                    fprintf('%s, %g PCs, %g trials of %g \n', mode, n, t_trial_num-Tinit, T);
                    
                    %project down to the PCs
                    temp = wy_tilde' * vtilde_s;
                    max_temp = 0;
                    %plot a few of them
                    for i = 1:nplt
                        max_temp = max_temp + abs(min(temp(i,:)));
                        set(lh_ytilde(i),'XData',dt*[ttrial-TTrial+1:ttrial],'YData', temp(i,:) + max_temp);
                        max_temp = max_temp + max(temp(i,:));
                    end
                    axis tight
                    
                case {'demean','RLStrain','RLStest','LSdata','LStest','data','RLStest_weakPCs'}
                    
                    if any(strcmp(mode,{'demean'}))                     
                        fprintf('%s, %g trials of %g \n', mode, t_trial_num-Tinit, T);                       
                    else                      
                        %print median (across trials) nMSE
                        fprintf('%s, %g Error, %g trials of %g \n', ...
                            mode, nanmedian(nMSE), t_trial_num-Tinit, T);                                               
                    end
                    
                    %plot generated output
                    set(lh_z,'XData',dt*[ttrial-TTrial+1:ttrial],'YData',zs(1,:));
                    axis tight
                    
                    %plot some factor trajectories
                    maxV = 0;
                    for i = 1:nplt
                        maxV = maxV + abs(min(ys_2(i,:)));
                        set(lh_y(i),'XData',dt*[ttrial-TTrial+1:ttrial],'YData', ys_2(i,:) + maxV);
                        set(lh_ytilde(i),'XData',dt*[ttrial-TTrial+1:ttrial],'YData', y_tilde_s(i,:) + maxV);
                        maxV = maxV + max(ys_2(i,:));
                    end                  
                    axis tight
                    
                    %plot some voltages for spiking and rate variable for
                    %rate models
                    maxV = 0;
                    for i = 1:nplt
                        maxV = maxV + 0.05 * abs(min(Vs(i,:)));
                        set(lh_V(i),'XData',dt*[ttrial-TTrial+1:ttrial],'YData', 0.05 * Vs(i,:) + maxV);
                        maxV = maxV + 0.05 * max(Vs(i,:));
                    end                 
                    axis tight
                    
            end           
            drawnow; %update figure
                          
        end
        
        %counter of number of timesteps that have passed in total, over all
        %trials, only starts counting after initial period is passed
        ttime = ttime + 1;
        
    end
    
    %quit simulation loop
    if t_trial_num == T+Tinit && ttrial == TTrial      
        %quit loop
        go = false;
        
    end
    
    %counter for number of timesteps that have passed in THIS trial
    ttrial = ttrial + 1;
    
end

%% output
switch mode    
    case 'PCA'       
        varargout{1} = wy_tilde; %PC loadings
        n = size(wy_tilde,2); 
        varargout{2} = n; %number of PCs
        varargout{3} = COV_vtilde; %rate covariance matrix
        
    case 'demean'        
        varargout{1} = VmuJ; %mean input into each neuron
        
    case 'RLStrain'        
        varargout{1} = wy; %learned projection to factors
        varargout{2} = w; %learned output
        varargout{3} = Pv; %inverse covariance of v
        varargout{4} = Py; %inverse covariance of y
        
    case {'RLStest','LStest','RLStest_weakPCs'}      
        varargout{1} = nMSE; %normalized mean square error for each trial
        varargout{2} = Vinf_bar;
        
    case 'LSdata'       
        varargout{1} = noise; %generated factors minus target factors
        varargout{2} = ys; %generatd factors
        varargout{3} = vs_s; %synaptic currents
        
    case 'data'
        varargout{1} = data; %collected data
        
end

close(fh); %close figure
