function varargout = spiking_network(task,mode,T,varargin)

global N u u_in muf gs gf dt

%etaus: timescale of s synaptic currents in spiking model
%etauf: timescale of f synaptic currents in spiking model
%etauV: timescale of voltage in spiking model
%DTRLS: number of time steps between RLS updates
%dt: timestep
%tauf: decay for fast synapses
%taus: decay for slow synapses
%tauV: decay for voltage in spiking net
%vth: spike threshold
%vr: reset voltage

Tinit = 50;
DTRLS = 2;
tauV = 1e-2;
taus = 1e-1;
tauf = 5e-3;
vth = 1e-16;
vr = -10;
etaus = exp(-dt/taus);
etauf = exp(-dt/tauf);
etauV = exp(-dt/tauV);

%Jmu: mean recurrent matrix in factor approximating network for spiking
    %neurons
%J0f: random f connections for facdtor approximating network for spiking
    %neurons
%J0s: random s connections for facdtor approximating network for spiking
    %neurons

rng(1);
Jmu = muf * (1/tauf) * 1/(N) * ones(N);
J0f = gf * (1/tauf) * 1/sqrt(N) * randn(N);
J0s = gs * (1/taus) * 1/sqrt(N) * randn(N);

%% Unpack varargin, depending on which subroutine is being run

switch task
    case 'reaching'
        V = varargin{1};
        emg = varargin{2};
        m = size(emg, 3) + 1;
        P = size(V, 2);
    case 'cycling'
        factors = varargin{1};
        emg = varargin{2};
        m = size(emg, 2);
        P = size(factors, 2);
    case 'CI'
        V = varargin{1};
        m = 1;
        P = size(V, 2);
    case 'reaching-cycling'
        
        factors_V = varargin{1};
        emg = varargin{2};
        
        reaching_emg = emg{1};
        cycling_emg = emg{2};
        
        m = size(reaching_emg, 3) + 1 + size(cycling_emg, 2);
        
        V = factors_V{1};
        factors = factors_V{2};
        
        P = size(V, 2) + size(factors, 2);
        
end

if any(strcmp(mode,{'train', 'test', 'data'}))       
    vJsbar = varargin{3};
end

if any(strcmp(mode,{'test', 'data'}))
    w = varargin{4}; %learned connection, N-space to P-space
    W = varargin{5}; %learned output connections    
end   

%% Figure, for plotting various things as simulation runs

%figure handle
fh = figure('Color','w', 'menubar', 'none','NumberTitle','off','ToolBar','none');

%axes handles
ah1 = axes(fh,'LineWidth',2,'FontSize',12,'Position',[0.1 0.75 0.8 0.2]);
ah2 = axes(fh,'LineWidth',2,'FontSize',12,'Position',[0.1 0.05 0.8 0.3]);
ah3 = axes(fh,'LineWidth',2,'FontSize',12,'Position',[0.1 0.40 0.8 0.3]);

xlabel(ah2,'time (s)');

%delete any old lines in the axes handles, from previous executions of code
delete(findobj(ah1,'Type','Line')); delete(findobj(ah2,'Type','Line'));

%line handles for input and target output
lh_fin = line(ah1,'Color','k','LineWidth',1,'Marker','none','LineStyle',':');

nplt = 3; %number of factors and voltages to plot
        
%line handle for generated output
nplt_z = min(m, 7);
for i = 1:nplt_z
    lh_f(i) = line(ah1,'Color','b','LineWidth',2,'Marker','none','LineStyle',':');
    lh_z(i) = line(ah1,'Color','r','LineWidth',1,'Marker','none','LineStyle','-');
end

%line handles for factors, factor targets and rates/voltages
for i = 1:nplt
    lh_y(i) = line(ah2,'Color','k','LineWidth',2,'Marker','none','LineStyle','-');
    lh_ytilde(i) = line(ah2,'Color','r','LineWidth',2,'Marker','none','LineStyle',':');
    lh_V(i) = line(ah3,'Color','b','LineWidth',2,'Marker','none','LineStyle','-');
end
       

%% Initialize matrices for saving data, depending on which subroutine is being run

if strcmp(mode,'demean')
    sum_vJsbar = zeros(N,1);
    vJsbar = zeros(N,1);
    
elseif any(strcmp(mode,{'train','test','data'}))      
    nMSE = NaN(T,1); %normalized mean square error, saved for each trial
    
    if any(strcmp(mode,{'train'}))                       
        W = zeros(m,P+1); %for computing learned output matrix
        PW = 1 * eye(P+1); %inverse covariance matrix

        w = zeros(P,2*N); %P readout weights, half of J
        Pw = 1 * eye(2*N); %inverse covariance matrix
                                                                    
    end   
end

%% Initalize state variables

%initialize random seed, to set network into same initial state
%go to different initial state it testing or gathering data
rng(3);
     
y = zeros(P,1); %learned input current
ss = zeros(N,1); %slow presynaptic current/firing rate when using RLS
sf = zeros(N,1); %fast presynaptic current when using RLS

v = 1e-3 * randn(N,1); %spike net state
J0ss = zeros(N,1); %J0 slow input current
J0fs = zeros(N,1); %J0 fast input current
J0fbars = zeros(N,1); %J0 mean current
  
%set various counters
ttrial = inf; %time in current trial
TTrial = 0; %total time in current trial
t_trial_num = 0; %trial number
ttime = 1; %time across all trials
go = true; %flag to quit loop

%% For CI task, produce a more adapted initial state for the spiking network and save it

if strcmp(task,'CI')     
    
    for i = 1:1000
        [~,fin] = backprop_rate_model;
        vinf = J0fbars - vJsbar + u * y + J0ss + J0fs + u_in * fin(:,i);
        v = vinf + (v - vinf) * etauV;
        S = v >= vth;
        v(S) = vth + vr;
        J0ss = J0ss * etaus + sum(J0s(:,S),2);
        J0fs = J0fs * etauf + sum(J0f(:,S),2); 
        J0fbars = J0fbars * etauf + sum(Jmu(:,S),2);
        ss = etaus * ss + S;
        sf = etauf * sf + S;
    end

    y0 = y;
    v0 = v;
    J0ss0 = J0ss; 
    J0fs0 = J0fs;
    J0fbars0 = J0fbars;
    sf0 = sf;
    ss0 = ss;    
    
end
    
%%

if any(strcmp(mode,{'test'}))
    rng(4);
end

%% Run simulation

while go %run until stop condition is met
    
    %generate trial data
    if ttrial > TTrial  
        
        ttrial = 1; %reset trial time counter to 1
        t_trial_num = t_trial_num + 1; %increase trial count by 1
        
        switch task
            
            case 'reaching-cycling'
                
                if rand > 1/2
                
                    if t_trial_num == 1
                        [xprimes,x,fin,f,TTrial] = rate_model(reaching_emg);
                    else
                        [xprimes,x,fin,f,TTrial] = rate_model(reaching_emg, x);
                    end
                    
                    ytilde = cat(1, V' * xprimes, zeros(size(factors, 2), size(xprimes,2)));
                    f = cat(1, f, zeros(size(cycling_emg, 2), size(xprimes,2)));
                    fin = cat(1, fin, zeros(size(u_in,2) - size(fin, 1), size(xprimes,2)));
                
                else
                    
                    [fin,TTrial] = trial_cycling(dt);         
                    ytilde = factors';
                    f = cycling_emg';
                    
                    f = cat(2, f, zeros(size(f,1), TTrial/2));
                    ytilde = cat(2, ytilde, zeros(size(ytilde,1), TTrial/2));
                    fin = cat(2, fin, ones(size(fin,1), TTrial/2));              
                    TTrial = size(f,2);
                    
                    ytilde = cat(1, zeros(size(V, 2), size(ytilde,2)), ytilde);
                    f = cat(1, zeros(size(reaching_emg, 3) + 1, size(ytilde,2)), f);
                    fin = cat(1, zeros(size(u_in, 2) - size(fin, 1), size(ytilde,2)), fin);

                end
            
            case 'reaching'
                if t_trial_num == 1
                    [xprimes,x,fin,f,TTrial] = rate_model(emg);
                else
                    [xprimes,x,fin,f,TTrial] = rate_model(emg, x);
                end
                ytilde = V' * xprimes;
                
            case 'cycling'
                [fin,TTrial] = trial_cycling(dt);         
                ytilde = factors';
                f = emg';    
                
            case 'CI'
                [xprimes,fin,f,TTrial] = backprop_rate_model;
                ytilde = V' * xprimes;
                
                %return state parameters of spiking network to initial
                %state
                y = y0;
                v = v0;
                J0ss = J0ss0; 
                J0fs = J0fs0;
                J0fbars = J0fbars0;
                sf = sf0;
                ss = ss0; 
                
        end
                 
        zs = zeros(m,TTrial); %for collecting z(t), for plotting
        vs = zeros(nplt,TTrial); %for collecting v(t), for plotting
        ys = zeros(P,TTrial);  %for collecdting y(t) for plotting

        if strcmp(mode,'demean')
            zsT = zeros(N,TTrial); %for collecting recurrent inputs for mean computation
            
        elseif strcmp(mode, 'data')
            
            if t_trial_num > Tinit
                data(t_trial_num - Tinit) = struct('z', nan(m, TTrial), 'f', nan(m, TTrial), ...
                    'y', nan(P, TTrial), 'ytilde', nan(P, TTrial), 'fin', nan(size(u_in,2), TTrial), ...
                    'S', nan(N, TTrial));
            end
            
        end  
        
    end
           
    %integrate factor approximating model
    vinf = J0fbars - vJsbar + u * y + J0ss + J0fs + u_in * fin(:,ttrial);
    v = vinf + (v - vinf) * etauV; %Voltage

    S = v >= vth; %spikes
    v(S) = vth + vr; %reset

    J0ss = J0ss * etaus + sum(J0s(:,S),2); %slow J0 currents
    J0fs = J0fs * etauf + sum(J0f(:,S),2); %fast J0 currents
    J0fbars = J0fbars * etauf + sum(Jmu(:,S),2); %mean J0 currents

    %make presynaptic currents and concatenate
    ss = etaus * ss + S;
    sf = etauf * sf + S;
    s = [ss;sf];   

    if t_trial_num > Tinit 
        
        switch mode                
            case {'demean'}
                y = ytilde(:,ttrial); %use factor targets as feedback 
                z = f(:,ttrial);

            case {'train','test','data'}                   
                y = w * s;  %feedback in P
                z = W * [y;1];

        end

    else
        
        y = ytilde(:,ttrial);
        z = f(:,ttrial);
      
    end
    
    zs(:,ttrial) = z; %output
    ys(:,ttrial) = y; %factors

    vs(:,ttrial) = v(1:nplt);
    vs(S(1:nplt),ttrial) = 2 * (vth - vr);               

    if strcmp(mode,'demean')                                       
       zsT(:,ttrial) = u * ytilde(:,ttrial) + J0ss + J0fs;                            
    end
            
    if t_trial_num > Tinit      
                
        if strcmp(mode,'train') && (rand < 1/DTRLS)

            xP = Pw * s;
            k = (1 + s' * xP)\xP';                        
            Pw = Pw - xP * k;
            w = w - (y - ytilde(:,ttrial)) * k;

            xP = PW * [y;1];
            k = (1 + [y;1]' * xP)\xP';                        
            PW = PW - xP * k;
            W = W - (z - f(:,ttrial)) * k;
            
        elseif strcmp(mode, 'data')                           
                        
            data(t_trial_num - Tinit).ytilde(:,ttrial)  = ytilde(:,ttrial);
            data(t_trial_num - Tinit).y(:,ttrial)  = y;
            data(t_trial_num - Tinit).f(:,ttrial)  = f(:,ttrial);
            data(t_trial_num - Tinit).z(:,ttrial) = z;
            data(t_trial_num - Tinit).fin(:,ttrial) = fin(:,ttrial);
            data(t_trial_num - Tinit).S(:,ttrial) = S;

        end
                        
        % text output, plot things, perform computations
        if ttrial == TTrial

            if strcmp(task, 'CI')
                nMSE(t_trial_num-Tinit) = (sign(zs(end)) == sign(f(end)));
            else
                %compute normalized output error on this trial
                nMSE(t_trial_num-Tinit) = sum(diag((ys - ytilde) * (ys - ytilde)'))/...
                    sum(diag(ytilde * ytilde'));
                
            end

            if strcmp(mode,'demean')                                       

                sum_vJsbar = sum_vJsbar + sum(zsT,2);
                vJsbar = sum_vJsbar/(ttime * size(zsT,2)); %divde by total elapsed time to compute mean

                %counter of number of timesteps that have passed in total, over all
                %trials, only starts counting after initial period is passed
                ttime = ttime + 1;

            end
                                             
        end
        
    end
            
    %after initial period
    if ttrial == TTrial
                        
        %printing and plotting
        clc;

        %plot f and fin
        set(lh_fin,'XData',dt*[ttrial-TTrial+1:ttrial],'YData',fin(1,:));

        if any(strcmp(mode,{'demean'}))                     
            fprintf('%s, %g trials of %g \n', mode, t_trial_num-Tinit, T);                       
        else                      
            %print median (across trials) nMSE
            if strcmp(task, 'CI')
                fprintf('%s, %g frac. correct, %g trials of %g \n', ...
                    mode, nanmean(nMSE), t_trial_num-Tinit, T); 
            else
                fprintf('%s, %g Error, %g trials of %g \n', ...
                    mode, nanmedian(nMSE), t_trial_num-Tinit, T);  
            end
        end

        %plot generated output
        for i = 1:nplt_z
            set(lh_f(i),'XData',dt*[ttrial-TTrial+1:ttrial],'YData', f(i,:));
            set(lh_z(i),'XData',dt*[ttrial-TTrial+1:ttrial],'YData', zs(i,:));
        end
        axis tight

        %plot some factor trajectories
        maxV = 0;
        for i = 1:nplt
            maxV = maxV + abs(min(ys(i,:)));
            set(lh_y(i),'XData',dt*[ttrial-TTrial+1:ttrial],'YData', ys(i,:) + maxV);
            set(lh_ytilde(i),'XData',dt*[ttrial-TTrial+1:ttrial],'YData', ytilde(i,:) + maxV);
            maxV = maxV + max(ys(i,:));
        end                  
        axis tight

        %plot some voltages
        maxV = 0;
        for i = 1:nplt
            maxV = maxV + 0.05 * abs(min(vs(i,:)));
            set(lh_V(i),'XData',dt*[ttrial-TTrial+1:ttrial],'YData', 0.05 * vs(i,:) + maxV);
            maxV = maxV + 0.05 * max(vs(i,:));
        end                 
        axis tight

        drawnow; %update figure
                                  
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
        
    case 'demean'        
        varargout{1} = vJsbar; %mean input into each neuron
        
    case 'train'        
        varargout{1} = w; %learned projection to factors
        varargout{2} = W; %learned output
        varargout{3} = Pw; %inverse covariance of s
        varargout{4} = PW; %inverse covariance of y
        
    case {'test'}      
        varargout{1} = nMSE; %normalized mean square error for each trial
        
    case {'data'}      
        varargout{1} = data;   
        
end

close(fh); %close figure