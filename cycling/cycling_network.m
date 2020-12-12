function varargout = cycling_network(mode,T,factors,emg,varargin)

global N P m u u_in muf gs gf

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
dt = 1e-3;
taus = 1e-1;
tauf = 5e-3;
tauV = 1e-2;
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
lh_f = line(ah1,'Color','b','LineWidth',2,'Marker','none','LineStyle',':');
lh_fin = line(ah1,'Color','k','LineWidth',1,'Marker','none','LineStyle',':');

nplt = 3; %number of factors and rates/voltages to plot
        
%line handle for generated output
lh_z = line(ah1,'Color','r','LineWidth',1,'Marker','none','LineStyle','-');

%line handles for factors, factor targets and rates/voltages
for i = 1:nplt
    lh_y(i) = line(ah2,'Color','k','LineWidth',2,'Marker','none','LineStyle','-');
    lh_ytilde(i) = line(ah2,'Color','r','LineWidth',2,'Marker','none','LineStyle',':');
    lh_V(i) = line(ah3,'Color','b','LineWidth',2,'Marker','none','LineStyle','-');
end
                
%% Unpack varargin, depending on which subroutine is being run

if any(strcmp(mode,{'train','test'}))       
    vJsbar = varargin{1};
end

if strcmp(mode,{'test'})
    w = varargin{2}; %learned connection, N-space to P-space
    W = varargin{3}; %learned output connections    
end    

%% Initialize matrices for saving data, depending on which subroutine is being run

if strcmp(mode,'demean')
    sum_vJsbar = zeros(N,1);
    vJsbar = zeros(N,1);
    
elseif any(strcmp(mode,{'train','test'}))      
    nMSEy = NaN(T,1); %normalized mean square error, saved for each trial
    nMSEf = NaN(T,1); %normalized mean square error, saved for each trial
    
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

if any(strcmp(mode,{'test'}))
    rng(4);
end

%% Run simulation

while go %run until stop condition is met
    
    %generate trial data
    if ttrial > TTrial  
        
        ttrial = 1; %reset trial time counter to 1
        t_trial_num = t_trial_num + 1; %increase trial count by 1
        
        [fin,TTrial] = trial_oscfactor(dt);         
        ytilde = factors';
        f = emg';
                 
        zs = zeros(m,TTrial); %for collecting z(t), for plotting
        vs = zeros(nplt,TTrial); %for collecting v(t), for plotting
        ys = zeros(P,TTrial);  %for collecdting y(t) for plotting

        if strcmp(mode,'demean')
            zsT = zeros(N,TTrial); %for collecting recurrent inputs for mean computation
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

    %generate output and learned feedback output
    %genearate real feedback after number of initialization trials
    if t_trial_num > Tinit 
        
        switch mode                
            case {'demean'}
                y = ytilde(:,ttrial); %use factor targets as feedback 
                z = f(:,ttrial);

            case {'train','test'}                   
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

        end
                        
        % text output, plot things, perform computations
        if ttrial == TTrial

            %compute normalized output error on this trial
            nMSEy(t_trial_num-Tinit) = sum(diag((ys - ytilde) * (ys - ytilde)'))/...
                sum(diag(ytilde * ytilde'));

            nMSEf(t_trial_num-Tinit) = sum(diag((zs - f) * (zs - f)'))/...
                sum(diag(f * f'));

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
        set(lh_f,'XData',dt*[ttrial-TTrial+1:ttrial],'YData',f(1,:));
        set(lh_fin,'XData',dt*[ttrial-TTrial+1:ttrial],'YData',fin(1,:));

        if any(strcmp(mode,{'demean'}))                     
            fprintf('%s, %g trials of %g \n', mode, t_trial_num-Tinit, T);                       
        else                      
            %print median (across trials) nMSEy
            fprintf('%s, %g Error, %g trials of %g \n', ...
                mode, nanmedian(nMSEy), t_trial_num-Tinit, T);                                               
        end

        %plot generated output
        set(lh_z,'XData',dt*[ttrial-TTrial+1:ttrial],'YData',zs(1,:));
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

        %plot some voltages for spiking and rate variable for
        %rate models
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
        varargout{1} = nMSEy; %normalized mean square error for each trial
        varargout{2} = nMSEf;
        
end

close(fh); %close figure