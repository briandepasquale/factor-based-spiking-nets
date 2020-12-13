function [xprimes,x,fin,f,TTrial] = rate_model(emg,varargin)

global Ntilde m dt utilde utilde_in Jtilde

F = @(x)tanh(x); %nonlinearity
tau = 1e-2;
etau = exp(-dt/tau);

%% Initalize state variables

ttrial = inf; %time in current trial
TTrial = 0; %total time in current trial
go = true; %flag to quit loop

%%

if isempty(varargin)
    x = 1e0 * randn(Ntilde,1);
else
    x = varargin{1};
end

%% Run simulation

while go %run until stop condition is met
    
    if ttrial > TTrial       
        ttrial = 1; %reset trial time counter to 1
        [fin,f,TTrial] = trial_reaching(dt,emg);            
        xprimes = zeros(Ntilde+m,TTrial);  
    end

    xprime = [x; f(:,ttrial)];

    xinf = F(Jtilde * x + utilde * f(:,ttrial) + utilde_in * fin(:,ttrial));
    x = xinf + (x - xinf) * etau;                                            
    xprimes(:,ttrial) = xprime;
    
    %quit simulation loop
    if ttrial == TTrial      
        go = false;      
    end
    
    %counter for number of timesteps that have passed in THIS trial
    ttrial = ttrial + 1;
    
end   