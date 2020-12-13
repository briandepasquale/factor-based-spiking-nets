function [xprimes,fin,f,TTrial] = backprop_rate_model

global Ntilde dt utilde_in Jtilde btilde DT x0

F = @(x)tanh(x);
DT = 1e-2;

%% Initalize state variables

ttrial = inf; %time in current trial
TTrial = 0; %total time in current trial
go = true; %flag to quit loop
x = x0;

%% Run simulation

while go %run until stop condition is met
    
    if ttrial > TTrial       
        ttrial = 1; %reset trial time counter to 1
        [fin,f,TTrial] = trial_CI(dt,DT);            
        xprimes = zeros(Ntilde,TTrial);  
    end
    
    if mod(ttrial,round(DT/dt)) == 1
        x = F(utilde_in * fin(:,ttrial) + Jtilde' * x + btilde');
    end
                                           
    xprimes(:,ttrial) = x;
    
    %quit simulation loop
    if ttrial == TTrial      
        go = false;      
    end
    
    %counter for number of timesteps that have passed in THIS trial
    ttrial = ttrial + 1;
    
end   