clearvars;

%load the test data
load('./factor_based_rate_data.mat', 'y', 'fin', 'spikes', 'etaus', 'etauf', 'J0', 'dt', 'uin', 'u');

min = size(fin,1); %size of external input
N = size(spikes,1); %number of spiking neurons
R = size(spikes,3); %number of trials
T = size(spikes,2); %length of trial
P = size(y,1); %number of factors

%flatten all trials together
fin = reshape(fin, min, T*R);
y = reshape(y, P, T*R);
spikes = reshape(spikes, N, T*R);

raw_factor_input = u*y + uin*fin; %factors via J_fac and external inputs
fin = cat(1, ones(1, T*R), fin); %add vector of 1's to fin
y_prime = cat(1, y, fin); %cat factors and externa linputs

%filter spikes to get s
ss = zeros(N, T * R); sf = zeros(N, T * R);
for i = 2:T*R
    ss(:,i) = etaus * ss(:,i-1) + spikes(:,i);
    sf(:,i) = etauf * sf(:,i-1) + spikes(:,i);
end

raw_non_factor_input = J0 * [ss; sf]; %mutliply s and J0 to get J0 input
uJ0 = raw_non_factor_input / y_prime; %solve for uJ0 to find factor input due to J0
J0_factor_input = uJ0 * y_prime; %find the J0 factor input 

factor_based_input = raw_factor_input + J0_factor_input; %find the factor based input as the sum of the raw input and J0 input
non_factor_based_input = raw_non_factor_input - J0_factor_input; %find the non-factor based input as the difference of the raw J0 input and J0 factor-based input

bs = glmfit(factor_based_input(633,:), spikes(633,:), 'poisson', 'link', 'log'); %run GLM to find the factor-based rate for neuron 633, the neuron plotted in Figure 4
factor_based_rate = exp(bs(1) + bs(2) * factor_based_input(633,:))/dt;

figure; %plot some stuff

subplot(311); 
plot(reshape(non_factor_based_input(633,:), T, R));
set(gca,'ylim', [-20, 20]);
xlabel('time (ms)');
title('non-factor based inputs for neuron 633');

subplot(312); 
plot(reshape(factor_based_input(633,:), T, R));
set(gca,'ylim', [-20, 20]);
xlabel('time (ms)');
title('factor based inputs for neuron 633');

subplot(313); 
plot(reshape(factor_based_rate, T, R));
set(gca,'ylim', [0, 60]);
xlabel('time (ms)');
title('factor-based firing rate for neuron 633');
    