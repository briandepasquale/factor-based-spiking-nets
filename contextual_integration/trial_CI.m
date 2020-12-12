function [fin,fout,TTrial] = trial_CI(dt,DT)
%%

bias_val = 0.1;
stddev_val = 0.04;
%number of white noise inputs
nwninputs = 2;

%is input one hot or not
cond = round(rand);
%both inputs, hot or not
cond2 = [cond, ~cond];

%length of trial in ms
event = [1.0] * (1/dt); %timing of various trial events
%number of timesteps for discrete time model
ntime = [1.0] * (1/DT);

ITIb = 0.0 * (1/dt); %minimum time of inter-trial interval
ITI = 0.0 * (1/dt); %avearage time of inter-trial interval
ITI = 2 * round(0.5 * (ITIb + round(exprnd(ITI)))); %ITI for this trial

TTrial = sum(event) + ITI; %length of this trial

ttrialb = [ITI/2 - ITIb/2 + 1, sum(event) + ITI/2 + ITIb/2]; %data recording boundaries

bias = bias_val * 2.0 * (rand(1,nwninputs) - 0.5);
stddev = stddev_val / sqrt(1.0/ntime);
%noise for both inputs
%added ones(ntime,1) on 1/26, way after generated all results, not sure why
%it was needed
noise = stddev * randn(ntime, nwninputs) + ones(ntime,1) * bias;
%David's convention, for allowing context to "establish"
noise(1:6,:) = 0.0;

%make condition vector in time
cond2 = ones(ntime,1) * cond2;

%not used
fout0 = 0.0 * ones(1,ITI/2);
%compute cumsum for both noise inputs
fout1 = cumsum(noise);
%select the hot one as the output
fout1 = fout1(logical(cond2));

%define elements of task input
fin0 = zeros(1,ITI/2);
%concatenate the noise and the condition inputs
fin1 = cat(2,noise, cond2);

%concatenate
fin = [fin0, reshape(repmat(fin1(:,1),1,10)',1,[]),fin0;...
    fin0, reshape(repmat(fin1(:,2),1,10)',1,[]), fin0;...
    fin0, reshape(repmat(fin1(:,3),1,10)',1,[]), fin0;...
    fin0, reshape(repmat(fin1(:,4),1,10)',1,[]), fin0];
fout = [fout0, reshape(repmat(fout1(:,1),1,10)',1,[]),fout0];

