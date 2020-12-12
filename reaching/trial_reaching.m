function [fin,fout,TTrial,cond,ttrialb,ITIb,ITI] = trial_reaching(dt,EMG)
%%

C = size(EMG,1); %number of unique trial types
Nout = size(EMG,3); %number of outputs

cond = randsample(C,1); %trial type for this trial

%change made here 9/17 to deal with removed interpolation signal at end of
%EMG
%event = [0.5,1.31-0.5] * (1/dt); %timing of various trial events
event = round([0.35,0.15,0.81-0.5] * (1/dt)); %timing of various trial events

ITIb = 0.4 * (1/dt); %minimum time of inter-trial interval
ITI = 2.0 * (1/dt); %avearage time of inter-trial interval
ITI = 2 * round(0.5 * (ITIb + round(exprnd(ITI)))); %ITI for this trial

TTrial = sum(event) + ITI; %length of this trial

ttrialb = [ITI/2 - ITIb/2 + 1, sum(event) + ITI/2 + ITIb/2]; %data recording boundaries

%define elements of task input
fin0 = zeros(1,ITI/2);
%fin1 = (1/sqrt(2)) * 1.0 * cos((cond/C) * 2*pi) * ones(1,sum(event(1:2)));
%fin2 = (1/sqrt(2)) * 1.0 * sin((cond/C) * 2*pi) * ones(1,sum(event(1:2)));
%fin3 = (1/sqrt(2)) * 0.5 * ones(1,sum(event(1)));

fin1 = 0.5 * cos((cond/C) * 2*pi) * ones(1,sum(event(1:3)));
fin2 = 0.5 * sin((cond/C) * 2*pi) * ones(1,sum(event(1:3)));
fin3 = 1.0 * ones(1,sum(event(1)));

fout1 = -0.0 + 2 * reshape(repmat(reshape(squeeze(EMG(cond,:,:)),1,[]),10,1),[],Nout)';
fout1(1,:) = -1 * fout1(1,:);
fout0 = -0.0 * ones(1,ITI/2);
fout2 = -0.5 * ones(1,ITI/2);

%concatenate
fin = [fin0, fin3, zeros(1,sum(event(2:3))),fin0;...
    fin0, fin1,fin0;...
    fin0, fin2,fin0];
fout = [fout0, fout1(1,:),fout0;...
    fout0, fout1(2,:), fout0;...
    fout2, -0.5 * ones(1,sum(event(1:3))) ,fout2];

