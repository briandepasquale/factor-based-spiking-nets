function [fin,TTrial,cond,ttrialb,ITIb,ITI] = trial_oscfactor(dt)
%%

%timing of event sequence, for each trial
event = [0.05,1.95] * (1/dt);

%length of a trial
TTrial = sum(event);

%inter-trial interval basline
ITIb = 0;     
ITI = 0;

%trial type
cond = 1;
%start and stop of trial for collecting data
ttrialb = [1,TTrial];

%input pulse
fin1 = 2.0 * ones(1,event(1));
fin2 = zeros(1,sum(event(2)));

%concatenate various elements into fin and fout
fin = [fin1, fin2];

