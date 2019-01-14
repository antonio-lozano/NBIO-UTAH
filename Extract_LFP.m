%%%%%%%%%%%%%%%%%%%%%%%%%

%      Extracts an LFP file:
%
% This function extracts loads the information in a .Ns6 file into a matlab
% matrix in witch each line is the meassurement of an electrode
%
% It needs the functions openNEN and openNsx to be in any subfoulder.
%
% It also desamples the meassurement for a factor 'down'. It takes a
% predifined value of 100 if not given any.
%
% It reordes the electrodes given the file 'Electrodes_Order.mat' and
% includes the trigger saved in the .NEV file in the last line.
%
% The resulting file can be loaded into braimstorm.

%%%%%%%%%%%%%%%%%%%%%%%%%

function data = Extract_LFP(a,b)
%% Conditions in the arguments of the function:

addpath (genpath (pwd))

switch nargin
    case 2
        if ischar(a) && isnumeric(b)
            file_name = a;
            down = b;
        elseif ischar(b) && isnumeric(a)
            file_name = b;
            down = a;
        else
            error ('invalid arguments')
        end           
    case 1
        if ischar(a)
            file_name = a;
            down = 100;
        elseif isnumeric(a)
            file_name = strsplit(uigetfile({'*.ns6'}),'.');
            file_name = char(file_name(1));
            down = a;
        else          
            error ('invalid arguments')
        end
    case 0
        file_name = strsplit(uigetfile({'*.ns6'}),'.');
        file_name = char(file_name(1));
        down = 100;
    otherwise
            error ('too many arguments')
end

%% load Data:

% use openNSx toopen the ns6 file
NS6=openNSx(strcat(file_name,'.ns6'));

% Save register data in a matrix
data = NS6.Data(1:96,:)./4;
length_O=size(data,2);
clear NS6


%% Reduce sample rate if needed

% Downsamples the matrix first to make the processinf lighter
if down ~= 1    
    data = downsample(data',down)';
end

% pasa la señal a microvoltios
data = 1000*double(data); % This is done after the downssmple to avoid the
% overload of changing the format todouble precission. this format isneeded
% to open the file in brainstorm

%% Reorder chanels:

% Extracts electrodes order from a .mat file
load ('Electrodes_Order.mat');

% Reorders electrode matrix:

index = 1:96;
data(order,:) = data(index,:); clear index

%% Adds trigger:

trig = 1; % Canavoid the addition of the trigger line if needed

if trig == 1
    
    % Loads trigger times from NEV File
    NEV=openNEV(strcat(file_name,'.nev'),'nosave','nomat'); %Loads NEV file
    trigger = NEV.Data.SerialDigitalIO.TimeStamp;
    clear NEV % Clearsnev file to prevent overload
    
    % Creates a channel vector with 1's in the trigger times
    channel_t= double(zeros(1,length_O));
    for i= 1:length(trigger)
        channel_t(trigger(i))= 1;
    end
    
    % Downsamples the trigger leaving 1´s unchanged
    if down ~= 1
        channel_t1 = [];
        for i = 1: ceil(length(channel_t)/down)-1
        channel_t1(i) = sum(channel_t(10*i-9:10*i));
        end
        channel_t1(ceil(length(channel_t)/down))=0;
        channel_t=channel_t1; clear channel_t1
    end
    
    % Adds Trigger channel to the electrodes matrix 
    data=[data;channel_t];    
end

%% Save to mat file

save(file_name,'data')
