% Notice:
%   This file is provided by Verasonics to end users as a programming
%   example for the Verasonics Vantage Research Ultrasound System.
%   Verasonics makes no claims as to the functionality or intended
%   application of this program and the user assumes all responsibility
%   for its use.
%
% File name: SetUpP4_1FlashAngles.m - Example of phased array flash imaging
%                                       with multiple steering angles
% Description:
%   Sequence programming file for P4-1 phased array in virtual apex format,
%   using flash tranmists with multiple steering angles. All 96 transmit and
%   receive channels are active for each acquisition. The curvature of the
%   transmit pulse is set to match a circle with the same P.radius as the
%   distance to the virtual apex. Processing is asynchronous with respect
%   to acquisition. Note: The P4-1 is a 96 element probe that is wired to
%   the scanhead connector with element 0-31 connected to inputs 1-32, and
%   elements 32-63 connected to input 97-128. We therefore need a
%   Trans.Connector array to specify the connector channels used, which
%   will be defined by the computeTrans function.
%
% Last update:
% 12/13/2015 - modified for SW 3.0

clear all
clear transferPperFrame
clear saveIQperFrame
clear savePperFrame

% ytSetUp_ethernet_transfer_server_setup

P.startDepth = 0;
P.endDepth = 280;   % Acquisition depth in wavelengths

na = 21;      % Number of angles
if na > 1
    dtheta = (60*pi/180)/(na-1); startAngle = -60*pi/180/2;  % set dtheta to range over +/- 30 degrees.
else
    dtheta = 0; startAngle = 0;
end

% Specify system parameters.
Resource.Parameters.numTransmit = 128;  % number of transmit channels.
Resource.Parameters.numRcvChannels = 128;  % number of receive channels.
Resource.Parameters.speedOfSound = 1540;
Resource.Parameters.speedCorrectionFactor = 1.0;
Resource.Parameters.verbose = 2;
Resource.Parameters.initializeOnly = 0;
Resource.Parameters.simulateMode = 0;
%  Resource.Parameters.simulateMode = 1 forces simulate mode, even if hardware is present.
%  Resource.Parameters.simulateMode = 2 stops sequence and processes RcvData continuously.

% Specify Trans structure array.
Trans.name = 'P4-1';
Trans.units = 'wavelengths'; % Explicit declaration avoids warning message when selected by default
Trans = computeTrans(Trans);
Trans.maxHighVoltage = 50;  % set maximum high voltage limit for pulser supply.
Trans.name = 'custom'; % TEMP BEFORE FIXING VERASONICS

P.theta = -pi/4;
P.rayDelta = 2*(-P.theta);
P.aperture = Trans.numelements*Trans.spacing; % P.aperture in wavelengths
P.radius = (P.aperture/2)/tan(-P.theta); % dist. to virt. apex

% Set up PData structure.
PData(1).PDelta = [0.875, 0, 0.5];
PData(1).Size(1) = 10 + ceil((P.endDepth-P.startDepth)/PData(1).PDelta(3));
PData(1).Size(2) = 10 + ceil(2*(P.endDepth + P.radius)*sin(-P.theta)/PData(1).PDelta(1));
PData(1).Size(3) = 1;
PData(1).Origin = [-(PData(1).Size(2)/2)*PData(1).PDelta(1),0,P.startDepth];
PData(1).Region = struct(...
            'Shape',struct('Name','SectorFT', ...
            'Position',[0,0,-P.radius], ...
            'z',P.startDepth, ...
            'r',P.radius+P.endDepth, ...
            'angle',P.rayDelta, ...
            'steer',0));
PData(1).Region = computeRegions(PData(1));

% Specify Media.  Use point targets in middle of PData.
% Set up Media points
% - Uncomment for speckle
% Media.MP = rand(40000,4);
% Media.MP(:,2) = 0;
% Media.MP(:,4) = 0.04*Media.MP(:,4) + 0.04;  % Random amplitude
% Media.MP(:,1) = 2*halfwidth*(Media.MP(:,1)-0.5);
% Media.MP(:,3) = P.acqDepth*Media.MP(:,3);
Media.MP(1,:) = [-45,0,30,1.0];
Media.MP(2,:) = [-15,0,30,1.0];
Media.MP(3,:) = [15,0,30,1.0];
Media.MP(4,:) = [45,0,30,1.0];
Media.MP(5,:) = [-15,0,60,1.0];
Media.MP(6,:) = [-15,0,90,1.0];
Media.MP(7,:) = [-15,0,120,1.0];
Media.MP(8,:) = [-15,0,150,1.0];
Media.MP(9,:) = [-45,0,120,1.0];
Media.MP(10,:) = [15,0,120,1.0];
Media.MP(11,:) = [45,0,120,1.0];
Media.MP(12,:) = [-10,0,69,1.0];
Media.MP(13,:) = [-5,0,75,1.0];
Media.MP(14,:) = [0,0,78,1.0];
Media.MP(15,:) = [5,0,80,1.0];
Media.MP(16,:) = [10,0,81,1.0];
Media.MP(17,:) = [-75,0,120,1.0];
Media.MP(18,:) = [75,0,120,1.0];
Media.MP(19,:) = [-15,0,180,1.0];
Media.numPoints = 19;
Media.attenuation = -0.5;
Media.function = 'movePoints';

% Specify Resources.
Resource.RcvBuffer(1).datatype = 'int16';
Resource.RcvBuffer(1).rowsPerFrame = na*4096;
Resource.RcvBuffer(1).colsPerFrame = Resource.Parameters.numRcvChannels;
Resource.RcvBuffer(1).numFrames = 30;     % 30 frames used for RF cineloop.
Resource.InterBuffer(1).numFrames = 1;    % one intermediate buffer defined but not used.
Resource.ImageBuffer(1).numFrames = 1; %not 1 creates lag
Resource.DisplayWindow(1).Title = 'P4-1FlashAngles';
Resource.DisplayWindow(1).pdelta = 0.35;
ScrnSize = get(0,'ScreenSize');
DwWidth = ceil(PData(1).Size(2)*PData(1).PDelta(1)/Resource.DisplayWindow(1).pdelta);
DwHeight = ceil(PData(1).Size(1)*PData(1).PDelta(3)/Resource.DisplayWindow(1).pdelta);
Resource.DisplayWindow(1).Position = [250,(ScrnSize(4)-(DwHeight+150))/2, ...  % lower left corner position
                                      DwWidth, DwHeight];
Resource.DisplayWindow(1).ReferencePt = [PData(1).Origin(1),0,PData(1).Origin(3)];   % 2D imaging is in the X,Z plane
Resource.DisplayWindow(1).Type = 'Verasonics';
Resource.DisplayWindow(1).numFrames = 20;
Resource.DisplayWindow(1).AxesUnits = 'mm';
Resource.DisplayWindow.Colormap = gray(256);

% Specify TW structure array.
% Specify Transmit waveform structure.  These structures are persistent and we
%   only need to specify what changes in subsequent structures.
TW.type = 'parametric';
TW.Parameters = [Trans.frequency,.67,2,1];

% Specify TX structure array.
TX = repmat(struct('waveform', 1, ...
                   'Origin', [0.0,0.0,0.0], ...
                   'focus', -P.radius, ...
                   'Steer', [0.0,0.0], ...
                   'Apod', ones(1,Trans.numelements), ...  % set TX.Apod for 96 elements
                   'Delay', zeros(1,Trans.numelements)), 1, na);
% - Set event specific TX attributes.
for n = 1:na   % na transmit events
    TX(n).Steer = [(startAngle+(n-1)*dtheta),0.0];
    TX(n).Delay = computeTXDelays(TX(n));
end

% Specify Receive structure arrays.
maxAcqLength = ceil(sqrt(P.aperture^2 + P.endDepth^2 - 2*P.aperture*P.endDepth*cos(P.theta-pi/2)) - P.startDepth);
wlsPer128 = 128/(4*2); % wavelengths in 128 samples for 4 samplesPerWave
Receive = repmat(struct('Apod', ones(1,Trans.numelements), ...
                        'startDepth', P.startDepth, ...
                        'endDepth', P.startDepth + wlsPer128*ceil(maxAcqLength/wlsPer128), ...
                        'TGC', 1, ...
                        'bufnum', 1, ...
                        'framenum', 1, ...
                        'acqNum', 1, ...
                        'sampleMode', 'NS200BW', ...
                        'mode', 0, ...
                        'callMediaFunc', 0),1,na*Resource.RcvBuffer(1).numFrames);
% - Set event specific Receive attributes.
for i = 1:Resource.RcvBuffer(1).numFrames
    Receive(na*(i-1)+1).callMediaFunc = 1;
    for j = 1:na
        Receive(na*(i-1)+j).framenum = i;
        Receive(na*(i-1)+j).acqNum = j;
    end
end

% Specify TGC Waveform structure.
TGC.CntrlPts = [91,590,651,710,820,931,992,1023];
TGC.rangeMax = P.endDepth;
TGC.Waveform = computeTGCWaveform(TGC);

% Specify Recon structure arrays.
Recon = struct('senscutoff', 0.45, ...
               'pdatanum', 1, ...
               'rcvBufFrame', -1, ...
               'IntBufDest', [1,1], ...
               'ImgBufDest', [1,-1], ...
               'RINums', 1:na);

% Define ReconInfo structures.
% We need na ReconInfo structures for na steering angles.
ReconInfo = repmat(struct('mode', 4, ...  % accumulate IQ data.
                   'txnum', 1, ...
                   'rcvnum', 1, ...
                   'regionnum', 1), 1, na);
% - Set specific ReconInfo attributes.
if na>1
    ReconInfo(1).mode = 'replaceIQ';
    for j = 1:na  % For each row in the column
        ReconInfo(j).txnum = j;
        ReconInfo(j).rcvnum = j;
    end
    ReconInfo(na).mode = 'accumIQ_replaceIntensity';  % accumulate and detect
else
    ReconInfo(1).mode = 'replaceIntensity';
end

% Specify Process structure array.
pers = 20;
cmpFactor = 40;
Process(1).classname = 'Image';
Process(1).method = 'imageDisplay';
Process(1).Parameters = {'imgbufnum',1,...   % number of buffer to process.
                         'framenum',-1,...   % (-1 => lastFrame)
                         'pdatanum',1,...    % number of PData structure to use
                         'pgain',1.0,...     % pgain is image processing gain
                         'reject',2,...
                         'grainRemoval','none',...
                         'persistMethod','none',...
                         'persistLevel',pers,...
                         'interpMethod','4pt',...
                         'processMethod','none',...
                         'averageMethod','none',...
                         'compressMethod','power',...
                         'compressFactor',cmpFactor,...
                         'mappingMethod','full',...
                         'display',1,...      % display image after processing
                         'displayWindow',1};

%Save Bmode Data Process
Process(2).classname = 'External';
Process(2).method = 'transferPperFrame'; % calls the 'saveRFperFrame' function 'saveIQperFrame'
Process(2).Parameters = {'srcbuffer','image',... % name of buffer to process. % 'inter'
    'srcbufnum',1,...
    'srcframenum',1,... % process the most recent frame.
    'dstbuffer','none'};
% EF(2).Function = text2cell('%EF#1'); % '%EF#1'

% Specify SeqControl structure arrays.  Missing fields are set to NULL.
SeqControl(1).command = 'jump'; % jump back to start
SeqControl(1).argument = 1;
SeqControl(2).command = 'timeToNextAcq';  % time between each transmit
SeqControl(2).argument = 400;  % 400 us
SeqControl(3).command = 'timeToNextAcq';  % time between frames
SeqControl(3).argument = 20000 - (na-1)*400;  % 20 msec
SeqControl(4).command = 'returnToMatlab';
nsc = 5;

% Specify Event structure arrays.
n = 1;
for i = 1:Resource.RcvBuffer(1).numFrames
    for j = 1:na                 % Acquire frame
        Event(n).info = 'Acquire full aperture.';
        Event(n).tx = j;
        Event(n).rcv = na*(i-1)+j;
        Event(n).recon = 0;
        Event(n).process = 0;
        Event(n).seqControl = 2;
        n = n+1;
    end
    Event(n-1).seqControl = [3,nsc]; % modify last event's seqCntrl: time between frames & transferToHostuse
       SeqControl(nsc).command = 'transferToHost';
       nsc = nsc + 1;

    Event(n).info = 'recon and process';
    Event(n).tx = 0;
    Event(n).rcv = 0;
    Event(n).recon = 1;
    Event(n).process = 1;
    Event(n).seqControl = 0;
    if floor(i/3) == i/3     % Exit to Matlab every 3rd frame
        Event(n).seqControl = 4;
    end
    n = n+1;


    % SAVES IMAGE NEED TO UNCOM WHEN RUNNING!!
    Event(n).info = 'Save bmode data';
    Event(n).tx = 0;
    Event(n).rcv = 0;
    Event(n).recon = 0;
    Event(n).process = 2;
    Event(n).seqControl = 0;
    n=n+1;

end

Event(n).info = 'Jump back';
Event(n).tx = 0;
Event(n).rcv = 0;
Event(n).recon = 0;
Event(n).process = 0;
Event(n).seqControl = 1;


% User specified UI Control Elements
% - Sensitivity Cutoff
UI(1).Control =  {'UserB7','Style','VsSlider','Label','Sens. Cutoff',...
                  'SliderMinMaxVal',[0,1.0,Recon(1).senscutoff],...
                  'SliderStep',[0.025,0.1],'ValueFormat','%1.3f'};
UI(1).Callback = text2cell('%SensCutoffCallback');

% - Range Change
MinMaxVal = [64,300,P.endDepth]; % default unit is wavelength
AxesUnit = 'wls';
if isfield(Resource.DisplayWindow(1),'AxesUnits')&&~isempty(Resource.DisplayWindow(1).AxesUnits)
    if strcmp(Resource.DisplayWindow(1).AxesUnits,'mm');
        AxesUnit = 'mm';
        MinMaxVal = MinMaxVal * (Resource.Parameters.speedOfSound/1000/Trans.frequency);
    end
end
UI(2).Control = {'UserA1','Style','VsSlider','Label',['Range (',AxesUnit,')'],...
                 'SliderMinMaxVal',MinMaxVal,'SliderStep',[0.1,0.2],'ValueFormat','%3.0f'};
UI(2).Callback = text2cell('%RangeChangeCallback');

% Specify factor for converting sequenceRate to frameRate.
frameRateFactor = 3;

EF(1).Function = ...
    vsv.seq.function.ExFunctionDef('savePperFrame',@savePperFrame);

EF(2).Function = ...
    vsv.seq.function.ExFunctionDef('saveIQperFrame',@saveIQperFrame);

EF(3).Function = ...
    vsv.seq.function.ExFunctionDef('transferPperFrame',@transferPperFrame);

% Save all the structures to a .mat file.
save('MatFiles/P4-1FlashAngles');
% save('MatFiles/P4-1FlashAngles','-regexp','^(?!(server)$).');
filename  = ('MatFiles/P4-1FlashAngles');

yt_ROSinit
VSX;
return


% **** Callback routines to be converted by text2cell function. ****
%SensCutoffCallback - Sensitivity cutoff change
ReconL = evalin('base', 'Recon');
for i = 1:size(ReconL,2)
    ReconL(i).senscutoff = UIValue;
end
assignin('base','Recon',ReconL);
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'Recon'};
assignin('base','Control', Control);
return
%SensCutoffCallback

%RangeChangeCallback - Range change
simMode = evalin('base','Resource.Parameters.simulateMode');
% No range change if in simulate mode 2.
if simMode == 2
    set(hObject,'Value',evalin('base','P.endDepth'));
    return
end
Trans = evalin('base','Trans');
Resource = evalin('base','Resource');
scaleToWvl = Trans.frequency/(Resource.Parameters.speedOfSound/1000);

P = evalin('base','P');
P.endDepth = UIValue;
if isfield(Resource.DisplayWindow(1),'AxesUnits')&&~isempty(Resource.DisplayWindow(1).AxesUnits)
    if strcmp(Resource.DisplayWindow(1).AxesUnits,'mm');
        P.endDepth = UIValue*scaleToWvl;
    end
end
assignin('base','P',P);

PData = evalin('base','PData');
PData(1).Size(1) = 10 + ceil((P.endDepth-P.startDepth)/PData(1).PDelta(3));
PData(1).Region = struct(...
            'Shape',struct('Name','SectorFT', ...
            'Position',[0,0,-P.radius], ...
            'z',P.startDepth, ...
            'r',P.radius+P.endDepth, ...
            'angle',P.rayDelta, ...
            'steer',0));
PData(1).Region = computeRegions(PData(1));
assignin('base','PData',PData);

evalin('base','Resource.DisplayWindow(1).Position(4) = ceil(PData(1).Size(1)*PData(1).PDelta(3)/Resource.DisplayWindow(1).pdelta);');
Receive = evalin('base', 'Receive');
maxAcqLength = ceil(sqrt(P.aperture^2 + P.endDepth^2 - 2*P.aperture*P.endDepth*cos(P.theta-pi/2)) - P.startDepth);
wlsPer128 = 128/(4*2);
for i = 1:size(Receive,2)
    Receive(i).endDepth = P.startDepth + wlsPer128*ceil(maxAcqLength/wlsPer128);
end
assignin('base','Receive',Receive);
evalin('base','TGC.rangeMax = P.endDepth;');
evalin('base','TGC.Waveform = computeTGCWaveform(TGC);');
Control = evalin('base','Control');
Control.Command = 'update&Run';
Control.Parameters = {'PData','InterBuffer','ImageBuffer','DisplayWindow','Receive','TGC','Recon'};
assignin('base','Control', Control);
assignin('base', 'action', 'displayChange');
return
%RangeChangeCallback

% External functions

% function savePperFrame(PData)
% 
% persistent Pfilename AcqCountP
% P_frame_limit = 10;
% 
% % file size can be reduced by elimating all zeros
% % TXApod = evalin('base','TX.Apod');
% % endSample = evalin('base','Receive(end).endSample');
% 
% if isempty(AcqCountP)
%     AcqCountP = 1;
% else
%     AcqCountP = AcqCountP + 1;
% end
% 
% if AcqCountP <= P_frame_limit
%     Pfilename = ['tmp_P4_1\Pdata_acquisition',num2str(AcqCountP)];
% 
%     p_data = PData;
% 
%     tic
%     save(Pfilename,'p_data','-v6');
%     fprintf('saving time for frame %g = %g s \n',AcqCountP, toc)
% end
% 
% end


function savePperFrame(PData)

persistent AcqCountP TagCountP

% file size can be reduced by elimating all zeros
% TXApod = evalin('base','TX.Apod');
% endSample = evalin('base','Receive(end).endSample');

num_Tag_Group = 500;

if isempty(AcqCountP)
    AcqCountP = 1;
else
    AcqCountP = AcqCountP + 1;
end
Pfilename = ['tmp_P4_1\Pdata_acquisition',num2str(AcqCountP),'.mat'];

if isempty(TagCountP)
    TagCountP = 0;
end

p_data = PData;

time = clock;
name_time = ['time_tag_acq_',num2str(AcqCountP)];
eval([name_time,' = time;']);

if rem(AcqCountP,num_Tag_Group) == 1
    TagCountP = TagCountP + 1;
    Tagfilename = ['tmp_P4_1\time_tag_acq_group_',num2str(TagCountP),'.mat'];
    save(Tagfilename,name_time,'-v6');
else
    Tagfilename = ['tmp_P4_1\time_tag_acq_group_',num2str(TagCountP),'.mat'];
    save(Tagfilename,name_time,'-v6','-append');
end

tic
save(Pfilename,'p_data','-v6');
fprintf('saving time for frame %g = %g s \n',AcqCountP, toc)

end


function saveIQperFrame(IData,QData)

persistent AcqCountIQ
IQ_frame_limit = 10;

if isempty(AcqCountIQ)
    AcqCountIQ = 1;
else
    AcqCountIQ = AcqCountIQ + 1;
end

if AcqCountIQ <= IQ_frame_limit
    Ifilename = ['tmp_P4_1\Idata_acquisition',num2str(AcqCountIQ)];
    Qfilename = ['tmp_P4_1\Qdata_acquisition',num2str(AcqCountIQ)];
    i_data = IData;
    q_data = QData;
    
    tic
    save(Ifilename,'i_data','-v6');
    save(Qfilename,'q_data','-v6');
    fprintf('saving time for frame %g = %g s \n',AcqCountIQ, toc)
end
    
end

function transferPperFrame(PData)

% global server
persistent TransCount

if isempty(TransCount)
    TransCount = 1;
else
    TransCount = TransCount + 1;
end

p_data = PData;
p_data = rescale(p_data, 0, 255);

p_data_trans = reshape(p_data,[1,570*500]);

% fwrite(server,p_data_trans,'uint8');
disp(['image data transfered.','#',num2str(TransCount)]);

yt_ROSsync

end
