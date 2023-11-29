function levelSignal = sound2dB(ampSignal)
% levelSignal = sound2dB_Sp(ampSignal,dBSPL)
%  
%  In vocoded speech sessions, sound signal was saved after calibration 
%  macro. Simply convert to dB scale. 
%
%  KP, 2019-12

levelSignal = real( log10( ampSignal ) * 20 );

end
