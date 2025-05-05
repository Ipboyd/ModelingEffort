% Saves audio data to a wav file before playback.
%
% audio - audio samples
% fs - sampling rate (Hz) [default: 8000]
% nbits - precision (bits) [default: 16]
% wavfile - filename for the wav file [default: '/tmp/tmp.wav']
%
% usage: playsound(audio, fs, nbits, wavfile)

function playsoundsc(varargin)

varargin{1} = 0.999*varargin{1}/max(abs(varargin{1}(:)));
playsound(varargin{:})