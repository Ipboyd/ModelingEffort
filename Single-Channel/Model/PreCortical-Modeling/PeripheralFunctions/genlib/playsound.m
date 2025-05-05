% Saves audio data to a wav file before playback in terminal.
%
% audio - audio samples
% fs - sampling rate (Hz) [default: 8000]
% nbits - precision (bits) [default: 16]
% wavfile - filename for the wav file [default: '/tmp/tmp.wav']
%
% usage: playsound(audio, fs, nbits, wavfile)

function playsound(audio, fs, nbits, wavfile)

% parse input arguments or use defaults
switch(nargin)
	case 1, wavfile='/tmp/tmp.wav'; nbits=16; fs=8000;
	case 2, wavfile='/tmp/tmp.wav'; nbits=16;
	case 3, wavfile='/tmp/tmp.wav';
	case 4,
	otherwise, error(sprintf('usage: play(audio, 8000, 16, ''/tmp/tmp.wav'')\ntype: help playsound for further details.'));
end

% save audio to file, scale prior to writing to avoid clipping
% audio = 0.999*audio/max(abs(audio));
wavwrite(audio, fs, nbits, wavfile);

% playback the audio
if(exist(wavfile, 'file'))
	syscmd = sprintf('mplayer -really-quiet %s', wavfile); % uses mplayer for playback
	system(syscmd);
else
	error(sprintf('%s: file not found.', wavfile));
end
