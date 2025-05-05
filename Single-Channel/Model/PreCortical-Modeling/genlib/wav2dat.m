function wav2dat(wavfn, datfn)
% wav2dat(wavfn, datfn)

[x, fs] = wavread(wavfn);
x = resample(x, 32e3, fs);
datwrite(x, datfn);