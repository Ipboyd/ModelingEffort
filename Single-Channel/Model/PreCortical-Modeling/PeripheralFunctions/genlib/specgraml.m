function [y,f,t] = specgraml(varargin)
%SPECGRAML Spectrogram using a Short-Time Fourier Transform (STFT).
%   SPECGRAM has been replaced by SPECTROGRAM.  SPECGRAM still works but
%   may be removed in the future. Use SPECTROGRAM instead. Type help
%   SPECTROGRAM for details.
%
%   See also PERIODOGRAM, SPECTRUM/PERIODOGRAM, PWELCH, SPECTRUM/WELCH, GOERTZEL.

%   Author(s): L. Shure, 1-1-91
%              T. Krauss, 4-2-93, updated
%   Copyright 1988-2004 The MathWorks, Inc.
%   $Revision: 1.8.4.4 $  $Date: 2004/11/01 19:29:55 $

[y,f,t] = specgram(varargin{:});

if nargout == 0
    newplot;
    if length(t)==1
        pcolor([0 1/f(2)],f,20*log10(abs(y)+eps));axis xy; colormap(jet);
    else
        pcolor(t,f,20*log10(abs(y)+eps));axis xy; colormap(jet);
    end
    shading interp
    f_inds = [1 2 4 8 16 32 63 125 250 500];
    set(gca,'yscale','log','ytick',[f_inds/1e6 f_inds/1e3 f_inds f_inds*1e3 f_inds*1e6 f_inds*1e9 f_inds*1e12]);
    xlabel('Time')
    ylabel('Frequency')
end
