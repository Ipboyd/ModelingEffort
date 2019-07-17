function y = circconv(x1, x2)
% Performs circular convolution between the two 1-D inputs.

len = max(length(x1), length(x2));

if size(x1, 1) == 1 && size(x2, 1) == 1
    y = real(ifft(fft(x1, len).*fft(x2, len)));
else
    y = real(ifft(fft(x1(:), len).*fft(x2(:), len)));
end