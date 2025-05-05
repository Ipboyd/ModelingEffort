function done(scale, email)
% DONE(LEVEL, EMAIL)
%   Plays three nice, soft (-18 dB default) dings to let you know the job
%   is done. Can input dB down (LEVEL) if desired. EMAIL is an e-mail
%   address to send to saying "Your MATLAB job finsished at [date/time]."
%   SENDMAIL preferences must have been set prior to use.

if ~exist('scale', 'var')
	scale = 8;
else
	scale = 10^(abs(scale/20));
end

fs = 3000;
t = 0:1/fs:.5;
env = exp(-t*10);
car = sin(2*pi*800*t) + sin(2*pi*1000*t) + sin(2*pi*1200*t);
ding = car.*env/3/scale;
sound(repmat(ding, 1, 3),fs);

if ~strcmpi(getpref('Internet','E_mail'), 'matlab.alert@gmail.com')
	setpref('Internet','E_mail','matlab.alert@gmail.com');
	setpref('Internet','SMTP_Server','smtp.gmail.com');
	setpref('Internet','SMTP_Username','matlab.alert@gmail.com');
	setpref('Internet','SMTP_Password','zfinch421');
	props = java.lang.System.getProperties;
	props.setProperty('mail.smtp.auth','true');
	props.setProperty('mail.smtp.socketFactory.class', 'javax.net.ssl.SSLSocketFactory');
	props.setProperty('mail.smtp.socketFactory.port','465');
end

try
	if exist('email', 'var')
		sendmail(email, '', sprintf('Your MATLAB job finished on %s.', datestr(now)))
	end
catch
	warning('There was an error with SENDMAIL. No e-mail was sent.')
end