function setfiguredefaults(journal)

% 'nn' is Nature Neurosci
% 'pb' is PLoS Bio
% 'sci' is Science
% 'jn' will be JNeurosci
% 'jnp' is JNeurophys
% 'neuron' is Neuron
% 'jasaWilly' is JASA for the Willy experiments
% 'rkm' is RKM's preferred settings

if ~exist('journal', 'var')
	journal = computer;
end

% if you add a property, make sure to put it here first, then at the bottom, so
% it doesn't break the journals you don't add it to.
fontname = get(0, 'DefaultAxesFontName');
fontsize = get(0, 'DefaultAxesFontSize');
colororder = get(0, 'DefaultAxesColorOrder');
box = get(0, 'DefaultAxesBox');
markersize = get(0, 'DefaultLineMarkerSize');
figurecolor = get(0, 'DefaultFigureColor');
linelinewidth = get(0, 'DefaultLineLineWidth');
axeslinewidth = get(0, 'DefaultAxesLineWidth');
patchlinewidth = get(0, 'DefaultPatchLineWidth');

% make some default color orders
boldcolors = min(1, 1.2*hex2rgb({'002099';'4d9900';'990000';'009999';'4d0099';'d69900';'404040'}));
eastercolors = min(1, hsv(6) + 0.35);
matlabdefault = [0 0 1; 0 0.5 0; 1 0 0; 0 0.75 0.75; 0.75 0 0.75; 0.75 0.75 0; 0.25 0.25 0.25];

switch lower(journal)
	case 'nn' % Nature Neuroscience
		fontname = 'Myriad Pro';
		fontsize = 9;
		colororder = boldcolors;
		markersize = 3;
		box = 'on';
		figurecolor = 'w';
		axeslinewidth = 0.5;
		linelinewidth = 1.0;
	case 'nn2' % Nature Neuroscience
		fontname = 'Arial'; % replace with Helvetica when you have it
		fontsize = 9;
		colororder = boldcolors;
		markersize = 3;
		box = 'on';
		figurecolor = 'w';
		axeslinewidth = 0.5;
		linelinewidth = 1.0;
	case 'pb' % PLoS Bio
		fontname = 'Arial';
		fontsize = 9;
		colororder = boldcolors;
		markersize = 3;
		box = 'off';
		figurecolor = 'w';
		axeslinewidth = 0.5;
		linelinewidth = 1.0;
    case 'pb_s' % PLoS Bio
        fontname = 'Arial';
        fontsize = 8;
        colororder = boldcolors;
        markersize = 3;
        box = 'off';
        figurecolor = 'w';
        axeslinewidth = 0.5;
        linelinewidth = 1.0;
	case 'sci' % Science
		fontname = 'Helvetica';
		fontsize = 7;
		colororder = boldcolors;
		markersize = 3;
		box = 'on';
		figurecolor = 'w';
		axeslinewidth = 0.5;
		linelinewidth = 1.0;
	case 'nn_nobox'
		fontname = 'Myriad Pro';
		fontsize = 9;
		colororder = boldcolors;
		markersize = 3;
		box = 'off';
		figurecolor = 'w';
		axeslinewidth = 0.5;
		linelinewidth = 1.0;
	case 'pvloc'
		fontname = 'Myriad Pro';
		fontsize = 9;
		colororder = [0 0 0];
		markersize = 2;
		box = 'off';
		figurecolor = 'w';
		axeslinewidth = 0.5;
		linelinewidth = 0.5;
	case 'jnp'
		fontname = 'Helvetica';
		fontsize = 9;
	case 'neuron' % Neuron
		fontname = 'Arial'; % replace with Helvetica when you have it
		fontsize = 9;
		colororder = boldcolors;
		markersize = 3;
		box = 'on';
		figurecolor = 'w';
		axeslinewidth = 0.5;
		linelinewidth = 1.0;
	case 'pres'
		fontname = 'Comic Sans MS';
		fontsize = 12;
		markersize = 10;
		linelinewidth = 2;
		axeslinewidth = 1.5;
	case 'presgordon'
		fontname = 'Droid Sans';
		fontsize = 20;
		markersize = 10;
		linelinewidth = 1.5;
		axeslinewidth = 1;
		patchlinewidth = 1.5;
	case 'presbash'
		fontname = 'Myriad Pro';
		fontsize = 16;
		markersize = 8;
		linelinewidth = 1.5;
		axeslinewidth = 1;
		patchlinewidth = 1.5;
	case 'labsnposter'
		fontname = 'Myriad Pro';
		fontsize = 24;
		markersize = 8;
		linelinewidth = 1.5;
		axeslinewidth = 1;
		patchlinewidth = 1.5;
	case 'jasa'
		fontname = 'Arial';
		fontsize = 8;
		colororder = repmat([0 0.45 0.63 0.8].', 1, 3);
		markersize = 3;
		box = 'off';
		figurecolor = 'w';
		axeslinewidth = 0.5;
		linelinewidth = 1.0;
		patchlinewidth = 1.0;
	case 'jasawilly'
		fontname = 'Arial';
		fontsize = 8;
		colororder = flipud(repmat([0 0.45 0.63 0.8].', 1, 3)); % starts light and gets dark
		markersize = 3;
		box = 'off';
		figurecolor = 'w';
		axeslinewidth = 0.5;
		linelinewidth = 1.0;
	case 'rkm'

	otherwise
		switch lower(computer)
			case 'pcwin' % Windows
				fontname = 'Calibri';
				fontsize = 10;
				colororder = boldcolors;
			case 'glnxa64' % Linux
				fontname = 'Myriad Pro';
				fontsize = 10;
				colororder = boldcolors;
		end
end

% set the parameters
set(0, 'DefaultAxesFontName', fontname)
set(0, 'DefaultTextFontName', fontname)
set(0, 'DefaultAxesFontSize', fontsize)
set(0, 'DefaultTextFontSize', fontsize)
set(0, 'DefaultAxesColorOrder', colororder)
set(0, 'DefaultLineLineWidth', 1)
set(0, 'DefaultLineMarkerSize', markersize);
set(0, 'DefaultAxesBox', box)
set(0, 'DefaultFigureColor', figurecolor);
set(0, 'DefaultLineLineWidth', linelinewidth);
set(0, 'DefaultAxesLineWidth', axeslinewidth);
set(0, 'DefaultPatchLineWidth', patchlinewidth);