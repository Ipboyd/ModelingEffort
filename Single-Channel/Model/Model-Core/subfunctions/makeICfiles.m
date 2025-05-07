function makeICfiles

ICfiles = struct;

% masker-only: 1-4
songloc = zeros(1,4);
maskloc = 1:4;

for i = 1:4
    songloc = [songloc , ones(1,5)*i];
    maskloc = [maskloc , 0:4];
end
for i = 1:24
    ICfiles(i,1).name = sprintf('s%im%i.mat',songloc(i),maskloc(i));
end
save('ICfiles.mat','ICfiles');