function gSYN = buildgSYN(N_pre,N_post,chan1,chan2,chan3,chan4)

gSYN = zeros(N_pre,N_post);
if N_pre ~= N_post
    gSYN(1) = chan1;
    gSYN(2) = chan2;
    gSYN(3) = chan3;
    gSYN(4) = chan4;
else
    gSYN(1,1) = chan1;
    gSYN(2,2) = chan2;
    gSYN(3,3) = chan3;
    gSYN(4,4) = chan4;
end

end