function out = isDisp(ind,Ntot,Nup)
% function out = isdisp(ind,Ntot,Nup)
%   OUT = ISDISP(IND,NTOT,NUP) is used to determine if it's time to update
%   the user on the progress of a function. Given a for loop on iteration 
%   IND of NTOT total operations where you want to give Nup updates, 
%   ISDISP will return 1 if it's time to update and 0 otherwise.
%   
%   Note that ISDISP never returns 1 if Nup is zero, and always displays on
%   the first operation, so you may end up with Nup+1 updates.
%
out = (mod(ind,round(Ntot/Nup))==0 || ind == 1) && Nup;