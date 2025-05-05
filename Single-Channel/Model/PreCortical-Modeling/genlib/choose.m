function x = choose(n, k)
% Shorthand wrapper function for NCHOOSEK

x = nchoosek(n,k);%factorial(n(:))./(factorial(k(:)).*factorial(n(:) - k(:)))';