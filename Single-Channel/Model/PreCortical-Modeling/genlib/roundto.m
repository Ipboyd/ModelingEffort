function x = roundto(x, grade, shift)

if nargin < 2
    grade = 1;
end
if nargin < 3
    shift = 0;
end

x = grade*round((x - shift)/grade) + shift;