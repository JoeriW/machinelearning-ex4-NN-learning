function [y] = yVectorize(lngth,pos)

y  = zeros(lngth,1);
y(pos) = 1;


end