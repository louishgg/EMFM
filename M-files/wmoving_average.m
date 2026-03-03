function trend=wmoving_average(data,window)

l=window; 
w = [1/(l*4);repmat(1/(l*2),l*2-1,1);1/(l*4)];
trend = conv(data,w,'same');

end