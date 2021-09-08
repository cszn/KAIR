function [im] = center_replace(im,im2)

[w,h,~] = size(im);

[a,b,~] = size(im2);
c1 = w-a-(w-a)/2;
c2 = h-b-(h-b)/2;
im(c1+1:c1+a,c2+1:c2+b,:) = im2;

end

