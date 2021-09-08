function [I]=zoom_function(I,upperleft_pixel,box,zoomfactor,zoom_position,nline)

y       = upperleft_pixel(1);
x       = upperleft_pixel(2);
box1    = box(1);
box2    = box(2); %4

s_color = [0 255 0];
l_color = [255 0 0];



[~, ~, hw]  =  size( I );

if hw == 1
    I=repmat(I,[1,1,3]);
end

Imin = I(x:x+box1-1,y:y+box2-1,:);
I(x-nline:x+box1-1+nline,y-nline:y+box2-1+nline,1) = s_color(1);
I(x-nline:x+box1-1+nline,y-nline:y+box2-1+nline,2) = s_color(2);
I(x-nline:x+box1-1+nline,y-nline:y+box2-1+nline,3) = s_color(3);
I(x:x+box1-1,y:y+box2-1,:) = Imin;
Imax = imresize(Imin,zoomfactor,'nearest');

switch lower(zoom_position)
    case {'uper_left','ul'}
    
    I(1:2*nline+zoomfactor*box1,1:2*nline+zoomfactor*box2,1) = l_color(1);
    I(1:2*nline+zoomfactor*box1,1:2*nline+zoomfactor*box2,2) = l_color(2);
    I(1:2*nline+zoomfactor*box1,1:2*nline+zoomfactor*box2,3) = l_color(3);
    I(1+nline:zoomfactor*box1+nline,1+nline:zoomfactor*box2+nline,:) = Imax;
    
    case {'uper_right','ur'}
        
    I(1:2*nline+zoomfactor*box1,end-2*nline-zoomfactor*box2+1:end,1) = l_color(1);
    I(1:2*nline+zoomfactor*box1,end-2*nline-zoomfactor*box2+1:end,2) = l_color(2);
    I(1:2*nline+zoomfactor*box1,end-2*nline-zoomfactor*box2+1:end,3) = l_color(3);
    I(1+nline:zoomfactor*box1+nline,end-nline-zoomfactor*box2+1:end-nline,:) = Imax;      

    case {'lower_left','ll'}
        
    I(end-2*nline-zoomfactor*box1+1:end,1:2*nline+zoomfactor*box2,1) = l_color(1);
    I(end-2*nline-zoomfactor*box1+1:end,1:2*nline+zoomfactor*box2,2) = l_color(2);
    I(end-2*nline-zoomfactor*box1+1:end,1:2*nline+zoomfactor*box2,3) = l_color(3);
    I(end-nline-zoomfactor*box1+1:end-nline,1+nline:zoomfactor*box2+nline,:) = Imax;
    
    case {'lower_right','lr'}
       
    I(end-2*nline-zoomfactor*box1+1:end,end-2*nline-zoomfactor*box2+1:end,1) = l_color(1);
    I(end-2*nline-zoomfactor*box1+1:end,end-2*nline-zoomfactor*box2+1:end,2) = l_color(2);
    I(end-2*nline-zoomfactor*box1+1:end,end-2*nline-zoomfactor*box2+1:end,3) = l_color(3);
    I(end-nline-zoomfactor*box1+1:end-nline,end-nline-zoomfactor*box2+1:end-nline,:) = Imax;    
        
        
               
end



