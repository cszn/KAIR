


input_folder    =  'denoising_gray';
output_folder   =  'denoising_gray_results';

upperleft_pixel =  [172, 218];
box             =  [35, 35];
zoomfactor      =  3;
zoom_position   =  'ur';
nline           =  2;

ext             = {'*.jpg','*.png','*.bmp'};

images          = [];
for i = 1:length(ext)
    images = [images, dir(fullfile(input_folder, ext{i}))];
end

if isfolder(output_folder) == 0
    mkdir(output_folder);
end

for i = 1:numel(images)
    
    [~, name, exte] = fileparts(images(i).name);
    I  =  imread( fullfile(input_folder,images(i).name));
    
%     if i == 1
%         imtool(double(I)/256)
%     end
    
    I   =   zoom_function(I, upperleft_pixel, box, zoomfactor, zoom_position,nline);
    
    imwrite(I, fullfile(output_folder,images(i).name), 'Compression','none');
    
    imshow(I)
    title(name);
    pause(1)
    
    
end









