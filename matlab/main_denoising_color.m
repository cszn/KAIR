


input_folder    = 'denoising_color';
output_folder   = 'denoising_color_results';

upperleft_pixel = [220, 5];
box             = [60, 60];
zoomfactor      = 3;
zoom_position   = 'lr';
nline = 2;

ext             = {'*.jpg','*.png','*.bmp'};

images          = [];

for i = 1:length(ext)
    
    images = [images dir(fullfile(input_folder, ext{i}))];
    
end

if isdir(output_folder) == 0
    mkdir(output_folder);
end

for i = 1:numel(images)
    
    [~, name, exte] = fileparts(images(i).name);
    I   =   imread( fullfile(input_folder,images(i).name));

    % if i == 1
    %     imtool(double(I)/256)
    % end

    I   =   zoom_function(I, upperleft_pixel, box, zoomfactor, zoom_position,nline);
    
    imwrite(I, fullfile(output_folder,images(i).name), 'Compression','none');
    
    imshow(I)
    title(name);
    
    pause(1)
    
end

close;





