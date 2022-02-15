function generate_LR_Vimeo90K()
%% matlab code to genetate blur-downsampled (BD) for Vimeo90K dataset
% Euler module load matlab/R2020a; cd scripts/matlab_scripts; matlab -nodisplay -nojvm -singleCompThread -r generate_LR_Vimeo90K_BD

up_scale = 4;
mod_scale = 4;
sigma = 1.6;
idx = 0;
filepaths = dir('/scratch/190250671.tmpdir/vimeo90k/vimeo_septuplet/sequences/*/*/*.png');
for i = 1 : length(filepaths)
    [~,imname,ext] = fileparts(filepaths(i).name);
    folder_path = filepaths(i).folder;
    save_LR_folder = strrep(folder_path,'vimeo_septuplet','vimeo_septuplet_BDLRx4');
    if ~exist(save_LR_folder, 'dir')
        mkdir(save_LR_folder);
    end
    if isempty(imname)
        disp('Ignore . folder.');
    elseif strcmp(imname, '.')
        disp('Ignore .. folder.');
    else
        idx = idx + 1;
        str_result = sprintf('%d\t%s.\n', idx, imname);
        fprintf(str_result);
        % read image
        img = imread(fullfile(folder_path, [imname, ext]));
        img = im2double(img);
        % modcrop
        img = modcrop(img, mod_scale);
        % LR
        im_LR = BD_degradation(img, up_scale, sigma);
        if exist('save_LR_folder', 'var')
            fprintf('\n %d, %s', idx, imname)
            imwrite(im_LR, fullfile(save_LR_folder, [imname, '.png']));
        end
    end
end
end

%% modcrop
function img = modcrop(img, modulo)
if size(img,3) == 1
    sz = size(img);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2));
else
    tmpsz = size(img);
    sz = tmpsz(1:2);
    sz = sz - mod(sz, modulo);
    img = img(1:sz(1), 1:sz(2),:);
end
end

%% blur-downsampling degradation
function img = BD_degradation(img, up_scale, sigma)
kernelsize = ceil(sigma * 3) * 2 + 2;
kernel = fspecial('gaussian', kernelsize, sigma);
img = imfilter(img, kernel, 'replicate');
img = img(up_scale/2:up_scale:end-up_scale/2, up_scale/2:up_scale:end-up_scale/2, :);
end