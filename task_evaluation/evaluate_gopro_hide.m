%% Restormer: Efficient Transformer for High-Resolution Image Restoration
%% Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, and Ming-Hsuan Yang
%% https://arxiv.org/abs/2111.09881

close all;clear all;

% datasets = {'Gopro'};
% pred_folder_name = {'GoPro'};
datasets = {'Hide'};
pred_folder_name = {'HIDE'};
% datasets = {'Gopro', 'Hide'};
num_set = length(datasets);

% tic
% delete(gcp('nocreate'))
% parpool('local',20);

for idx_set = 1:num_set
    file_path = strcat('motion_deblur_images/', pred_folder_name{idx_set}, '/');
    fprintf('Pred path : %s \n', file_path);
    gt_path = strcat('datasets/', datasets{idx_set}, 'Test/target/');
    fprintf('GT path : %s \n', gt_path);
    path_list = [dir(strcat(file_path,'*.jpg')); dir(strcat(file_path,'*.png'))];
    gt_list = [dir(strcat(gt_path,'*.jpg')); dir(strcat(gt_path,'*.png'))];
    img_num = length(path_list);

    fprintf('Pred count : %d \n', length(path_list));
    fprintf('GT count : %d \n', length(gt_list));

    total_psnr = 0;
    total_ssim = 0;
    if img_num > 0 
        % parfor j = 1:img_num 
        for j = 1:img_num 
           image_name = path_list(j).name;
           gt_name = gt_list(j).name;
           input = imread(strcat(file_path,image_name));
           gt = imread(strcat(gt_path, gt_name));
           ssim_val = ssim(input, gt);
           psnr_val = psnr(input, gt);
           total_ssim = total_ssim + ssim_val;
           total_psnr = total_psnr + psnr_val;
       end
    end
    qm_psnr = total_psnr / img_num;
    qm_ssim = total_ssim / img_num;
    
    fprintf('For %s dataset PSNR: %f SSIM: %f\n', datasets{idx_set}, qm_psnr, qm_ssim);

end
% delete(gcp('nocreate'))
% toc
