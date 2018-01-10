close all;
clear;
clc;
addpath('../func/util');

if exist('../+caffe', 'dir')
    addpath('..');
else
    error('Please run this demo from caffe/matlab/demo');
end

parent_path = './VGG_STN_300W';
result_path = './results_300W';

img_list_path = fullfile(parent_path, 'test_list.txt');
model_file = fullfile(parent_path, 'facial_point_iter_220000.caffemodel');
model_def_file = fullfile(parent_path, 'deploy.prototxt');
mat_result_path = fullfile(result_path, 'VGG-STN-0803.mat');
if ~exist(result_path, 'file')
    mkdir(result_path);
end

pts_num = 68;
norm_scale = 224;
% norm_scale = 120;
ext_scale = 0.2;
show_flag = false;

gpuDevice = 0;
caffe.set_mode_gpu();
caffe.set_device(gpuDevice);
% Initialize a network
phase = 'test';
net = caffe.Net(model_def_file, model_file, phase);

img_list = textread(img_list_path,'%s');
pts_list = regexprep(img_list,'\.jpg|\.png','\.pts');
rct_list = regexprep(img_list,'\.jpg|\.png','\.rct');

num = length(img_list);
detected_points = zeros(pts_num,2,num);
gt_pts = zeros(pts_num,2,num);
for i=1:num
    pts = load(pts_list{i});
    gt_pts(:,:,i) = pts;
end
ground_truth = zeros(pts_num,2,num);
for j = 1:num
    
    fprintf('%d/%d\n',j,num);
    try
        im = imread(img_list{j});
    catch
        error('Image open fail,may the directory is error!');
    end
    if ndims(im)==2
        im = cat(3,im,im,im);
    end
	[row, col,channel] = size(im);
    %% expand face roi
    if 1
        rct = importdata(rct_list{j});
    else
        rct = [min(g_pts(:,:,i)) max(g_pts(:,:,i))];
    end
	
	w = rct(3) - rct(1);
	h = rct(4) - rct(2);
	rct(1) = floor(rct(1) - ext_scale*w/2);
	rct(2) = floor(rct(2) - ext_scale*h/2);
	rct(3) = floor(rct(3) + ext_scale*w/2);
	rct(4) = floor(rct(4) + ext_scale*w/2);
	
	if rct(1) <= 0
		rct(1) = 1;
	end
	if rct(2) <= 0
		rct(2) = 1;
	end
	if rct(3) > col
		rct(3) = col;
	end
    if rct(4) > row
        rct(4) = row;
    end
    
% 	w = max(rct(3) - rct(1), rct(4) - rct(2));
% 	h = w;
    w = rct(3) - rct(1);
    h = rct(4) - rct(2);
	
    if rct(1) + w > col
        w = col - rct(1);
    end
    if rct(2) + h > row
        h = row - rct(2);
    end
    
    %%
    img = im(rct(2):rct(2)+h,rct(1):rct(1)+w,:);
    img = imresize(img,[norm_scale,norm_scale]); %,'bilinear'
    
    img = single(img);
    images = zeros(norm_scale,norm_scale,3,1,'single');
    im_data = img(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    images(:,:,:, 1) = permute(im_data, [2, 1, 3]);
    input_data = {images};
    
    scores = net.forward(input_data);
    fea = scores{1}(:);

    %%
    pts = (fea + 1) * norm_scale / 2;
	
    scale_x = w/norm_scale;
    scale_y = h/norm_scale;
	pts(1:pts_num) = pts(1:pts_num) * scale_x + rct(1);
	pts(pts_num+1:end) = pts(pts_num+1:end) * scale_y + rct(2);
    
    detected_points(:,:,j) = [pts(1:pts_num) , ...
                              pts(pts_num+1:end)];
    
    if show_flag
        figure(1);
        imshow(img_list{j});
        hold on;
        for n = 1:size(detected_points, 1)
            plot(detected_points(n,1,j),detected_points(n,2,j),'r.');
            text(detected_points(n,1,j) - 0.02, detected_points(n,2,j), num2str(n), 'color', 'g');
        end
        rectangle('Position', [rct(1), rct(2), rct(3) - rct(1), rct(4) - rct(2)], 'EdgeColor', 'y');
        
        hold off;
        pause;
    end
	
    close all
end
caffe.reset_all();

align_errors = ComputeError(gt_pts, detected_points);
fprintf('NME = %.3f\n', mean(align_errors)*100);
save(mat_result_path,'align_errors');

DrawROCs(result_path);