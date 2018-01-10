clc;
clear;
addpath('../func/util');

if exist('../+caffe', 'dir')
    addpath('..');
else
    error('Please run this demo from caffe/matlab/demo');
end

imglist = './VGG_STN_AFLW/test_list.txt'; %
model_file = './VGG_STN_AFLW/facial_point_iter_60000.caffemodel';
model_def_file = './VGG_STN_AFLW/deploy.prototxt';
dst_path = './results_norm';

IMG_DIM = 224;
sh_scale = -0;
pts_num = 19;

if ~exist(dst_path, 'file')
    mkdir(dst_path);
end

gpuDevice = 3;
caffe.reset_all();
caffe.set_mode_gpu();
caffe.set_device(gpuDevice);
% Initialize a network
phase = 'test';
net = caffe.Net(model_def_file, model_file, phase);

lists = textread(imglist,'%s');
rct_list = regexprep(lists,'\.jpg|\.png','\.rct');
pts_list = regexprep(lists,'\.jpg|\.png','\.pts');

num = length(lists);
detected_points = zeros(pts_num,2,num);
ground_truth = zeros(pts_num,2,num);
for j = 1:num
    
    fprintf('%d/%d\n',j,num);
    try
        im = imread(lists{j});
    catch
        error('Image open fail,may the directory is error!');
    end
    if ndims(im)==2
        im = cat(3,im,im,im);
    end
    
    loc = strfind(lists{j}, '/');
    img_name = lists{j}(loc(end)+1:end-4);
    
    %% expand face roi
    rct = importdata(rct_list{j});
    [row, col,channel] = size(im);
    w = rct(3) - rct(1);
    h = rct(4) - rct(2);
    rct(1) = floor(rct(1) - sh_scale*w/2);
    rct(2) = floor(rct(2) - sh_scale*h/2);
    rct(3) = floor(rct(3) + sh_scale*w/2);
    rct(4) = floor(rct(4) + sh_scale*h/2);
    
    if rct(1)<=0
        rct(1) = 1;
    end
    if rct(2)<=0
        rct(2) = 1;
    end
    if rct(3)>col
        rct(3) = col;
    end
    if rct(4)>row 
        rct(4) = row;
    end
    g_pts = importdata(pts_list{j});
    g_pts = g_pts(:, 1:2);
    ground_truth(:,:,j) = g_pts;
    
    figure(1);
    imshow(im);
    hold on;
    for n = 1:size(ground_truth, 1)
            plot(ground_truth(n, 1, j),ground_truth(n, 2, j),'b.');
%             text(ground_truth(n, 1, j) - 0.02, ground_truth(n, 1, j), num2str(n), 'color', 'g');
    end
    rectangle('Position', [rct(1), rct(2), rct(3) - rct(1), rct(4) - rct(2)], 'EdgeColor', 'r', 'LineWidth', 3);
    
    hold off;
    
    saveas(gcf, fullfile(dst_path, [img_name, '_init.jpg']), 'jpg');
    
    %% 1. ground true pts
    g_pts(:, 1) = 2 * (g_pts(:, 1) - rct(1)) / (rct(3) - rct(1)) - 1;
    g_pts(:, 2) = 2 * (g_pts(:, 2) - rct(2)) / (rct(4) - rct(2)) - 1;
    
%     label = single(g_pts(:));
    
    %%
    im = im(rct(2):rct(4),rct(1):rct(3),:);
    im = imresize(im,[IMG_DIM,IMG_DIM]); %,'bilinear'
    
    im = single(im);
    images = zeros(IMG_DIM,IMG_DIM,3,1,'single');
    im_data = im(:, :, [3, 2, 1]);  % permute channels from RGB to BGR
    images(:,:,:, 1) = permute(im_data, [2, 1, 3]);
%     input_data = {images, label};
    input_data = {images};
    
    tic;
    net.forward(input_data);
    theta = net.blobs('theta').get_data();
    theta = squeeze(theta)';
    st_label = net.blobs('local/19point').get_data();
    st_label = squeeze(st_label)';
    st_data = net.blobs('st_data').get_data();
    st_data = permute(uint8(st_data), [2, 1, 3]);
    st_data = st_data(:, :, [3, 2, 1]);
    
    %%
    pts = (st_label + 1) * IMG_DIM / 2;
    figure(2);
    imshow(st_data);
    hold on
    for k=1:pts_num
        plot(pts(k),pts(k + pts_num),'r*');
    end
    hold off;
    
    %% 
    rct_w = rct(3) - rct(1);
    rct_h = rct(4) - rct(2);
    local_rct = [theta(4) - theta(3), theta(2) - theta(1),  ...
                 theta(4) + theta(3), theta(2) + theta(1)];
    global_rct = [(local_rct(1) + 1) * rct_w / 2 + rct(1), ...
                  (local_rct(2) + 1) * rct_h / 2  + rct(2), ...
                  (local_rct(3) + 1) * rct_w / 2 + rct(1), ...
                  (local_rct(4) + 1) * rct_h / 2 + rct(2)];
    figure(3);
    imshow(lists{j});
    hold on;
    for n = 1:size(ground_truth, 1)
            plot(ground_truth(n, 1, j),ground_truth(n, 2, j),'b.');
%             text(ground_truth(n, 1, j) - 0.02, ground_truth(n, 1, j), num2str(n), 'color', 'g');
    end
    rectangle('Position', [global_rct(1), global_rct(2), ...
              global_rct(3) - global_rct(1), global_rct(4) - global_rct(2)], ...
              'EdgeColor', 'g', 'LineWidth', 3);
    
    hold off;
    saveas(gcf, fullfile(dst_path, [img_name, '_stn.jpg']), 'jpg');
    
%     pause;          
    
end
caffe.reset_all();