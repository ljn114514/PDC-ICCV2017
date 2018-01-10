clear
clc
img_list_path = './feature_v/test_list.txt';
img_list = textread(img_list_path,'%s');
pts_list = regexprep(img_list,'\.jpg','\.pts');
pts_num = 19;
show_flag = true;
num = length(img_list);
ground_truth = zeros(pts_num,3,num);
for i=1:num
    pts = importdata(pts_list{i});
    ground_truth(:,:,i) = pts;
end

%%
predictpoints = importdata('./feature_v/test_list.mat');
predictpoints = predictpoints*224+112;
rct_list = regexprep(img_list,'\.jpg','\.rct');
detected_points = zeros(pts_num,2,num);
for i=1:num
    fprintf('processing %d/%d image.\n', i, num);
    rct = importdata(rct_list{i});
    %%
    im = imread(img_list{i});
    sh_scale = 0.0;
    [row, col, dim] = size(im);
    w = rct(3) - rct(1);
    h = rct(4) - rct(2);
    rct(1) = rct(1) - sh_scale*w;
    rct(2) = rct(2) - sh_scale*h;
    rct(3) = rct(3) + sh_scale*w;
    rct(4) = rct(4) + sh_scale*w;
    
    if rct(1)<=0;
        rct(1) = 1;
    end
    if rct(2)<=0;
        rct(2) = 1;
    end
    if rct(3)>col;
        rct(3) = col;
    end
    if rct(4)>row;
        rct(4) = row;
    end
    %%
    scale_x = (rct(3)-rct(1))/224;
    scale_y = (rct(4)-rct(2))/224;
    for k=1:pts_num
%         detected_points(k,1,i) = predictpoints(2*k-1,i) * scale_x + rct(1);
%         detected_points(k,2,i) = predictpoints(2*k,i) * scale_y + rct(2);
        detected_points(k,1,i) = predictpoints(k,i) * scale_x + rct(1);
        detected_points(k,2,i) = predictpoints(k+pts_num,i) * scale_y + rct(2);
    end
    
    if show_flag
        figure(1);
        imshow(img_list{i});
        hold on;
        plot(detected_points(:,1,i),detected_points(:,2,i),'r.');
        % plot(ground_truth(:,1,i),ground_truth(:,2,i),'b.');
        for n = 1:size(detected_points, 1)
            text(detected_points(n,1,i) - 0.02, detected_points(n,2,i), num2str(n), 'color', 'g');
        end
        rectangle('Position', [rct(1), rct(2), rct(3) - rct(1), rct(4) - rct(2)], 'EdgeColor', 'y');
        
        hold off;
        pause;
    end
end

align_errors = ComputeError(ground_truth, detected_points);
save('result_v/vgg-72253.mat','align_errors');
DrawROCs( 'result_v' );
