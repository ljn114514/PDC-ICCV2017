load('pose_label.mat');
load('list.mat');

addpath(genpath('../caffe/caffe-master/matlab'));
use_gpu = 3;
% Set caffe mode
if exist('use_gpu', 'var') && use_gpu
  caffe.set_mode_gpu();
  gpu_id = 1;  % we will use the first gpu in this demo
  caffe.set_device(gpu_id);
else
  caffe.set_mode_cpu();
end


net_model = 'deploy.prototxt';
net_weights = 'FCN32sLSPx100s256Solver1_iter_60000.caffemodel';

phase = 'test'; % run with phase test (so that dropout isn't applied)
if ~exist(net_weights, 'file')
  error('Please download CaffeNet from Model Zoo before you run this demo');
end
% Initialize a network
net = caffe.Net(net_model,net_weights, phase);

ori_path = '/media/vmc/disk1/competition/test/vr_path/';
tar_path = '/media/vmc/disk1/competition/test_together2/vr_path/';


%img_num = length(img_path);
img_num = length(listvr);

disp(img_num);
for i=1:img_num
    ori_img_path = [ori_path listvr{i,1}];
    disp(ori_img_path);
    disp(i)
    disp(i/img_num);
    
    im = imread(ori_img_path);
    im = imresize(im,[256,128]);
    im = padarray(im, [0,64],'replicate','both');
    %imshow(im);
    
    im_data = im(:, :, [3, 2, 1]);   % permute channels from RGB to BGR    
    im_data = permute(im_data, [2, 1, 3]);  % flip width and height
    im_data = single(im_data);       % convert from uint8 to single    
    
    scores = net.forward({im_data});
    s = scores{1};
    
    result = cell(15,1);
    for k=1:15    
        s_max = s(:, :, k);
        s_max = s_max';
        %imshow(s_max,[]);
        [rows,cols] = size(s_max);
        x = ones(rows,1)*[1:cols];
        y = [1:rows]'*ones(1,cols);
        area = sum(sum(s_max));
        meanx = sum(sum(s_max.*x))/area;
        meany = sum(sum(s_max.*y))/area;
        result{k} = [meanx,meany];        
    end    
    res = cell2mat(result);
    
    sticks = zeros(9,5);
    sticks(1,:) = [res(15,1),res(15,2),res(15,1),res(15,2),0.0];%head	
    sticks(2,:) = [res(10,1),res(10,2),res(8,1),res(8,2),0.0];%right arm
    sticks(3,:) = [res(11,1),res(11,2),res(13,1),res(13,2),0.0];%left arm
    sticks(4,:) = [res(4,1),res(4,2),res(2,1),res(2,2),0.0];%right leg
    sticks(5,:) = [res(5,1),res(5,2),res(7,1),res(7,2),0.0];%left leg
    sticks(6,:) = [res(14,1),res(14,2),(res(4,1)+res(5,1))/2,(res(4,2)+res(5,2))/2,0.0];%body
    %color = {'g', 'y', 'b', 'r', 'c', 'k', 'm','w','g'};
    %theta for down clockwise
    
    for j=2:6
        x1 = sticks(j,1);y1 = sticks(j,2);
        x2 = sticks(j,3);y2 = sticks(j,4);
    
        if x2<x1
            if  y2<y1
                %disp((y1-y2)/(x1-x2));
                sticks(j,5)=atand((y1-y2)/(x1-x2))+90;
            else
                %disp((y1-y2)/(x1-x2));
                sticks(j,5)=atand((x1-x2)/(y2-y1));
            end
        else
            if y2<y1 
                %disp((y1-y2)/(x1-x2))
                sticks(j,5)=-(atand((y1-y2)/(x2-x1))+90);
            else
                %disp((y1-y2)/(x1-x2))
                sticks(j,5)=atand((x2-x1)/(y1-y2));
            end
        end    
        
        if x1>x2
            temp = x1;x1=x2;x2=temp;
        end
        if y1>y2
            temp = y1;y1=y2;y2=temp;
        end
        w = x2-x1; h = y2-y1;
        %rectangle('position',[x1-10,y1-10,w+20,h+20],'edgecolor',color{i});    
    end 
      
    
    im = imread(ori_img_path);
    disp(ori_img_path)
	a = imresize(im,[256,128]);
    d = a;
	a = padarray(a,[0,64],'replicate','both');
    %imshow(a);
    [m,n,k] = size(a);
    dx=n/2;dy=m/2;
    
    part = cell(6,1);    
    %head
    part{1} = imcrop(a,[sticks(1,1)-30,sticks(1,2)-30,60,60]);
    part{1} = imresize(part{1},[64,64]);
    part{1} = im2double(part{1});
    
    for j=2:6
        x1=sticks(j,1); y1 = sticks(j,2);
        x2=sticks(j,3); y2 = sticks(j,4);
        theta = sticks(j,5);
        b = imrotate(a,theta,'bilinear');
        [m1,n1,k1] = size(b);
    
        x12=floor( (x1-dx)*cosd(theta)+(y1-dy)*sind(theta) +n1/2);
        y12=floor(-(x1-dx)*sind(theta)+(y1-dy)*cosd(theta) +m1/2);
        x22=floor( (x2-dx)*cosd(theta)+(y2-dy)*sind(theta) +n1/2);
        y22=floor(-(x2-dx)*sind(theta)+(y2-dy)*cosd(theta) +m1/2);
    
        %figure;imshow(b);
        %rectangle('position',[x12-15,y12-20,x22-x12+30,y22-y12+40],'edgecolor',color{i});   
        
        %arm
        if j==2||j==3
            part{j} = imcrop(b,[x12-20,y12-15,x22-x12+40,y22-y12+30]);
            part{j} = imresize(part{j},[128,32]);
        end

        %leg
        if j==4||j==5
            part{j} = imcrop(b,[x12-20,y12-15,x22-x12+40,y22-y12+30]);
            part{j} = imresize(part{j},[128,32]);
        end
        
        %body
        if j==6
            part{j} = imcrop(b,[x12-40,y12-34,x22-x12+80,y22-y12+68]);
            part{j} = imresize(part{j},[128,64]);
            %figure;imshow(part{i});
        end
        part{j} = im2double(part{j});
        %
    end
    
    b = imresize(im,[512,256]);
    b = im2double(b);    
    a = im2double(a);    
   
    respic = zeros(256,128,3);
    
    respic(1:64,33:96,:) = part{1};
    
    respic(1:128,1:32,:) = part{2};
    respic(1:128,97:128,:) = part{3};
    
    respic(129:256,1:32,:) = part{4};
    respic(129:256,97:128,:) = part{5};  
    
    respic(65:192,33:96,:) = part{6};    

    %respic = imresize(respic,[224,224]);  
    d = im2double(d);
    respic = [d respic];
    imshow(respic);
    tar_full_path = [tar_path listvr{i}];
    
    imwrite(respic,tar_full_path);
    
end
caffe.reset_all();

