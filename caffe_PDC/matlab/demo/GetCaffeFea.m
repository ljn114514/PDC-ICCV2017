function wfea = GetCaffeFea(lists,caffePara) 

% matcaffe_init(1, caffePara.defineDir,caffePara.modelDir);
% caffe('set_device', caffePara.gpuDevice);

caffe.set_mode_gpu();
caffe.set_device(caffePara.gpuDevice);
% Initialize a network
phase = 'test'; 
net = caffe.Net(caffePara.defineDir, caffePara.modelDir, phase);
ext_scale = caffePara.ext_scale;

wfea = zeros(caffePara.feaDim,length(lists)*caffePara.batNum);
if caffePara.isrgb
    images = zeros(caffePara.imgDim, caffePara.imgDim,3,caffePara.batNum, 'single');
else
    images = zeros(caffePara.imgDim, caffePara.imgDim,1,caffePara.batNum, 'single');
end
for j = 1:length(lists)
    disp(['batch{',int2str(j),'/',int2str(length(lists)),'} extract ...']);
    tic;
    rct_list = regexprep(lists{j},'\.jpg|\.png','\.rct');
    for k = 1:caffePara.batNum
        try
%             img_name = load(lists{j}{k});
            img_name = lists{j}{k};
            im = imread(img_name);
        catch
            error('Image open fail,may the directory is error!');
        end
        %% expand face roi
        rct = importdata(rct_list{k});
        [row, col, ~] = size(im);
        w = rct(3) - rct(1);
        h = rct(4) - rct(2);
        rct(1) = floor(rct(1) - ext_scale*w/2);
        rct(2) = floor(rct(2) - ext_scale*h/2);
        rct(3) = floor(rct(3) + ext_scale*w/2);
        rct(4) = floor(rct(4) + ext_scale*w/2);
        
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
        
%         w = max(rct(3) - rct(1), rct(4) - rct(2));
%         h = w;
        w = rct(3) - rct(1);
        h = rct(4) - rct(2);
        
        try
            if rct(1) + w > col
                w = col - rct(1);
            end
            if rct(2) + h > row
                h = row - rct(2);
            end
            %% 
            im = im(rct(2):rct(2)+h,rct(1):rct(1)+w,:);
        catch
            fprintf('(%d, %d, %d, %d)\n', rct(2), h, rct(1), w);
            fprintf('%d, %d %d\n', size(im, 1), size(im, 2), size(im, 3)); 
            fprintf('%s\n', lists{j}{k});
        end
%         w = min(rct(3) - rct(1), rct(4) - rct(2));
%         h = w;
        
        if caffePara.isrgb
            if ismatrix(im) == 1
                im = cat(3,im,im,im);
            end
            im = im(:, :, [3, 2, 1]);
            try
                im = imresize(im,[caffePara.imgDim,caffePara.imgDim],'bilinear');
            catch
                fprintf('(%d, %d)\n', size(im, 1), size(im, 2));
                pause;
            end
            im = single(im);
            images(:,:,:, k) = permute(im,[2, 1, 3]);
        else
            if ~ismatrix(im)
                im = rgb2gray(im);
            end
            im = imresize(im,[caffePara.imgDim,caffePara.imgDim],'bilinear');
            im = single(im);
            images(:, :, 1, k) = im;
        end
        
        
    end
    input_data = {images};
    fea = net.forward(input_data);
    fea = fea{1};
%     a = squeeze(fea);
    wfea(:,(j-1)*caffePara.batNum+1:j*caffePara.batNum) = squeeze(fea);
    toc;
end
caffe.reset_all();

end