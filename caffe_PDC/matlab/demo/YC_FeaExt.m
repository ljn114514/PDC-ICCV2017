function ret = YC_FeaExt(imglist,para)
% this function is ext caffe feature for gray image
ret = 0;
if length(para.modelDir) ~= length(para.defineDir)
    disp('input CF parameter is wrong\n'); 
    return;
end

savepath = para.savepath;
if ~exist(savepath,'dir')
    mkdir(savepath);
end

% model para set
caffePara.imgDim = para.resize;
caffePara.batNum = para.batSize;
caffePara.feaDim = para.feaDim;
caffePara.ext_scale = para.ext_scale;
caffePara.isrgb = para.isrgb;
% feature extract
[~,name] = fileparts(imglist);
templist = textread(imglist,'%s');
cur = length(para.modelDir);
lists = cut2piece(templist,caffePara.batNum); 

wfea = cell(cur,1);
for i =1 :cur
    caffePara.defineDir = para.defineDir{i};
    caffePara.modelDir = para.modelDir{i};
    if length(para.gpuDevice) == 1
        caffePara.gpuDevice = para.gpuDevice;
        disp('feature extract single , disp info in command window!');
        tempfea = GetCaffeFea(lists,caffePara);
    else
        disp('feature extract paralle , No disp info in command window!');
        deviceNum = para.gpuDevice;
        plists = cutNpiece(lists,length(deviceNum));
        jm = findResource;
        disp('create job...');
        job = createJob(jm,'PathDependencies',para.toolPath);
        for k = 1:length(deviceNum)
            caffePara.gpuDevice = para.gpuDevice(k);
            createTask(job,@GetCaffeFea,1,{plists{k},caffePara});
        end
        submit(job);
        disp('submit job...');
        waitForState(job,'finished');
        results = getAllOutputArguments(job);
        disp('get results from paralle ');
        tempfea = [];
        for k = 1:length(deviceNum)
            tempfea = [tempfea,results{k}];
        end
        destroy(job);
    end
    wfea{i} = tempfea(:,1:length(templist));
end

if strcmp(para.feaType,'link')
    wwfea = [];
    for i = 1:cur
        wwfea = [wwfea;wfea{i}];
    end
else if strcmp(para.feaType,'add')
        wwfea = zeros(size(wfea{i}));
        for i = 1:cur
            wwfea = wwfea + wfea{i};
        end
    end
end
save(fullfile(para.savepath,para.savename),'-v7.3','wwfea');
disp('done!');
ret = 1;
end