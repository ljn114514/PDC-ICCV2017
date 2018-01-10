function DrawROCs( result_dir, save_roc )
%DrawROCs Draw ROCs from the results in a directory
%   Inputs:
%           result_dir:     path of the training data
%           save_roc:       detected rectangles of the faces
%   Outputs:
%           features:       extracted features
%------------------------------------------------------------------------
%   Author:
%       Junliang Xing (junliangxing@gmail.com), 2014.04.23
%------------------------------------------------------------------------
%

if nargin < 2
    save_roc = true;
end

if nargin < 1
    result_dir = './result/batch/';
end

dirs =  dir([result_dir, '/*.mat']);

color = 'rgbkcmykrgbcmyk';
style1 = '---------------';
style2 = '.....................';
marker= '.o*+xsdv<>^ph.o*+xsdv<>^ph';
plot_name = cell(numel(dirs), 1);

%% 1. draw roc curves
fh = figure('name', 'DrawROCs');
ah = axes('parent', fh);
hold on;grid on;

for i = 1:length(dirs)
    name = dirs(i).name;
    datapath = fullfile(result_dir, name);
    disp(datapath);
    load(datapath);
    align_errors_sort = sort(align_errors);
    n = numel(align_errors_sort);
    p = 100*(1:n)/n;
    plot(ah, align_errors_sort, p,[color(mod(i, length(color))),style1(mod(i, length(style1)))]);
    plot_name{i}=name(1:end-4);
end
axis (ah, [0 0.1 0 100]);
grid on;
legend(plot_name,'Location','SouthEast');
xlabel('Error metric');
ylabel('Cumulative correct rate');
title('Performance Landmark Detection');
hold off;

if save_roc
    saveas(gcf, fullfile(result_dir,'ROCs.png'), 'png');
end

% 2. draw error on each frame
fh = figure('name', 'DrawErrorOnEachFrame');
ah = axes('parent', fh);
hold on;grid on;

for i = 1:length(dirs)
    name = dirs(i).name;
    datapath = fullfile(result_dir, name);
    disp(datapath);
    load(datapath);
    plot(ah, 1:length(align_errors), align_errors, ...
          [color(mod(i, length(color))),style2(mod(i, length(style2)))]);
    plot_name{i}=name(1:end-4);
    
    fprintf('NME of %s is: %f.\n', name, mean(align_errors)*100);
end
% axis(ah, [0 500 0 ]);
grid on;
legend(plot_name,'Location','best');
xlabel('frame');
ylabel('alignment errors');
title('Performance in testing phase');
hold off;

if save_roc
    saveas(gcf, fullfile(result_dir,'alignerrorframe.png'), 'png');
end

end