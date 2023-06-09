clc;
clear;
close all;

% save path
save_path = '.\data\TxtList';
if ~exist(save_path)
    mkdir(save_path);
end

%% overall
% train set
fid = fopen([save_path, '/train_deep_flow_overall.txt'], 'w');
txt_files = dir(fullfile('../../OpticalNet/Replay-Attack/data/DataList/train_overall', '*.txt'));
% 获取每个id的数据
for i = 1 : length(txt_files)
    txt_name = txt_files(i).name;
    txt_path = fullfile('./data/DataList/train_overall', txt_name);
    fprintf(fid,'%s \n', txt_path);
end
fclose(fid);

% devel set
fid = fopen([save_path, '/devel_deep_flow_overall.txt'], 'w');
txt_files = dir(fullfile('../../OpticalNet/Replay-Attack/data/DataList/devel_overall', '*.txt'));
% 获取每个id的数据
for i = 1 : length(txt_files)
    txt_name = txt_files(i).name;
    txt_path = fullfile('./data/DataList/devel_overall', txt_name);
    fprintf(fid,'%s \n', txt_path);
end
fclose(fid);

% test face
fid = fopen([save_path, '/test_deep_flow_overall.txt'], 'w');
txt_files = dir(fullfile('../../OpticalNet/Replay-Attack/data/DataList/test_overall', '*.txt'));
% 获取每个id的数据
for i = 1 : length(txt_files)
    txt_name = txt_files(i).name;
    txt_path = fullfile('./data/DataList/test_overall', txt_name);
    fprintf(fid,'%s \n', txt_path);
end
fclose(fid);