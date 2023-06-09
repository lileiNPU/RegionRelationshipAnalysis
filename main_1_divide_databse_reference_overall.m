clc;
clear;
close all;

% image path
% save path
save_path_train = '.\data\DataList\train_overall';
if ~exist(save_path_train)
    mkdir(save_path_train);
end
save_path_devel = '.\data\DataList\devel_overall';
if ~exist(save_path_devel)
    mkdir(save_path_devel);
end
save_path_test = '.\data\DataList\test_overall';
if ~exist(save_path_test)
    mkdir(save_path_test);
end

% length of video sequence
num_frame = 2;
length_frame = 16;
over_lap_frame = 10;
face_mode = 'AllFrameImages'; % 'AllFaceImagesWith64';

%% train set
%% real face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/train/real']};
client_id = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        client_id{end + 1} = img_files(i).name(1 : 9);
    end
end
client_id = unique(client_id);

real_mode = {'session01_webcam_authenticate_adverse_1_counter_', 'session01_webcam_authenticate_adverse_2_counter_',...
             'session01_webcam_authenticate_controlled_1_counter_', 'session01_webcam_authenticate_controlled_2_counter_'};

train_id_all = {};
for t0 = 1 : length(client_id)
    for t1 = 1 : length(real_mode)
        train_id_all{end + 1} = strcat(client_id{t0}, '_', real_mode{t1});
    end
end

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    img_name = img_files(i).name;
    img_id = img_name;
    index_cut = strfind(img_id, '_');
    id_all{end + 1} = img_id(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(train_id_all)
    id_sub = train_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_train, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 0;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
        
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% printed fake face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/train/print_paper_attack']};
train_id_all = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        index_cut = strfind(img_files(i).name, '_counter_');
        train_id_all{end + 1} = img_files(i).name(1 : index_cut); 
    end
end
train_id_all = unique(train_id_all);

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    index_cut = strfind(img_files(i).name, '_counter_');
    id_all{end + 1} = img_files(i).name(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(train_id_all)
    id_sub = train_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_train, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, 'counter_', num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 1;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% replay fake face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/train/replay_attack']};
train_id_all = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        index_cut = strfind(img_files(i).name, '_counter_');
        train_id_all{end + 1} = img_files(i).name(1 : index_cut); 
    end
end
train_id_all = unique(train_id_all);

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    index_cut = strfind(img_files(i).name, '_counter_');
    id_all{end + 1} = img_files(i).name(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(train_id_all)
    id_sub = train_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_train, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, 'counter_', num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 1;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% screen fake face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/train/print_screen_attack']};
train_id_all = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        index_cut = strfind(img_files(i).name, '_counter_');
        train_id_all{end + 1} = img_files(i).name(1 : index_cut); 
    end
end
train_id_all = unique(train_id_all);

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    index_cut = strfind(img_files(i).name, '_counter_');
    id_all{end + 1} = img_files(i).name(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(train_id_all)
    id_sub = train_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_train, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, 'counter_', num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 1;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
    end
end

%% devel set

over_lap_frame = over_lap_frame * 1;

% real face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/devel/real']};
client_id = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        client_id{end + 1} = img_files(i).name(1 : 9);
    end
end
client_id = unique(client_id);

real_mode = {'session01_webcam_authenticate_adverse_1_counter_', 'session01_webcam_authenticate_adverse_2_counter_',...
             'session01_webcam_authenticate_controlled_1_counter_', 'session01_webcam_authenticate_controlled_2_counter_'};

devel_id_all = {};
for t0 = 1 : length(client_id)
    for t1 = 1 : length(real_mode)
        devel_id_all{end + 1} = strcat(client_id{t0}, '_', real_mode{t1});
    end
end

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    img_name = img_files(i).name;
    img_id = img_name;
    index_cut = strfind(img_id, '_');
    id_all{end + 1} = img_id(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(devel_id_all)
    id_sub = devel_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_devel, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 0;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
%         fclose(fid_train);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% printed fake face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/devel/print_paper_attack']};
devel_id_all = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        index_cut = strfind(img_files(i).name, '_counter_');
        devel_id_all{end + 1} = img_files(i).name(1 : index_cut); 
    end
end
devel_id_all = unique(devel_id_all);

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    index_cut = strfind(img_files(i).name, '_counter_');
    id_all{end + 1} = img_files(i).name(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(devel_id_all)
    id_sub = devel_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_devel, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, 'counter_', num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 1;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
%         fclose(fid_train);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% replayed fake face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/devel/replay_attack']};
devel_id_all = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        index_cut = strfind(img_files(i).name, '_counter_');
        devel_id_all{end + 1} = img_files(i).name(1 : index_cut); 
    end
end
devel_id_all = unique(devel_id_all);

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    index_cut = strfind(img_files(i).name, '_counter_');
    id_all{end + 1} = img_files(i).name(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(devel_id_all)
    id_sub = devel_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_devel, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, 'counter_', num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 1;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
%         fclose(fid_train);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% screen fake face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/devel/print_screen_attack']};
devel_id_all = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        index_cut = strfind(img_files(i).name, '_counter_');
        devel_id_all{end + 1} = img_files(i).name(1 : index_cut); 
    end
end
devel_id_all = unique(devel_id_all);

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    index_cut = strfind(img_files(i).name, '_counter_');
    id_all{end + 1} = img_files(i).name(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(devel_id_all)
    id_sub = devel_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_devel, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, 'counter_', num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 1;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
%         fclose(fid_train);
    end
end


%% test set
% real face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/test/real']};
client_id = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        client_id{end + 1} = img_files(i).name(1 : 9);
    end
end
client_id = unique(client_id);

real_mode = {'session01_webcam_authenticate_adverse_1_counter_', 'session01_webcam_authenticate_adverse_2_counter_',...
             'session01_webcam_authenticate_controlled_1_counter_', 'session01_webcam_authenticate_controlled_2_counter_'};

test_id_all = {};
for t0 = 1 : length(client_id)
    for t1 = 1 : length(real_mode)
        test_id_all{end + 1} = strcat(client_id{t0}, '_', real_mode{t1});
    end
end

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    img_name = img_files(i).name;
    img_id = img_name;
    index_cut = strfind(img_id, '_');
    id_all{end + 1} = img_id(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(test_id_all)
    id_sub = test_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_test, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 0;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% printed fake face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/test/print_paper_attack']};
test_id_all = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        index_cut = strfind(img_files(i).name, '_counter_');
        test_id_all{end + 1} = img_files(i).name(1 : index_cut); 
    end
end
test_id_all = unique(test_id_all);

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    index_cut = strfind(img_files(i).name, '_counter_');
    id_all{end + 1} = img_files(i).name(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(test_id_all)
    id_sub = test_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_test, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, 'counter_', num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 1;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% replayed fake face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/test/replay_attack']};
test_id_all = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        index_cut = strfind(img_files(i).name, '_counter_');
        test_id_all{end + 1} = img_files(i).name(1 : index_cut); 
    end
end
test_id_all = unique(test_id_all);

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    index_cut = strfind(img_files(i).name, '_counter_');
    id_all{end + 1} = img_files(i).name(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(test_id_all)
    id_sub = test_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_test, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, 'counter_', num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 1;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
    end
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% screen fake face
set_img_path = {['../../OpticalFlow/Replay-Attack/', face_mode, '/test/print_screen_attack']};
test_id_all = {};
for s = 1 : 1
    img_files = dir(fullfile(set_img_path{s}, '*.jpg'));
    for i = 1 : length(img_files)
        index_cut = strfind(img_files(i).name, '_counter_');
        test_id_all{end + 1} = img_files(i).name(1 : index_cut); 
    end
end
test_id_all = unique(test_id_all);

% 获取所有id
id_all = {};
img_files = dir(fullfile(set_img_path{1}, '*.jpg'));
for i = 1 : length(img_files)
    index_cut = strfind(img_files(i).name, '_counter_');
    id_all{end + 1} = img_files(i).name(1 : index_cut(end));    
end

% 获取每个id的数据
for i = 1 : length(test_id_all)
    id_sub = test_id_all(i); 
    [index, ~] = ismember(id_all, id_sub);
    img_num = sum(index(:));
    id_img_sounter = 0;
    break_index = 0;
    for index = 1 : over_lap_frame : img_num
        sub_frame_index = [];
        for f = 1 : length_frame
            select_index = index + num_frame * (f - 1);
            if select_index > img_num
                break_index = break_index + 1;
            end
            sub_frame_index = [sub_frame_index select_index];
        end
        if break_index > 0
            break;
        end
        id_img_sounter = id_img_sounter + 1;
        fid = fopen([save_path_test, '\', id_sub{1}, num2str(id_img_sounter),'.txt'], 'w');
        for s = 1 : length(sub_frame_index)
            img_name = strcat(id_sub{1}, 'counter_', num2str(sub_frame_index(s)), '.jpg');
            fprintf(fid,'%s ', fullfile(set_img_path{1}, img_name));
        end
        img_label = 1;
        fprintf(fid,'%s ', num2str(img_label));
        fprintf(fid, '\n');
        fclose(fid);
    end
end
