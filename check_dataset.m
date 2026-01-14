
file_name = 'datasets/ServerMachineDataset/interpretation_label/machine-1-1.txt';
train_table = readmatrix(file_name);

file_name = 'datasets/ServerMachineDataset/train/machine-1-1.txt';
labeled_table = readmatrix(file_name);

%%

size(train_table) % 8     2

% 4629-4688:9,10,11,13,15,18
% 5486-5491:18
% 5875-5951:2,10,11,12,13,18,24,25,26,32,33,34,35,36
% 15415-15418:18
% 15540-15605:7,18
% 15925-15973:6,7,10,11,13,14,20,30
% 18645-18801:1,6,7,11,12,14,16,19,20,21,22,23,28,31
% 20235-20271:6,7,12,13,20,30
% 22264-22336:1,2,3,4
% 23093-23115:1,3,4,7,19,21,22,23,28,31

size(labeled_table) % 28479          38



temp_label_range = [4629,4688;...
    5486,5491;...
    5875,5951;...
    15415,15418;...
    15540,15605;...
    15925,15973;...
    18645,18801;...
    20235,20271;...
    22264,22336;...
    23093,23115];

temp_label_all = zeros(size(labeled_table,1),1);


temp_label_list = [];
for i = 1 : size(temp_label_range,1)
    temp_label_list = cat(1,temp_label_list,[temp_label_range(i,1):temp_label_range(i,2)]');
    temp_label_all(temp_label_range(i,1):temp_label_range(i,2)) = 1;
end

area(1:4,[1 0 1 2],'EdgeColor',[1 1 1]);
alpha(.1)
% scatter(temp_label_list,2*ones(size(temp_label_list)),'*');
figure,
a = area(-1*temp_label_all,'FaceColor',[1 0 0],'EdgeColor',[1 1 1]);
alpha(.75)
hold on;
plot(labeled_table(:,10));
%%
file_name = 'datasets/ServerMachineDataset/test_label/machine-1-1.txt';
train_table = readmatrix(file_name);

file_name = 'datasets/ServerMachineDataset/test/machine-1-1.txt';
labeled_table = readmatrix(file_name);


figure,
a = area(-1*train_table,'FaceColor',[1 0 0],'EdgeColor',[1 1 1]);
alpha(.75)
hold on;
plot(labeled_table);
%%
filename = 'output/SMD/1-1/27062021_114402/test_output.pkl';
% fid=py.open(filename,'rb');
fid = py.open(filename,'rb');
data = py.pickle.load(fid);


%%

file_name = 'datasets/data/smap_train_md.csv';
train_proc_table = readmatrix(file_name);

file_name = 'datasets/data/labeled_anomalies.csv';
labeled_proc_table = readmatrix(file_name);