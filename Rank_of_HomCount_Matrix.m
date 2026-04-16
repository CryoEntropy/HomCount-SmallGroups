% 读取 CSV 文件，将第一列作为行名，第一行作为变量名
filename = 'Hom_counts_16_to_16.csv';
T = readtable(filename, 'ReadRowNames', true);

% 提取数值矩阵（所有数据）
data = T{:,:};

% 计算矩阵的秩
r = rank(data);

% 显示结果
fprintf('矩阵的秩为: %d\n', r);