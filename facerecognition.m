clear;
%由于计算协方差矩阵的特征向量和特征值运算量巨大，所以在第一次导出数据后注销计算
%环节，生成矩阵.mat文件，往后是通过导入该矩阵文件对特征向量和特征值进行引用
load ./eigenfaces/v.mat;
load ./eigenfaces/latent1.mat;

%提取faceR的人脸样本说明，提取出表情并以转换为数值分类。
%1：微笑 2：严肃 3：滑稽
faceDR = importdata('./eigenfaces/faceDR1.txt');
face_category_X = zeros(2000,2);
for i = 1:2000;
    faceDRstr = char(faceDR(i));
    face_category_X(i,1) = str2num(faceDRstr(1:5));
    if strfind(faceDRstr,'smiling') > 0
        face_category_X(i,2) = 1;%微笑
    elseif strfind(faceDRstr,'serious') > 0
        face_category_X(i,2) = 2;%严肃
    else
        face_category_X(i,2) = 3;%滑稽
    end
end
N = 1223-1;
b = [1228-N,1232-N,1808-N,2412-N,2416-N];
face_category_X(b,:) = [];

%PCA主成分分析算法
fclose('all');
filepath = './rawdata/rawdata/' ;
j = 0;
M = 2000;
X = zeros(M,16384);
%从rawdata文件夹里读取前2000个图片信息作为训练样本，其中不能打开的文件剔除
for i = 1223:3222
    fid=fopen([filepath num2str(i)]);
    if fid > 0
        I = fread(fid)';
        [row,col] = size(I);
        if col ~= 16384
            fprintf('i = %d\n',i);
            fclose(fid);
            continue;
        end
        j = j + 1;
        X(j,:) = I;
    fclose(fid);
    elseif fid < 0
            fprintf('i = %d\n',i);
    end
end
X = X(1:j,:);%样本集合
M = 1995;

mean_X = mean(X,1);%平均脸
meanX = zeros(M,128*128);
for i = 1:M
    meanX(i,:) = X(i,:) - mean_X;
end

% %计算样本集的协方差矩阵
% var = cov(meanX);
% [v,latent1] = eig(var);
% %导出特征向量矩阵和特征值矩阵
% save('./eigenfaces/v','v');
% save('./eigenfaces/latent1','latent1');

latent = diag(latent1);
%按特征值大小降序排列
latentsort = flipud(latent);


%希望新特征所能代表的数据总方差的比例为90%
latentsum = sum(latent);
latentsort_acc = 0;
p = 0;
while ((latentsort_acc/latentsum) < 0.9)
    p = p + 1;
    latentsort_acc = sum(latentsort(1:p));
end

v_extract = v(:,16384-p+1:16384);
X_extract = X*v_extract;

% t = X_extract(2,:)';    
% T = v_extract * t + mean_X';
% imagesc(reshape(T, 128, 128)'); 
% colormap(gray(256));
X_extract_mean = mean(X_extract,1);

%测试样本
%提取faceS的人脸样本说明，提取出表情并以转换为数值分类。
%1：微笑 2：严肃 3：滑稽
faceDS = importdata('./eigenfaces/faceDS1.txt');
face_category_T = zeros(2000,2);
for i = 1:2000;
    faceDSstr = char(faceDS(i));
    face_category_T(i,1) = str2num(faceDSstr(1:5));
    if strfind(faceDSstr,'smiling') > 0
        face_category_T(i,2) = 1;%微笑
    elseif strfind(faceDSstr,'serious') > 0
        face_category_T(i,2) = 2;%严肃
    else
        face_category_T(i,2) = 3;%滑稽
    end
end
N = 3223-1;
b = [4056-N,4135-N,4136-N,5004-N];
face_category_T(b,:) = [];

fclose('all');
filepath = './rawdata/rawdata/' ;

j = 0;
M = 2000;
T = zeros(M,16384);
%从rawdata文件夹里读取后2000个图片信息作为测试样本，其中不能打开的文件剔除
for i = 3223:5222
    fid=fopen([filepath num2str(i)]);
    if fid > 0
        I = fread(fid)';
        [row,col] = size(I);
        if col ~= 16384
            fprintf('i = %d\n',i);
            fclose(fid);
            continue;
        end
        j = j + 1;
        T(j,:) = I;
    fclose(fid);
    elseif fid < 0
            fprintf('i = %d\n',i);
    end
end
T = T(1:j,:);%样本集合
%对测试样本进行特征降维
T_extract = T*v_extract;

%K折交叉验证,k取3
k = 3;
classification_rate = zeros(1,k);%分类率数组
%合并训练集和测试集
sample_merge = [X_extract face_category_X(:,2);T_extract face_category_T(:,2)];
[rows,cols] = size(sample_merge);
mean_merge = floor(rows/k);
for i = 1:k
    if i == 1
        test_sample = sample_merge(((i-1)*mean_merge+1):i*mean_merge,:);
        train_sample = sample_merge((i*mean_merge+1):rows,:);
    elseif i == k
        test_sample = sample_merge(((i-1)*mean_merge+1):rows,:);
        train_sample = sample_merge(1:(i-1)*mean_merge,:);
    else
        test_sample = sample_merge(((i-1)*mean_merge+1):i*mean_merge,:);
        train_sample = [sample_merge(1:(i-1)*mean_merge,:);sample_merge((i*mean_merge+1):rows,:)];
    end
    
    train_sample_mean = mean(train_sample(:,1:cols-1),1);
    
    %id3算法的数据预处理
    %将特征提取后的样本集转换成符合决策树输入矩阵的形式
    train_bin = datapreprocess(train_sample,train_sample_mean);
    %将特征提取后的测试集转换成符合决策树输入矩阵的形式
    test_bin = datapreprocess(test_sample,train_sample_mean);
    
    %id3算法
    %由训练样本集构建id3树
    [~,n] = size(train_bin);
    train_feature = ones(1,n-1);
    %root是id3树的根节点
    root = createTree(train_bin,train_feature);
    %计算分类率（正确个数/测试样本数）
    
    right_count = 0;
    wrong_count = 0;
    %对测试样本进行分类，并检测正确率
    [m,~] = size(test_bin);
    for j = 1:m
        category = facejudge(root,test_bin(j,:));
        if category == test_bin(j,n);
            right_count = right_count + 1;
        else
            wrong_count = wrong_count + 1;
        end
    end
    classification_rate(1,i) = right_count/m;
end
classification_rate_mean = mean(classification_rate(:,1),1);







