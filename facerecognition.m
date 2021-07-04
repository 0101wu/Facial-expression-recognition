clear;
%���ڼ���Э����������������������ֵ�������޴������ڵ�һ�ε������ݺ�ע������
%���ڣ����ɾ���.mat�ļ���������ͨ������þ����ļ�����������������ֵ��������
load ./eigenfaces/v.mat;
load ./eigenfaces/latent1.mat;

%��ȡfaceR����������˵������ȡ�����鲢��ת��Ϊ��ֵ���ࡣ
%1��΢Ц 2������ 3������
faceDR = importdata('./eigenfaces/faceDR1.txt');
face_category_X = zeros(2000,2);
for i = 1:2000;
    faceDRstr = char(faceDR(i));
    face_category_X(i,1) = str2num(faceDRstr(1:5));
    if strfind(faceDRstr,'smiling') > 0
        face_category_X(i,2) = 1;%΢Ц
    elseif strfind(faceDRstr,'serious') > 0
        face_category_X(i,2) = 2;%����
    else
        face_category_X(i,2) = 3;%����
    end
end
N = 1223-1;
b = [1228-N,1232-N,1808-N,2412-N,2416-N];
face_category_X(b,:) = [];

%PCA���ɷַ����㷨
fclose('all');
filepath = './rawdata/rawdata/' ;
j = 0;
M = 2000;
X = zeros(M,16384);
%��rawdata�ļ������ȡǰ2000��ͼƬ��Ϣ��Ϊѵ�����������в��ܴ򿪵��ļ��޳�
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
X = X(1:j,:);%��������
M = 1995;

mean_X = mean(X,1);%ƽ����
meanX = zeros(M,128*128);
for i = 1:M
    meanX(i,:) = X(i,:) - mean_X;
end

% %������������Э�������
% var = cov(meanX);
% [v,latent1] = eig(var);
% %���������������������ֵ����
% save('./eigenfaces/v','v');
% save('./eigenfaces/latent1','latent1');

latent = diag(latent1);
%������ֵ��С��������
latentsort = flipud(latent);


%ϣ�����������ܴ���������ܷ���ı���Ϊ90%
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

%��������
%��ȡfaceS����������˵������ȡ�����鲢��ת��Ϊ��ֵ���ࡣ
%1��΢Ц 2������ 3������
faceDS = importdata('./eigenfaces/faceDS1.txt');
face_category_T = zeros(2000,2);
for i = 1:2000;
    faceDSstr = char(faceDS(i));
    face_category_T(i,1) = str2num(faceDSstr(1:5));
    if strfind(faceDSstr,'smiling') > 0
        face_category_T(i,2) = 1;%΢Ц
    elseif strfind(faceDSstr,'serious') > 0
        face_category_T(i,2) = 2;%����
    else
        face_category_T(i,2) = 3;%����
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
%��rawdata�ļ������ȡ��2000��ͼƬ��Ϣ��Ϊ�������������в��ܴ򿪵��ļ��޳�
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
T = T(1:j,:);%��������
%�Բ�����������������ά
T_extract = T*v_extract;

%K�۽�����֤,kȡ3
k = 3;
classification_rate = zeros(1,k);%����������
%�ϲ�ѵ�����Ͳ��Լ�
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
    
    %id3�㷨������Ԥ����
    %��������ȡ���������ת���ɷ��Ͼ���������������ʽ
    train_bin = datapreprocess(train_sample,train_sample_mean);
    %��������ȡ��Ĳ��Լ�ת���ɷ��Ͼ���������������ʽ
    test_bin = datapreprocess(test_sample,train_sample_mean);
    
    %id3�㷨
    %��ѵ������������id3��
    [~,n] = size(train_bin);
    train_feature = ones(1,n-1);
    %root��id3���ĸ��ڵ�
    root = createTree(train_bin,train_feature);
    %��������ʣ���ȷ����/������������
    
    right_count = 0;
    wrong_count = 0;
    %�Բ����������з��࣬�������ȷ��
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







