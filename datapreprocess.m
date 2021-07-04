function M_bin =datapreprocess(M_extract,extract_mean)
%id3算法的数据预处理
%转换成属性为01矩阵
[rows,cols] = size(M_extract);
M_bin = zeros(rows,cols-1);
for i = 1:rows
    for j = 1:(cols-1)
        if M_extract(i,j) >= extract_mean(j)
            M_bin(i,j) = 1;
        else
            M_bin(i,j) = 0;
        end
    end
end
M_bin = [M_bin M_extract(:,cols)];
end