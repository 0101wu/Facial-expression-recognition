%生成决策树ID3算法
%data:训练集
%feature:属性集
function [node] =createTree(data,feature)
  type=mostType(data);
  [m,n]=size(data);
  %生成节点node
  %value：分类结果，若为null则表示该节点是分支节点
  %name:节点划分属性
  %type:节点属性值
  %children:子节点
  node=struct('value','null','name','null','type','null','children',[]);
  temp_type=data(1,n);%记录第一个样本的类别
  temp_b=true;
  %判断样本集是否为同一个类别，是的话temp_b=true，否则temp_b=false
  for i=1:m
    if temp_type ~= data(i,n)
      temp_b=false;
    end
  end
  %样本中全为同一分类结果，则node节点为叶子节点
  if temp_b==true
    node.value=data(1,n);
    return;
  end
  %属性集合为空，将结果标记为样本中最多的分类
  if sum(feature)==0
    node.value=type;
    return;
  end
  feature_bestColumn=bestFeature(data);%找到信息增益最大的属性
  best_feature=data(:,feature_bestColumn);
  best_distinct=unique(best_feature);
  best_num=length(best_distinct);%该属性子属性的个数
  best_proc=zeros(best_num,2);
  best_proc(:,1)=best_distinct(:,1);%第一列为子属性列
  %循环该属性的每一个值
  for i=1:best_num
    Dv=[];
    Dv_index=1;
    %为node创建一个bach_node分支，设样本data中该属性值为best_proc(i,1)的集合为Dv
    bach_node=struct('value','null','name','null','type','null','children',[]);
    %为该属性的每个子属性划分样本集
    for j=1:m
      if best_proc(i,1)==data(j,feature_bestColumn)
        Dv(Dv_index,:)=data(j,:);
        Dv_index=Dv_index+1;
      end
    end
    %Dv为空则将结果标记为样本中最多的分类
    if length(Dv)==0
      bach_node.value=type;
      bach_node.type=best_proc(i,1);
      bach_node.name=feature_bestColumn;
      node.children=[node.children;bach_node];
      return;
    else
      feature(feature_bestColumn)=0;%该属性已使用
      %递归调用createTree方法
      bach_node=createTree(Dv,feature);
      bach_node.type=best_proc(i,1);
      bach_node.name=feature_bestColumn;
      node.children=[node.children;bach_node];
    end
  end
end

%计算样本最多的类别
function [res] = mostType(data)
  [m,n]=size(data);
  res_distinct = unique(data(:,n));
  res_proc = zeros(length(res_distinct),2);%第一列为类别类，第二列为对应的类别数量
  res_proc(:,1)=res_distinct(:,1);
  for i=1:length(res_distinct)
    for j=1:m
      if res_proc(i,1)==data(j,n)
        res_proc(i,2)=res_proc(i,2)+1;
      end
    end
  end
  for i=1:length(res_distinct)
    if res_proc(i,2)==max(res_proc(:,2))
      res=res_proc(i,1);%返回样本数量最多的类别(下标)
      break;
    end
  end
end

%计算信息熵
function [entropy] = getEntropy(data)
  entropy=0;
  [m,n]=size(data);
  label=data(:,n);
  label_distinct=unique(label);
  label_num=length(label_distinct);
  proc=zeros(label_num,2);%行是分类，列是类别的个数占样本总数的比例
  proc(:,1)=label_distinct(:,1);%第一列赋值为类别列
  for i=1:label_num
    for j=1:m
      if proc(i,1)==data(j,n)%遍历样本集，统计分类对应的样本数
        proc(i,2)=proc(i,2)+1;
      end
    end
    proc(i,2)=proc(i,2)/m;%计算不同类所占样本比例
  end
  for i=1:label_num%根据信息熵公式求信息熵
    entropy=entropy-proc(i,2)*log2(proc(i,2));
  end
end

%计算信息增益
function [gain] = getGain(entropy,data,column)
  [m,n]=size(data);
  feature=data(:,column);%找到当前所求属性对应那一列
  feature_distinct=unique(feature);%该属性有几个类别
  feature_num=length(feature_distinct);
  feature_proc=zeros(feature_num,2);%第一列为column属性的子属性列，第二列是该子属性的个数
  feature_proc(:,1)=feature_distinct(:,1);
  f_entropy=0;
  for i=1:feature_num
    feature_data=[];
    feature_proc(:,2)=0;
    feature_row=1;
    for j=1:m
      if feature_proc(i,1)==data(j,column)%对当前样本下属性的子属性进行计数
        feature_proc(i,2)=feature_proc(i,2)+1;
      end
      if feature_distinct(i,1)==data(j,column)%获得该属性的子属性所拥有的样本集
        feature_data(feature_row,:)=data(j,:);
        feature_row=feature_row+1;
      end
    end
    f_entropy=f_entropy+feature_proc(i,2)/m*getEntropy(feature_data);
  end
  gain=entropy-f_entropy;
end

%获取最优划分属性
function [column] = bestFeature(data)
  [m,n]=size(data);
  featureSize=n-1;
  gain_proc=zeros(featureSize,2);%第一列是属性列，第二列是对应的信息增益
  entropy=getEntropy(data);%当前样本集的样本熵
  for i=1:featureSize%遍历当前样本的属性
    gain_proc(i,1)=i;
    gain_proc(i,2)=getGain(entropy,data,i);%第i个属性对应的信息增益
  end
  for i=1:featureSize
    if gain_proc(i,2)==max(gain_proc(:,2))%找到信息增益最大的属性
      column=i;%返回该属性下表
      break;
    end
  end
end


