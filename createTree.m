%���ɾ�����ID3�㷨
%data:ѵ����
%feature:���Լ�
function [node] =createTree(data,feature)
  type=mostType(data);
  [m,n]=size(data);
  %���ɽڵ�node
  %value������������Ϊnull���ʾ�ýڵ��Ƿ�֧�ڵ�
  %name:�ڵ㻮������
  %type:�ڵ�����ֵ
  %children:�ӽڵ�
  node=struct('value','null','name','null','type','null','children',[]);
  temp_type=data(1,n);%��¼��һ�����������
  temp_b=true;
  %�ж��������Ƿ�Ϊͬһ������ǵĻ�temp_b=true������temp_b=false
  for i=1:m
    if temp_type ~= data(i,n)
      temp_b=false;
    end
  end
  %������ȫΪͬһ����������node�ڵ�ΪҶ�ӽڵ�
  if temp_b==true
    node.value=data(1,n);
    return;
  end
  %���Լ���Ϊ�գ���������Ϊ���������ķ���
  if sum(feature)==0
    node.value=type;
    return;
  end
  feature_bestColumn=bestFeature(data);%�ҵ���Ϣ������������
  best_feature=data(:,feature_bestColumn);
  best_distinct=unique(best_feature);
  best_num=length(best_distinct);%�����������Եĸ���
  best_proc=zeros(best_num,2);
  best_proc(:,1)=best_distinct(:,1);%��һ��Ϊ��������
  %ѭ�������Ե�ÿһ��ֵ
  for i=1:best_num
    Dv=[];
    Dv_index=1;
    %Ϊnode����һ��bach_node��֧��������data�и�����ֵΪbest_proc(i,1)�ļ���ΪDv
    bach_node=struct('value','null','name','null','type','null','children',[]);
    %Ϊ�����Ե�ÿ�������Ի���������
    for j=1:m
      if best_proc(i,1)==data(j,feature_bestColumn)
        Dv(Dv_index,:)=data(j,:);
        Dv_index=Dv_index+1;
      end
    end
    %DvΪ���򽫽�����Ϊ���������ķ���
    if length(Dv)==0
      bach_node.value=type;
      bach_node.type=best_proc(i,1);
      bach_node.name=feature_bestColumn;
      node.children=[node.children;bach_node];
      return;
    else
      feature(feature_bestColumn)=0;%��������ʹ��
      %�ݹ����createTree����
      bach_node=createTree(Dv,feature);
      bach_node.type=best_proc(i,1);
      bach_node.name=feature_bestColumn;
      node.children=[node.children;bach_node];
    end
  end
end

%���������������
function [res] = mostType(data)
  [m,n]=size(data);
  res_distinct = unique(data(:,n));
  res_proc = zeros(length(res_distinct),2);%��һ��Ϊ����࣬�ڶ���Ϊ��Ӧ���������
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
      res=res_proc(i,1);%�������������������(�±�)
      break;
    end
  end
end

%������Ϣ��
function [entropy] = getEntropy(data)
  entropy=0;
  [m,n]=size(data);
  label=data(:,n);
  label_distinct=unique(label);
  label_num=length(label_distinct);
  proc=zeros(label_num,2);%���Ƿ��࣬�������ĸ���ռ���������ı���
  proc(:,1)=label_distinct(:,1);%��һ�и�ֵΪ�����
  for i=1:label_num
    for j=1:m
      if proc(i,1)==data(j,n)%������������ͳ�Ʒ����Ӧ��������
        proc(i,2)=proc(i,2)+1;
      end
    end
    proc(i,2)=proc(i,2)/m;%���㲻ͬ����ռ��������
  end
  for i=1:label_num%������Ϣ�ع�ʽ����Ϣ��
    entropy=entropy-proc(i,2)*log2(proc(i,2));
  end
end

%������Ϣ����
function [gain] = getGain(entropy,data,column)
  [m,n]=size(data);
  feature=data(:,column);%�ҵ���ǰ�������Զ�Ӧ��һ��
  feature_distinct=unique(feature);%�������м������
  feature_num=length(feature_distinct);
  feature_proc=zeros(feature_num,2);%��һ��Ϊcolumn���Ե��������У��ڶ����Ǹ������Եĸ���
  feature_proc(:,1)=feature_distinct(:,1);
  f_entropy=0;
  for i=1:feature_num
    feature_data=[];
    feature_proc(:,2)=0;
    feature_row=1;
    for j=1:m
      if feature_proc(i,1)==data(j,column)%�Ե�ǰ���������Ե������Խ��м���
        feature_proc(i,2)=feature_proc(i,2)+1;
      end
      if feature_distinct(i,1)==data(j,column)%��ø����Ե���������ӵ�е�������
        feature_data(feature_row,:)=data(j,:);
        feature_row=feature_row+1;
      end
    end
    f_entropy=f_entropy+feature_proc(i,2)/m*getEntropy(feature_data);
  end
  gain=entropy-f_entropy;
end

%��ȡ���Ż�������
function [column] = bestFeature(data)
  [m,n]=size(data);
  featureSize=n-1;
  gain_proc=zeros(featureSize,2);%��һ���������У��ڶ����Ƕ�Ӧ����Ϣ����
  entropy=getEntropy(data);%��ǰ��������������
  for i=1:featureSize%������ǰ����������
    gain_proc(i,1)=i;
    gain_proc(i,2)=getGain(entropy,data,i);%��i�����Զ�Ӧ����Ϣ����
  end
  for i=1:featureSize
    if gain_proc(i,2)==max(gain_proc(:,2))%�ҵ���Ϣ������������
      column=i;%���ظ������±�
      break;
    end
  end
end


