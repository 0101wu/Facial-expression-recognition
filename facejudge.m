function category =facejudge(node,data)
if isempty(node.children)
    category = node.value;
    return;
end
node_search = node;
[a,~] = size(node_search.children);
for i = 1:a
    if data(node_search.children(i).name) == node_search.children(i).type
        category = facejudge(node_search.children(i),data);
        return;
    end
end
end