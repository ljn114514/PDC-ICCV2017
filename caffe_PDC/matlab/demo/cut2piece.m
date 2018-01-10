function lists = cut2piece(list,batNum)
    N = length(list);
    X = ceil(N/batNum);
    lists = cell(X,1);
    for i = 1:X-1
        lists{i} = list((i-1)*batNum+1:i*batNum);
    end
    endlist1 = list((X-1)*batNum+1:end);
    endlist2 = repmat(list(end),[batNum-length(endlist1),1]);
    lists{X} = [endlist1;endlist2];
end