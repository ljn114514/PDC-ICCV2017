function lists = cutNpiece(list,X)
   N = length(list);
   batNum = ceil(N/X);
   lists = cell(X,1);
   for i = 1:X-1
       lists{i} = list((i-1)*batNum+1:i*batNum);
   end
   lists{X} = list((X-1)*batNum+1:end);
end