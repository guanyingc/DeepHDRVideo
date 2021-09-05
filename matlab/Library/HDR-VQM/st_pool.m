function res = st_pool(raw_data,percentile)
% error checking.
[r,c,t] = size(raw_data);
[rows,cols] = size(raw_data);
%no. of rows to consider
%along temporal axis
desired_rows = 1 + round((rows-1) * percentile);
sorted_data = sort(raw_data, 1);
newdata = mean(sorted_data(1:desired_rows,:,:,:),1);
[d1,d2,d3,d4] = size(newdata);
res = reshape(newdata,d2,d3,d4);
