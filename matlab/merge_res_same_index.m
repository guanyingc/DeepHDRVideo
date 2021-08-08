function [avg_res] = merge_res_same_index(matrix2d, index)
    [row, col] = size(matrix2d);
    avg_res = zeros(1, col);
    for i = 1: length(index)
        if index(i) == 1
            avg_res(1, :) = avg_res(1, :) + matrix2d(i, :);
        end
    end
    avg_res = avg_res / sum(index);
end
