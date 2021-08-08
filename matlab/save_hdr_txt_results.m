function [summary_str, tex_summary_str] = save_hdr_txt_results(save_dir, results, expos, summary_file_name)

num = size(results, 1) - 1; % The last value is the averaged result

logfile = fopen(fullfile(save_dir, sprintf('Log_%s_%d.txt', date, num)), 'w');
fwrite(logfile, sprintf('Expos\t PSNRT\t PSNRL\t SSIMT\t HDRVDP\t HDRVQM\n'));

for i = 1: num
    res = sprintf('%.2f\t ', expos(i), results(i, :));
    fwrite(logfile, sprintf('%s\n', res), 'char');
end

% Prepare summary string

summary_str = {};
tex_summary_str = {}; % For LaTex
summary_str{end+1} = sprintf('\nest_dir: %s\n', save_dir);

summary_str{end+1} = ['Avge\t' sprintf('%.2f\t', results(num+1, :)) '\n'];

expo_types = unique(expos);
fprintf('Number of unique exposures: %d\n', length(expo_types));

for i = 1: length(expo_types) 
    expo_index = (expos == expo_types(i));
    avg_expo_res = merge_res_same_index(results(1:end-1, :), expo_index);

    summary_str{end+1} = [sprintf('%.2f\t', expo_types(i), avg_expo_res) '\n'];
    tex_summary_str{end+1} = [sprintf('%.2f & ', avg_expo_res) ];
end
tex_summary_str{end+1} = [sprintf('%.2f & ', results(num+1, :)) '\n'];

summary_file = fopen(summary_file_name, 'at');
for i = 1: length(summary_str)
    %fprintf(summary_str{i});
    fprintf(logfile, summary_str{i});
    fprintf(summary_file, summary_str{i});
end
for i = 1: length(tex_summary_str)
    fprintf(logfile, summary_str{i});
    fprintf(summary_file, tex_summary_str{i});
end
fclose(logfile);
fclose(summary_file);
end
