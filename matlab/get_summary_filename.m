function [summary_filename] = get_summary_filename(summary_dir, dataname)
    if ~exist(summary_dir, 'dir')
        mkdir(summary_dir)
    end
    %[~, hostname] = system('hostname');
    %hostname = getenv('HOSTNAME');
    %hostname = hostname(1: min(7, length(hostname)));
    summary_filename = sprintf('summary_%s_%s.txt', date, dataname);
    %summary_filename = sprintf('summary_%s_%s_%s.txt', date, dataname, hostname);
    summary_filename = fullfile(summary_dir, summary_filename);
end
