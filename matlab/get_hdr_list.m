function [hdr_list] = get_hdr_list(in_dir)
hdr_list_name = fullfile(in_dir, 'hdr_list.txt');

if exist(hdr_list_name, 'file')
    fprintf('Loading HDR list: %s\n', hdr_list_name);
    fid = fopen(hdr_list_name);
    hdr_list = textscan(fid, '%s');
    hdr_list = hdr_list{1};
    fclose(fid);
else
    fprintf('Grab all *.hdr files in %s\n', in_dir);
    hdrs = dir(fullfile(in_dir, '*.hdr'));
    for i = 1: length(hdrs)
        hdr_list{i} = hdrs(i).name;
    end
    fprintf('Found %d hdr files\n', numel(hdrs));
end

end

