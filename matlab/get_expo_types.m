function expos = get_expo_types(in_dir)

% Load exposure information for each frame
hdr_expo_file = fullfile(in_dir, 'hdr_expos.txt');
if ~exist(hdr_expo_file, 'file')
    error('File not exist: %s', hdr_expo_file)
end
fid = fopen(hdr_expo_file);
list_data = textscan(fid, '%s %f');
fclose(fid);

hdr_list = list_data{1};
expos = list_data{2};
for i = 1: min(length(expos), 5) % print 5 sample pairs
    fprintf('%d: %s %f\n', i, hdr_list{i}, expos(i))
end

end

