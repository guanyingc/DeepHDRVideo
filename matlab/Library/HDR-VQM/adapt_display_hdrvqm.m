function flag = adapt_display_hdrvqm(path_native,path_adapted,cfg_hdrvqm)



%%% Variables
Param.Display_Max_Value_HDR = 20; 
Param.Display_Max_Value_EXR = 4500;
Param.Display_Max_Value = Param.Display_Max_Value_HDR;
Param.Perc_Max = 0.95;
Param.final_x = 1920;
Param.final_y = 1080;




%%% List the images

Images_List_HDR = dir(fullfile(path_native, '*.hdr'));
Images_List_EXR = dir(fullfile(path_native, '*.exr'));
Images_List = [Images_List_HDR;Images_List_EXR];
for index = 1:length(Images_List)
        Image_Name{index} = Images_List(index).name;
    end


    switch cfg_hdrvqm.do_adapt

    case ('none')
        flag = 1;


    case('linear')

        %%% Find the max value of the current sequence
        Max_Current_Video = 0;
        % Min_Current_Video = 1000;
        %fprintf('Find the max of the sequence\n');
        h = waitbar(0,'Computing scaling factor...');
        if ~exist(path_adapted, 'dir')
            mkdir(path_adapted)
        end
        for index_image = 1:length(Image_Name)
            %fprintf('Check the image %s\n',Image_Name{index_image});
            if(strcmp(Image_Name{index_image}(end-3:end),'.hdr'))
                current_image = hdrimread(fullfile(path_native, Image_Name{index_image}));
            else
                current_image = exrread(fullfile(path_native, Image_Name{index_image}));
            end
            current_image_resized = current_image;
            current_image_resized(current_image_resized<0) = 0;
            data_in_order = sort(current_image_resized(:));
            current_max = mean(data_in_order(floor(length(data_in_order)*Param.Perc_Max):end));
            if(current_max > Max_Current_Video)
                Max_Current_Video = current_max;
            end
            waitbar(index_image/length(Image_Name),h)
        end
        close all force
        h = waitbar(0,'Scaling the frames and writing in dest. folder...');
        for index_image = 1:length(Image_Name)
            %%% Read image
            %fprintf('Load the image %s\n',Image_Name{index_image});
            if(strcmp(Image_Name{index_image}(end-3:end),'.hdr'))
                current_image = hdrimread(fullfile(path_native, Image_Name{index_image}));
                Param.Display_Max_Value = Param.Display_Max_Value_HDR;
            else
                current_image = exrread(fullfile(path_native, Image_Name{index_image}));
                Param.Display_Max_Value = Param.Display_Max_Value_EXR;
            end
            %%% Adjust the luminance to use the dynamic of the Sim2 display
            current_image_resized = current_image;
            current_image_resized(current_image_resized<0) = 0;
            current_image_resized_adjusted = (current_image_resized*Param.Display_Max_Value/Max_Current_Video); %Max);
            if(strcmp(Image_Name{index_image}(end-3:end),'.hdr'))
                hdrimwrite(current_image_resized_adjusted,fullfile(path_adapted, Image_Name{index_image}));
            else
                exrwrite(current_image_resized_adjusted,fullfile(path_adapted, Image_Name{index_image}));
            end
            waitbar(index_image/length(Image_Name),h)
        end
    otherwise
        error('Invalid selection of display processing')
end
close all force	

end
