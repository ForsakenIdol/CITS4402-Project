classdef project_app_exported < matlab.apps.AppBase

    % Properties that correspond to app components
    properties (Access = public)
        UIFigure                        matlab.ui.Figure
        GridLayout                      matlab.ui.container.GridLayout
        LoadClassesButton               matlab.ui.control.Button
        TestImageLabel                  matlab.ui.control.Label
        PredictedImageLabel             matlab.ui.control.Label
        AppTitle                        matlab.ui.control.Label
        ClassesLoadedLabel              matlab.ui.control.Label
        NumClassesLoadedLabel           matlab.ui.control.Label
        ImagesLoadedLabel               matlab.ui.control.Label
        NumImagesLoadedLabel            matlab.ui.control.Label
        TrainingImagesLabel             matlab.ui.control.Label
        TestImagesLabel                 matlab.ui.control.Label
        NumTrainingImagesLabel          matlab.ui.control.Label
        NumTestImagesLabel              matlab.ui.control.Label
        TrainingImagesPerClassLabel     matlab.ui.control.Label
        TrainingImagesPerClassSlider    matlab.ui.control.Slider
        NumImageSizeLabel               matlab.ui.control.Label
        ImageSizeLabel                  matlab.ui.control.Label
        DownsampledImageSizeLabel       matlab.ui.control.Label
        NumDownsampledImageSizeLabel    matlab.ui.control.Label
        ClassifyTestImagesButton        matlab.ui.control.Button
        AssignedClassLabel              matlab.ui.control.Label
        ValueAssignedClassLabel         matlab.ui.control.Label
        ClassDistanceLabel              matlab.ui.control.Label
        NumClassDistanceLabel           matlab.ui.control.Label
        CurrentAccuracyLabel            matlab.ui.control.Label
        NumCurrentAccuracyLabel         matlab.ui.control.Label
        ImagesClassifiedCorrectlyLabel  matlab.ui.control.Label
        NumImagesClassifiedCorrectlyLabel  matlab.ui.control.Label
        ImagesClassifiedLabel           matlab.ui.control.Label
        NumImagesClassifiedLabel        matlab.ui.control.Label
        DelayTimeoutforDisplaySecondsLabel  matlab.ui.control.Label
        DelayTimeoutDisplaySlider       matlab.ui.control.Slider
        ActualClassLabel                matlab.ui.control.Label
        ValueActualClassLabel           matlab.ui.control.Label
        PredictedImageDisplay           matlab.ui.control.UIAxes
        InputTestImageDisplay           matlab.ui.control.UIAxes
    end

    
    properties (Access = private)
        classnames % A 1D array of all the class names.
        classFiles % A matrix where the second dimension has 11 elements. Each entry has the class name first, then the 10 image file paths for that class.
        trainingimagepaths % Paths to the training images. Each row has the same shape and first classname entry as for the classFiles matrix.
        testimagepaths % Paths to the test images. Each row has the same shape and first classname entry as for the classFiles matrix.
        num_columns % The number of columns of pixels in each original image.
        num_rows % The number of rows of pixels in each original image.
        downsampled_num_columns % The number of columns of pixels in each downsampled image.
        downsampled_num_rows % The number of rows of pixels in each downsampled image.
        preprocessed_training_images % The training images after they have been downsampled, their columns stacked, and their values normalized.
        downsampling_scale % The scale at which to downsample an image. By default, this is 4.
        images_tested % When classifying the test images, how many images have been classified?
        images_correct % When classifying the test images, how many images were classified correctly?
    end
    
    methods (Access = private)
        
        % If the classFiles object is not empty, get the dimensions of each
        % image. We assume that all the images we use have the same
        % dimensions.
        function success = get_image_size(app)
            if ~isempty(app.classFiles)
                num_images = 0;
                total_columns = 0;
                total_rows = 0;
                for i = 1:size(app.classFiles,1)
                    for j = 2:size(app.classFiles,2)

                        current_img = imread(app.classFiles(i,j));
                        dimensions = size(current_img);
                        % Dimensions is a 2-tuple of the form (rows, columns).
                        total_rows = total_rows + dimensions(1);
                        total_columns = total_columns + dimensions(2);
                        num_images = num_images + 1;
                    end
                end
                
                app.num_rows = total_rows ./ num_images;
                app.num_columns = total_columns ./ num_images;
                
                set(app.NumImageSizeLabel, 'Text', sprintf("%d x %d", app.num_rows, app.num_columns));
                
                success = true;
            else
                success = false;
            end
            
        end
        
        % If the classFiles object is not empty, split each class into
        % training and test images based on the value of "training_size".
        function success = train_test_split(app, training_size)
            if ~isempty(app.classFiles)
                
                % Initialize the training and test image matrices
                app.trainingimagepaths = strings([size(app.classFiles,1), training_size + 1]);
                app.testimagepaths = strings([size(app.classFiles,1), 10 - training_size + 1]);
                
                % For each class...
                for i = 1:size(app.classFiles,1)
                    % Assign the current class name.
                    app.trainingimagepaths(i,1) = app.classFiles(i,1);
                    app.testimagepaths(i,1) = app.classFiles(i,1);
                    
                        % Assign the training images to trainingimagepaths.
                        training_i = 2;
                        for j = 2:2+training_size - 1
                            app.trainingimagepaths(i,training_i) = app.classFiles(i,j);
                            training_i = training_i + 1;
                        end
                        
                        % Assign the test images to testimagepaths.
                        test_i = 2;
                        for j = 2+training_size:size(app.classFiles,2)
                            app.testimagepaths(i,test_i) = app.classFiles(i,j);
                            test_i = test_i + 1;
                        end
                        
                end
                
                % Update the relevant labels with the current sizes of each of the training and test image matrices.
                set(app.NumTrainingImagesLabel, 'Text', sprintf("%d",size(app.trainingimagepaths,1) * (size(app.trainingimagepaths,2) - 1)));
                set(app.NumTestImagesLabel, 'Text', sprintf("%d",size(app.testimagepaths,1) * (size(app.testimagepaths,2) - 1)))
                
                % Success!
                success = 1;
            else
                % Aww, that sucks.
                success = 0;
            end
        end
        
        % Given an input image "img", we perform downsampling using row and
        % column sampling with a fixed parameter of 4, which is a multiple
        % of the expected image size of 112 x 92.
        function downsampled = downsample_image(app, img)
            scale = 4;
            app.downsampling_scale = scale;
            
            img_size = size(img);
            app.downsampled_num_rows = floor(img_size(1) ./ scale);
            app.downsampled_num_columns = floor(img_size(2) ./ scale);
            downsampled = zeros([app.downsampled_num_rows, app.downsampled_num_columns], 'uint8');
            % Downsampling is only performed if the input image is not empty.
            if ~isempty(img)
                for i = 1:img_size(1)
                    for j = 1:img_size(2)
                        % Sample the current pixel if its coordinates are multiples of 4.
                        if (mod(i, scale) == 0 && mod(j, scale) == 0)
                            downsampled(i / scale, j / scale) = img(i,j);
                        end
                    end
                end
   
            end
            
            set(app.NumDownsampledImageSizeLabel, 'Text', sprintf("%d x %d", app.downsampled_num_rows, app.downsampled_num_columns));

        end
        
        % There are 3 steps to preprocessing each training image:
        % 1. Downsize each image using a hardcoded scale factor.
        % 2. Stack each column in the downsized image to form one tall, thin vector.
        % 3. Normalize all the values in the column vector so that they lie between 0 and 1 (before, they would've been between 0 and 255).
        function processed = preprocess_image(app, img)
            downsized = downsample_image(app, img);
            % Stack the downsized vector columns
            stacked = double(downsized(:));
            % Normalize vector values (store the result straight into the "processed" return variable)
            processed = stacked / 255;
        end
        
        % If the trainingimagepaths object is not empty, we apply the preprocess_images function to all the images in each class.
        % We then form the vector subspace matrix X_i for each class i, and append all the subspaces together to form X, all the vector
        % subspaces adjacent one after the other.
        % "app.preprocessed_training_images" thus becomes a 3D multidimensional array, where each set of rows and columns
        % represents the vector subspace for a separate class. See the diagrams at https://au.mathworks.com/help/matlab/math/multidimensional-arrays.html
        % for a visual example; each vector subspace is a new "page" in the multidimensional array.
        % We can access the vector subspace (the page) for class i by calling app.preprocessed_training_images(:,:,i).
        function success = preprocess_training_images(app)

            if ~isempty(app.trainingimagepaths)
                
                % Assign the downsampling parameters. These have not yet been set, so we need to assign them here.
                % We do this by reading in an arbitrary image first...
                assignment_img = imread(app.trainingimagepaths(1,2));
                % ... then we call downsample_image, which sets the global downsampling parameters for us.
                downsample_image(app, assignment_img);
                
                % Initialize the first "page" of the multidimensional array to a matrix of zeros.
                % We can now assign to any page of this array.
                all_subspaces = zeros([app.downsampled_num_rows * app.downsampled_num_columns, size(app.trainingimagepaths, 2) - 1], 'double');
                
                % For each class...
                for i = 2:size(app.trainingimagepaths,1)
                    % Get the current class number.
                    current_class_str = app.trainingimagepaths(i,1);
                    current_class_arr = split(current_class_str, 's');
                    current_class = floor(str2double(current_class_arr(2)));
                    
                    % Generate the subspace for the current class.
                    current_subspace = zeros([app.downsampled_num_rows * app.downsampled_num_columns, size(app.trainingimagepaths, 2) - 1], 'double');
                    % For each training image in the current class...
                    for j = 2:size(app.trainingimagepaths,2)
                        current_image = imread(app.trainingimagepaths(i,j)); % Get the image.
                        current_subspace(:,j - 1) = preprocess_image(app, current_image); % Preprocess it and add it to the current subspace.
                    end
                    
                    % Insert the subspace into the correct page for this class.
                    % All the subspaces NEED TO BE IN THE CORRECT ORDER BY CLASS.
                    all_subspaces(:,:,current_class) = current_subspace;
                    
                end
                
                app.preprocessed_training_images = all_subspaces;
                subspace_dims = size(all_subspaces);
                fprintf(1, "All training images have been preprocessed. There are %d pages or subspaces, each of size %d x %d.\n", ...
                    subspace_dims(3), subspace_dims(1), subspace_dims(2));
                success = 1;
            
            else
                success = 0;
            end
            
        end
        
        % Once we've preprocessed our training images, we then want to classify our test images.
        % We iterate over the matrix of test images. For each test image, we preprocess it in the same way we did the training images.
        % We then generate the projection of the test image onto each class, calculate each projection's distance from the test image,
        % then assign the test image to the class who's projection forms the shortest distance to the test image.
        % This function takes an unprocessed image as input, performs the steps above, and returns not only the label of the class assigned
        % to the image, but a list of all the classes that were checked before it.
        function [class_label, class_distance] = classify_test_image(app, z)
            
            if ~isempty(app.preprocessed_training_images)
                % We'll use the variable names in my notes so I don't get confused.
            
                y = preprocess_image(app, z);
                all_distances = zeros([size(app.classnames,1), 1], 'double'); % The distance to each class projection from the training image will appear here.
                
                % For each class...
                for classnum = 1:length(app.classnames)
                    
                    % Get the current class number 'i'.
                    i_str = app.classnames(classnum);
                    i_arr = split(i_str, 's');
                    i = floor(str2double(i_arr(2)));
                    
                    % Step 5, Equation 4b (Generate the projection for the current class).
                    Xi = app.preprocessed_training_images(:,:,i); % Get the subspace for the current class.
                    yi = Xi / (Xi.' * Xi) * Xi.' * y; % Compute the projection of the test image into the current class.
                    
                    % Calculate the distance of this projection from the test image.
                    current_distance = sum((yi - y) .^ 2);
                    % Assign this distance to the object with all distances.
                    all_distances(classnum) = current_distance;
                    
                end
                
                [dist, index] = min(all_distances);
                class_distance = dist;
                class_label = app.classnames(index);
                
                % Display the corresponding image for the closest class.
                sample_class_img = imread(app.trainingimagepaths(index,2));
                imshow(sample_class_img, 'Parent', app.PredictedImageDisplay);
                drawnow
                
            end

        end
        
    end

    % Callbacks that handle component events
    methods (Access = private)

        % Button pushed function: LoadClassesButton
        function LoadClasses(app, event)
            % Get the path to the folder containing the face classes.
            face_path = uigetdir();
            fprintf(1, "User opened path: %s\n", face_path);
            face_folders = dir(face_path); % Get each unique class in the face folder.
            % How many valid classes are there?
            % Valid class names start with an 's'.
            num_valid_classes = 0;
            for face_folders_i = 1:length(face_folders)
                if ~face_folders(face_folders_i).isdir
                    continue
                end
                if ~regexp(face_folders(face_folders_i).name, "^s\\d+")
                    continue
                end
                if face_folders(face_folders_i).name(1) == 's'
                    num_valid_classes = num_valid_classes + 1;
                end
            end
            
            % Construct an array of class names, excluding the '.' and '..' entries.
            % We assume each class name ALWAYS starts with the letter 's'.
            classes = strings([num_valid_classes,1]);
            classes_i = 1;
            for face_folders_i = 1:length(face_folders)
               if face_folders(face_folders_i).name(1) == 's'
                  classes(classes_i) = face_folders(face_folders_i).name;
                  classes_i = classes_i + 1;
               end
            end
            % Add the classes array to the global scope.
            % We can't use "sortrows" on the classes array because it is a row array.
            app.classnames = classes;
            
            % Compile a list of class paths.
            classpaths = strings([length(classes), 1]);
            for i = 1:length(classpaths)
                % fullfile() builds the full file path from its arguments, OS specific.
                classpaths(i) = fullfile(face_path, classes(i));
            end
            
            % Compile a list of image paths for each class.
            % Each class is guaranteed to have exactly 10 images and each row in "class_filepaths" has 11 entries.
            % The first entry is the class name, followed by the paths for each of the 10 images.
            class_filepaths = strings([length(classpaths), 11]);
            % For each class path...
            for i = 1:length(classpaths)
                class_filepaths(i,1) = classes(i); % Assign the current class name.
                files = dir(classpaths(i)); % Get all the files, including '.' and '..'.
                j = 2;
                % Add each file if it has a ".pgm" extension.
                for files_i = 1:length(files)
                    [~,~,ext] = fileparts(files(files_i).name);
                    % The only extension allowed for our image files is ".pgm".
                    if ext == ".pgm"
                        class_filepaths(i,j) = fullfile(classpaths(i), files(files_i).name);
                        j = j + 1;
                    end
                end
            end
            
            % Update the display text with the relevant information.
            set(app.NumClassesLoadedLabel, 'Text', sprintf("%d",length(app.classnames)));
            set(app.NumImagesLoadedLabel, 'Text', sprintf("%d",size(class_filepaths,1) * (size(class_filepaths,2) - 1)));
            % Add our class files to the global scope.
            app.classFiles = class_filepaths;
            
            % Add the average image size.
            get_image_size(app);
            
            % Split our data into training and test splits.
            training_size = floor(app.TrainingImagesPerClassSlider.Value);
            train_test_split(app, training_size);
            
        end

        % Value changing function: TrainingImagesPerClassSlider
        function NumTrainingImagesChanging(app, event)
            training_size = floor(event.Value);
            train_test_split(app, training_size);
        end

        % Button pushed function: ClassifyTestImagesButton
        function ClassifyTestImages(app, event)
            preprocess_training_images(app);
            
            % Reset Metrics.
            app.images_tested = 0;
            app.images_correct = 0;
            
            if ~isempty(app.testimagepaths)
                
                test_dims = size(app.testimagepaths);
                fprintf(1, "There are %d classes of test images, each with %d images in them.\n", test_dims(1), test_dims(2) - 1);
                
                % We iterate over the test images object in a different way. We go down the list of classes
                % such that no 2 images compared one after the other are of the same class.
                for j = 2:test_dims(2) % For each column...
                    for i = 1:test_dims(1) % For each row... (Test Class)
                        
                        % Generate the true class label
                        true_class_label = app.classnames(i);
                        
                        img = imread(app.testimagepaths(i,j));
                        imshow(img, 'Parent', app.InputTestImageDisplay);
                        [predicted_class_label, class_distance] = classify_test_image(app, img);
                        
                        % Update the metrics.
                        app.images_tested = app.images_tested + 1;
                        if (strcmp(true_class_label, predicted_class_label) == 1)
                            app.images_correct = app.images_correct + 1;
                        end
                        
                        % Assign the relevant labels.
                        set(app.ValueActualClassLabel, 'Text', true_class_label);
                        set(app.ValueAssignedClassLabel, 'Text', predicted_class_label);
                        set(app.NumClassDistanceLabel, 'Text', sprintf("%.4f", class_distance));
                        set(app.NumImagesClassifiedLabel, 'Text', sprintf("%d", app.images_tested));
                        set(app.NumImagesClassifiedCorrectlyLabel, 'Text', sprintf("%d", app.images_correct));
                        set(app.NumCurrentAccuracyLabel, 'Text', sprintf("%.4f", app.images_correct / app.images_tested));
                        drawnow;
                        java.lang.Thread.sleep(1000 * app.DelayTimeoutDisplaySlider.Value);
                        
                    end
                end
                
                
            end
        end
    end

    % Component initialization
    methods (Access = private)

        % Create UIFigure and components
        function createComponents(app)

            % Create UIFigure and hide until all components are created
            app.UIFigure = uifigure('Visible', 'off');
            app.UIFigure.Position = [100 100 1250 715];
            app.UIFigure.Name = 'MATLAB App';

            % Create GridLayout
            app.GridLayout = uigridlayout(app.UIFigure);
            app.GridLayout.ColumnWidth = {'1x', '1x', '1x', '1x', '1x', '1x'};
            app.GridLayout.RowHeight = {'1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x', '1x'};

            % Create LoadClassesButton
            app.LoadClassesButton = uibutton(app.GridLayout, 'push');
            app.LoadClassesButton.ButtonPushedFcn = createCallbackFcn(app, @LoadClasses, true);
            app.LoadClassesButton.FontWeight = 'bold';
            app.LoadClassesButton.Layout.Row = 9;
            app.LoadClassesButton.Layout.Column = [1 2];
            app.LoadClassesButton.Text = 'Load Classes';

            % Create TestImageLabel
            app.TestImageLabel = uilabel(app.GridLayout);
            app.TestImageLabel.HorizontalAlignment = 'center';
            app.TestImageLabel.FontWeight = 'bold';
            app.TestImageLabel.Layout.Row = 2;
            app.TestImageLabel.Layout.Column = [1 2];
            app.TestImageLabel.Text = 'Test Image';

            % Create PredictedImageLabel
            app.PredictedImageLabel = uilabel(app.GridLayout);
            app.PredictedImageLabel.HorizontalAlignment = 'center';
            app.PredictedImageLabel.FontWeight = 'bold';
            app.PredictedImageLabel.Layout.Row = 2;
            app.PredictedImageLabel.Layout.Column = [5 6];
            app.PredictedImageLabel.Text = 'Predicted Image';

            % Create AppTitle
            app.AppTitle = uilabel(app.GridLayout);
            app.AppTitle.HorizontalAlignment = 'center';
            app.AppTitle.FontSize = 16;
            app.AppTitle.FontWeight = 'bold';
            app.AppTitle.Layout.Row = 1;
            app.AppTitle.Layout.Column = [3 4];
            app.AppTitle.Text = 'LRC for Face Recognition & Classification';

            % Create ClassesLoadedLabel
            app.ClassesLoadedLabel = uilabel(app.GridLayout);
            app.ClassesLoadedLabel.HorizontalAlignment = 'right';
            app.ClassesLoadedLabel.FontWeight = 'bold';
            app.ClassesLoadedLabel.Layout.Row = 10;
            app.ClassesLoadedLabel.Layout.Column = 1;
            app.ClassesLoadedLabel.Text = 'Classes Loaded:';

            % Create NumClassesLoadedLabel
            app.NumClassesLoadedLabel = uilabel(app.GridLayout);
            app.NumClassesLoadedLabel.Layout.Row = 10;
            app.NumClassesLoadedLabel.Layout.Column = 2;
            app.NumClassesLoadedLabel.Text = '0';

            % Create ImagesLoadedLabel
            app.ImagesLoadedLabel = uilabel(app.GridLayout);
            app.ImagesLoadedLabel.HorizontalAlignment = 'right';
            app.ImagesLoadedLabel.FontWeight = 'bold';
            app.ImagesLoadedLabel.Layout.Row = 11;
            app.ImagesLoadedLabel.Layout.Column = 1;
            app.ImagesLoadedLabel.Text = 'Images Loaded:';

            % Create NumImagesLoadedLabel
            app.NumImagesLoadedLabel = uilabel(app.GridLayout);
            app.NumImagesLoadedLabel.Layout.Row = 11;
            app.NumImagesLoadedLabel.Layout.Column = 2;
            app.NumImagesLoadedLabel.Text = '0';

            % Create TrainingImagesLabel
            app.TrainingImagesLabel = uilabel(app.GridLayout);
            app.TrainingImagesLabel.HorizontalAlignment = 'right';
            app.TrainingImagesLabel.FontWeight = 'bold';
            app.TrainingImagesLabel.Layout.Row = 5;
            app.TrainingImagesLabel.Layout.Column = 3;
            app.TrainingImagesLabel.Text = 'Number of Training Images:';

            % Create TestImagesLabel
            app.TestImagesLabel = uilabel(app.GridLayout);
            app.TestImagesLabel.HorizontalAlignment = 'right';
            app.TestImagesLabel.FontWeight = 'bold';
            app.TestImagesLabel.Layout.Row = 6;
            app.TestImagesLabel.Layout.Column = 3;
            app.TestImagesLabel.Text = 'Number of Test Images:';

            % Create NumTrainingImagesLabel
            app.NumTrainingImagesLabel = uilabel(app.GridLayout);
            app.NumTrainingImagesLabel.Layout.Row = 5;
            app.NumTrainingImagesLabel.Layout.Column = 4;
            app.NumTrainingImagesLabel.Text = '0';

            % Create NumTestImagesLabel
            app.NumTestImagesLabel = uilabel(app.GridLayout);
            app.NumTestImagesLabel.Layout.Row = 6;
            app.NumTestImagesLabel.Layout.Column = 4;
            app.NumTestImagesLabel.Text = '0';

            % Create TrainingImagesPerClassLabel
            app.TrainingImagesPerClassLabel = uilabel(app.GridLayout);
            app.TrainingImagesPerClassLabel.HorizontalAlignment = 'right';
            app.TrainingImagesPerClassLabel.Layout.Row = 4;
            app.TrainingImagesPerClassLabel.Layout.Column = 3;
            app.TrainingImagesPerClassLabel.Text = 'Training Images Per Class';

            % Create TrainingImagesPerClassSlider
            app.TrainingImagesPerClassSlider = uislider(app.GridLayout);
            app.TrainingImagesPerClassSlider.Limits = [1 9];
            app.TrainingImagesPerClassSlider.ValueChangingFcn = createCallbackFcn(app, @NumTrainingImagesChanging, true);
            app.TrainingImagesPerClassSlider.MinorTicks = [1 2 3 4 5 6 7 8 9 10];
            app.TrainingImagesPerClassSlider.Layout.Row = 4;
            app.TrainingImagesPerClassSlider.Layout.Column = 4;
            app.TrainingImagesPerClassSlider.Value = 5;

            % Create NumImageSizeLabel
            app.NumImageSizeLabel = uilabel(app.GridLayout);
            app.NumImageSizeLabel.Layout.Row = 12;
            app.NumImageSizeLabel.Layout.Column = 2;
            app.NumImageSizeLabel.Text = '0 x 0';

            % Create ImageSizeLabel
            app.ImageSizeLabel = uilabel(app.GridLayout);
            app.ImageSizeLabel.HorizontalAlignment = 'right';
            app.ImageSizeLabel.FontWeight = 'bold';
            app.ImageSizeLabel.Layout.Row = 12;
            app.ImageSizeLabel.Layout.Column = 1;
            app.ImageSizeLabel.Text = 'Image Size:';

            % Create DownsampledImageSizeLabel
            app.DownsampledImageSizeLabel = uilabel(app.GridLayout);
            app.DownsampledImageSizeLabel.HorizontalAlignment = 'right';
            app.DownsampledImageSizeLabel.FontWeight = 'bold';
            app.DownsampledImageSizeLabel.Layout.Row = 12;
            app.DownsampledImageSizeLabel.Layout.Column = 3;
            app.DownsampledImageSizeLabel.Text = 'Downsampled Image Size:';

            % Create NumDownsampledImageSizeLabel
            app.NumDownsampledImageSizeLabel = uilabel(app.GridLayout);
            app.NumDownsampledImageSizeLabel.Layout.Row = 12;
            app.NumDownsampledImageSizeLabel.Layout.Column = 4;
            app.NumDownsampledImageSizeLabel.Text = '0 x 0';

            % Create ClassifyTestImagesButton
            app.ClassifyTestImagesButton = uibutton(app.GridLayout, 'push');
            app.ClassifyTestImagesButton.ButtonPushedFcn = createCallbackFcn(app, @ClassifyTestImages, true);
            app.ClassifyTestImagesButton.FontWeight = 'bold';
            app.ClassifyTestImagesButton.Layout.Row = 9;
            app.ClassifyTestImagesButton.Layout.Column = [3 4];
            app.ClassifyTestImagesButton.Text = 'Classify Test Images';

            % Create AssignedClassLabel
            app.AssignedClassLabel = uilabel(app.GridLayout);
            app.AssignedClassLabel.HorizontalAlignment = 'right';
            app.AssignedClassLabel.FontWeight = 'bold';
            app.AssignedClassLabel.Layout.Row = 10;
            app.AssignedClassLabel.Layout.Column = 5;
            app.AssignedClassLabel.Text = 'Assigned Class:';

            % Create ValueAssignedClassLabel
            app.ValueAssignedClassLabel = uilabel(app.GridLayout);
            app.ValueAssignedClassLabel.Layout.Row = 10;
            app.ValueAssignedClassLabel.Layout.Column = 6;
            app.ValueAssignedClassLabel.Text = '---';

            % Create ClassDistanceLabel
            app.ClassDistanceLabel = uilabel(app.GridLayout);
            app.ClassDistanceLabel.HorizontalAlignment = 'right';
            app.ClassDistanceLabel.FontWeight = 'bold';
            app.ClassDistanceLabel.Layout.Row = 11;
            app.ClassDistanceLabel.Layout.Column = 5;
            app.ClassDistanceLabel.Text = 'Class Distance:';

            % Create NumClassDistanceLabel
            app.NumClassDistanceLabel = uilabel(app.GridLayout);
            app.NumClassDistanceLabel.Layout.Row = 11;
            app.NumClassDistanceLabel.Layout.Column = 6;
            app.NumClassDistanceLabel.Text = '0';

            % Create CurrentAccuracyLabel
            app.CurrentAccuracyLabel = uilabel(app.GridLayout);
            app.CurrentAccuracyLabel.HorizontalAlignment = 'right';
            app.CurrentAccuracyLabel.FontWeight = 'bold';
            app.CurrentAccuracyLabel.Layout.Row = 12;
            app.CurrentAccuracyLabel.Layout.Column = 5;
            app.CurrentAccuracyLabel.Text = 'Current Accuracy:';

            % Create NumCurrentAccuracyLabel
            app.NumCurrentAccuracyLabel = uilabel(app.GridLayout);
            app.NumCurrentAccuracyLabel.Layout.Row = 12;
            app.NumCurrentAccuracyLabel.Layout.Column = 6;
            app.NumCurrentAccuracyLabel.Text = '0';

            % Create ImagesClassifiedCorrectlyLabel
            app.ImagesClassifiedCorrectlyLabel = uilabel(app.GridLayout);
            app.ImagesClassifiedCorrectlyLabel.HorizontalAlignment = 'right';
            app.ImagesClassifiedCorrectlyLabel.FontWeight = 'bold';
            app.ImagesClassifiedCorrectlyLabel.Layout.Row = 11;
            app.ImagesClassifiedCorrectlyLabel.Layout.Column = 3;
            app.ImagesClassifiedCorrectlyLabel.Text = 'Images Classified Correctly:';

            % Create NumImagesClassifiedCorrectlyLabel
            app.NumImagesClassifiedCorrectlyLabel = uilabel(app.GridLayout);
            app.NumImagesClassifiedCorrectlyLabel.Layout.Row = 11;
            app.NumImagesClassifiedCorrectlyLabel.Layout.Column = 4;
            app.NumImagesClassifiedCorrectlyLabel.Text = '0';

            % Create ImagesClassifiedLabel
            app.ImagesClassifiedLabel = uilabel(app.GridLayout);
            app.ImagesClassifiedLabel.HorizontalAlignment = 'right';
            app.ImagesClassifiedLabel.FontWeight = 'bold';
            app.ImagesClassifiedLabel.Layout.Row = 10;
            app.ImagesClassifiedLabel.Layout.Column = 3;
            app.ImagesClassifiedLabel.Text = 'Images Classified:';

            % Create NumImagesClassifiedLabel
            app.NumImagesClassifiedLabel = uilabel(app.GridLayout);
            app.NumImagesClassifiedLabel.Layout.Row = 10;
            app.NumImagesClassifiedLabel.Layout.Column = 4;
            app.NumImagesClassifiedLabel.Text = '0';

            % Create DelayTimeoutforDisplaySecondsLabel
            app.DelayTimeoutforDisplaySecondsLabel = uilabel(app.GridLayout);
            app.DelayTimeoutforDisplaySecondsLabel.HorizontalAlignment = 'right';
            app.DelayTimeoutforDisplaySecondsLabel.Layout.Row = 8;
            app.DelayTimeoutforDisplaySecondsLabel.Layout.Column = 3;
            app.DelayTimeoutforDisplaySecondsLabel.Text = 'Delay Timeout for Display (Seconds)';

            % Create DelayTimeoutDisplaySlider
            app.DelayTimeoutDisplaySlider = uislider(app.GridLayout);
            app.DelayTimeoutDisplaySlider.Limits = [0 2];
            app.DelayTimeoutDisplaySlider.Layout.Row = 8;
            app.DelayTimeoutDisplaySlider.Layout.Column = 4;
            app.DelayTimeoutDisplaySlider.Value = 1;

            % Create ActualClassLabel
            app.ActualClassLabel = uilabel(app.GridLayout);
            app.ActualClassLabel.HorizontalAlignment = 'right';
            app.ActualClassLabel.FontWeight = 'bold';
            app.ActualClassLabel.Layout.Row = 9;
            app.ActualClassLabel.Layout.Column = 5;
            app.ActualClassLabel.Text = 'Actual Class:';

            % Create ValueActualClassLabel
            app.ValueActualClassLabel = uilabel(app.GridLayout);
            app.ValueActualClassLabel.Layout.Row = 9;
            app.ValueActualClassLabel.Layout.Column = 6;
            app.ValueActualClassLabel.Text = '---';

            % Create PredictedImageDisplay
            app.PredictedImageDisplay = uiaxes(app.GridLayout);
            app.PredictedImageDisplay.XTick = [];
            app.PredictedImageDisplay.XTickLabel = '';
            app.PredictedImageDisplay.YTick = [];
            app.PredictedImageDisplay.Box = 'on';
            app.PredictedImageDisplay.Layout.Row = [3 8];
            app.PredictedImageDisplay.Layout.Column = [5 6];

            % Create InputTestImageDisplay
            app.InputTestImageDisplay = uiaxes(app.GridLayout);
            app.InputTestImageDisplay.XTick = [];
            app.InputTestImageDisplay.XTickLabel = '';
            app.InputTestImageDisplay.YTick = [];
            app.InputTestImageDisplay.Box = 'on';
            app.InputTestImageDisplay.Layout.Row = [3 8];
            app.InputTestImageDisplay.Layout.Column = [1 2];

            % Show the figure after all components are created
            app.UIFigure.Visible = 'on';
        end
    end

    % App creation and deletion
    methods (Access = public)

        % Construct app
        function app = project_app_exported

            % Create UIFigure and components
            createComponents(app)

            % Register the app with App Designer
            registerApp(app, app.UIFigure)

            if nargout == 0
                clear app
            end
        end

        % Code that executes before app deletion
        function delete(app)

            % Delete UIFigure when app is deleted
            delete(app.UIFigure)
        end
    end
end