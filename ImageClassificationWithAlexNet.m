
%% Overview 
% A Convolutional Neural Network (CNN) is a powerful machine learning
% technique from the field of deep learning. CNNs are trained using large
% collections of diverse images. From these large collections, CNNs can
% learn rich feature representations for a wide range of images. 
%
% In this example, images from Caltech 101 are classified into categories
% using a pretrained AlexNet model.
%
% Note: This example requires Deep Learning Toolbox, Statistics and
% Machine Learning Toolbox(TM).


%% Download Image Data
% Download the compressed data set from the following location
url = 'http://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz';

% Store the output in a folder
outputFolder = fullfile(tempdir, 'caltech101'); % define output folder

if ~exist(outputFolder, 'dir') % download only once
    disp('Downloading 126MB Caltech101 data set...');
    untar(url, outputFolder);
end

%% Load Images
rootFolder = fullfile(outputFolder, '101_ObjectCategories');

% Create an |ImageDatastore| to help you manage the data. 
imds = imageDatastore(rootFolder, 'IncludeSubfolders',true,...
    'LabelSource', 'foldernames');

%% Label Mapping
% |imds| variable now contains the images and the category labels
% associated with each image. The labels are automatically assigned from
% the folder names of the image files. Use |countEachLabel| to summarize
% the number of images per category.
tbl = countEachLabel(imds);
%% Handle Classes Imbalance
% Because |imds| above contains an unequal number of images per category,
% let's first adjust it, so that the number of images in the training set
% is balanced. Use splitEachLabel method to trim the set.

%% Determine the smallest amount of images in a category
%minSetCount = min(tbl{:,2}); 
%imds = splitEachLabel(imds, minSetCount, 'randomize');

%tbl = countEachLabel(imds);
% Notice that each set now has exactly the same number of images.
%countEachLabel(imds)

%% Load pretrained Network
net = alexnet();

% The first layer defines the input dimensions. Each CNN has a different
% input size requirements. The one used in this example requires image
% input that is 227-by-227-by-3.

net.Layers(1).InputSize

%%
% The intermediate layers make up the bulk of the CNN. These are a series
% of convolutional layers, interspersed with rectified linear units (ReLU)
% and max-pooling layers. Following the these layers are 3
% fully-connected layers.
%
% The final layer is the classification layer and its properties depend on
% the classification task. In this example, the CNN model that was loaded
% was trained to solve a 1000-way classification problem. Thus the
% classification layer has 1000 classes from the ImageNet dataset. 

net.Layers(end)

% Check the number of class names for classification task
numel(net.Layers(end).ClassNames)

%% Prepare Training and Test Image Sets
% Split the sets into training, testing and validation data. Pick 60% of
% images for the training and the remaining 40% for testing and validation
% data. Randomize the split to avoid biasing the results. 
% All the data partitions will be processed by the CNN model.

[imdsTrain, imdsValidation] = splitEachLabel(imds,0.8,'randomized');
[trainingSet, testSet] = splitEachLabel(imdsTrain, 0.6, 'randomize');

%% Display Images
% Display Training Images

%train_imgs = readall(trainingSet);
%for i = 1:length(train_imgs)
%    figure
%    imshow(train_imgs{i})
%end

%%
% Display Test Images

%test_imgs = readall(testSet);
%for i = 1:length(test_imgs)
%    figure
%    imshow(test_imgs{i})
%end

%% Pre-process Images For CNN
% As mentioned earlier, |net| can only process RGB images that are
% 224-by-224. To avoid re-saving all the images in Caltech 101 to this
% format, use an |augmentedImageDatastore| to resize and convert any
% grayscale images to RGB on-the-fly. The |augmentedImageDatastore| can be
% used for additional data augmentation as well when used for network
% training.

% Create augmentedImageDatastore to convert any grayscale images to RGB
% on-the-fly from training, test and validation sets and also resize the 
% images to the size required by the network.

imageSize = net.Layers(1).InputSize;
augmentedTrainingSet = augmentedImageDatastore(imageSize, trainingSet, ...
    'ColorPreprocessing', 'gray2rgb');
augmentedTestSet = augmentedImageDatastore(imageSize, testSet, ...
    'ColorPreprocessing', 'gray2rgb');
augmentedValidationSet = augmentedImageDatastore(imageSize, ...
    imdsValidation, 'ColorPreprocessing', 'gray2rgb');

% You can easily extract features from one of the deeper layers using the
% |activations| method. Selecting which of the deep layers to choose is a
% design choice, but typically starting with the layer right before the
% classification layer is a good place to start. In |net|, this layer
% is named 'fc1000'. Let's extract training features using that layer.

featureLayer = 'fc1000';
layersTransfer = net.Layers(1:end-3);
numClasses = numel(unique(trainingSet.Labels));
layers = [
    layersTransfer
    fullyConnectedLayer(numClasses,'WeightLearnRateFactor',20,...
    'BiasLearnRateFactor',20)
    softmaxLayer
    classificationLayer];

options = trainingOptions('sgdm', 'MiniBatchSize',10, 'MaxEpochs',6, ...
    'InitialLearnRate',1e-4, 'ValidationFrequency',3, ...
    'ValidationData',augmentedValidationSet, ...
    'ValidationPatience',Inf,'Verbose',false, ...
    'Plots','training-progress');

netTransfer = trainNetwork(augmentedTrainingSet,layers,options);
[YPred,scores] = classify(netTransfer, augmentedTestSet);
accuracy = mean(YPred == testSet.Labels);
fprintf('Test Accuracy = %f', accuracy);


