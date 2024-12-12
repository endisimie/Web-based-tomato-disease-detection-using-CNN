"# AI Driven web based tomato disease detection system" 
the system can predict 10 disease types incluuding healthy one.

Datasets
We used Total of 10984 image datasets 80 % for training, 10% for validating and the remaining 10% for testing. This means we use 1000 data for training, 492 data for each validating and testing.
Evaluating our trained model using testing data and the result shows that our model correctly predicted 97% of unseen data’s (testing data). 

The dataset has 10 classes including healthy class: 
['Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']

 ![alt text](https://github.com/endisimie/Web-based-tomato-disease-detection-using-CNN/blob/main/Early_blight_of_tomato.jpeg?raw=true))
Image processing 
The image_dataset_from_directory function is used to load image data from a directory. Images are resized to 256x256 pixels and grouped into batches of 32 for efficient processing. Labels are inferred from the directory structure and represented in a categorical format. Pixel values of the images are normalized to a range of [0, 1] by dividing by 255.0 for consistent model training.

Methodology and Methods
The code initializes a DenseNet121 model with weights pre-trained on the ImageNet dataset. The include_top=False argument specifies that the model should not include the fully connected layers at the top of the network, which are typically used for ImageNet classification. The input_shape=(256,256,3) argument specifies the shape of the input images that the model expects, which is 256x256 pixels with 3 channels (RGB).
The initialized model conv_base can be used as a feature extractor to extract features from images, which can then be used as input to a custom classification head.

It initializes a Sequential model and adds the pre-trained DenseNet121model (conv_base) as the base. The model flattens the output from the base model and adds two fully connected layers with ReLU activation. Batch normalization is applied after each fully connected layer to improve training stability and speed.



