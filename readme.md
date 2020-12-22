## Face Recognition System


From the dataset provided to us our solution is divided into two parts as shown below  
Extraction of faces  
Extraction Eigen Faces   
Performing recognition on a trained neural network

#### Extraction of Faces:

Since in the dataset each image consists of the entire face including the background and the neck region. Hence to make a more precise model we only extract the faces from each of the folders in the probe and gallery. This is done in an iterative fashion using an open source implementation known as MTCNN.

#### Extraction of Eigen Faces:

From the faces we extracted, we apply PCA to reduce the dimensions of each image to value k selected by hyper parameter method, in doing so we generate a vector which retains the most essential properties of a face for each of the faces we extracted.
Hence, we also reduce the number of coefficients for the neural network to train on the dataset since the dimensions are reduced

#### Performing recognition on a trained neural network:

After we reduce the face images to eigen face vectors we then construct a neural network having 3 hidden layers with a softmax loss in the output in order to train the model to classify the faces for the given 26 classes.
The dataset is split into train and validation from the gallery images and the model is fine tuned based on these data and finally its performance is measured on the probe dataset


Following is the process to run the source code:

    #To simply run the network/model on the saved data
    python GenModel.py
    
    #To plot ROC and CMC curve on the saved model
    python Plots.py	


##### Additional Information:
1. Dataset is private
2. Pickel files are stored in pickle_files folder


##### Contributors:
Jasmeet Narang  
Shirish Mecheri Vogga
