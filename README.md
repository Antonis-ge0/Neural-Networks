# Neural-Networks
### **Implementation of Python program with SMV and  CNN+MLP for CIFAR − 10 classification (with backprop)**
## **1. Python program with CNN +MLP for CIFAR − 10 classification (with backprop)**
* Loads the CIFAR − 10 from the pickled files we have. [https://www.cs.toronto.edu/~kriz/cifar.html] (You can find the CIFAR-10 files here and the code to "unpickle" them)
* Defines a convolutional network (CNN) that is finally connected to a fully connected layer (MLP). The size of the hidden layer of the MLP is parametric, so we can compare performance for different numbers of neurons.
* Trains with back−prop, records training/testaccuracy at each epoch, and measures training time.
* Stores and shows examples of correct and incorrect classification (images + predicted/actual label).
* It implements comparison with classical algorithms: k−NearestNeighbors(kNN) and NearestClassCentroid(NCC) using features (e.g.rawpixels or PCA).
* It sweeps on parameters (number of neurons in the hidden layer, learningrate, batchsize) and outputs a table of results.
* Comments in Greek.

### **Instructions**
* Install dependencies (recommended environment Python 3.8+):
```
$ pip install torch torchvision numpy scikit-learn matplotlib tqdm pillow
```
* Import CIFAR files (the pickled ones) into a folder:
```
data/cifar-10-batches-py/.
```
* We run the code:
```
python cifar cnn mlp.py –data dir data/cifar-10-batches-py –epochs 30
```
* The results (plots, CSV with experiments, images of incorrect/correct classifications) will be saved in the folder results/.

### **What each part does (short explanation)**
1. Loading CIFAR: The unpickle function reads the pickled files and converts them to numpy arrays. X are transformed from (N, 3, 32, 32) to float 0..1.
2. Dataset / DataLoader: CIFAR Dataset and DataLoader provide batches for PyTorch.
3. Model: SimpleCNN MLP has 3 conv blocks (Conv → BatchNorm → ReLU → MaxPool). Features are flattened and passed to an MLP: Linear(flat dim → mlp hidden) → dropout → Linear(mlp hidden → num classes).
4. T raining loop: The train epoch and eval model implement training/eval. Loss and accuracy are recorded per epoch. eval model(..., save examples = True) saves misclassifications as image files in the folder results/misclassif ied.
5. Classical algorithms: run_classical runs kNN (k = 3) and Nearest Centroid on PCA−reduced features (pca dim = 100) for fair comparison. Returns accuracy and training time.

Example:
```
$ python cif ar cnn mlp.py –data dir data/cif ar-10-batches−py –results dir ./results –epochs 25 –batch size 128 –lr 0.001 –mlp hidden 128 256 512
```
This will run three experiments (mlp hidden = 128, 256, 512). It will save accuracy plots, images of misclassifications, and summary_results.csv in the results folder . At the end, the kNN/Nearest Centroid results (with PCA 100) will also be printed.

## **2. Python program with SMV for CIFAR − 10 classification**
* Loads CIFAR − 10 from the ”pickled” files we have. [https://www.cs.toronto.edu/~kriz/cifar.html] (You can find the CIFAR-10 files here and the code to "unpickle" them)
* Optional PCA to retain > 90% of the information
* Trains SVM (selection for different kernels), gives time measurement and performance estimate (train/test)
* Compares 1-NN, 3-NN, Nearest Class Centroid (NCC) and MLP (with a hidden layer) using Hinge loss
* Displays examples of correct/incorrect classification, confusion matrix, classification report, and saves results to a JSON file.
* Allows parameter selection. For more information about the parameters we can run the following:
```
$ python cifar10 svm experiments.py -h
```
* Comments in Greek.

### **Instructions**
* Install dependencies (recommended environment Python 3.8+)
* Import CIFAR files (pickled) into a folder:
```
./cifar-10-batches-py/.
```
* Run the code (indicative):
```
python cif ar10 svm and experiments.py –data dir ./cif ar-10-batchespy –pca variance 0.90 –output dir ./cif ar results
```

### **What each part does (short explanation)**
1. SVM: we use LinearSVC for linear (faster) and SVC for non-linear kernels (rbf, poly). We consider small grids of parameters (C, gamma, degree) and choose the best one in the test set. Larger grids/cross − validation are possible but cost time.
2. PCA: by default we keep 90% of the variance (you can change –pca variance). This reduces the cost of SVM and improves compact representation.
3. k − NN: 1 − NN and 3 − NN with sklearn (we measure accuracy and fitting time ).
4. NCC: we calculate the centroid of each class and predict based on the nearest distance (fast).
5. MLP with hinge: if PyTorch exists, we train an MLP (with a hidden layer).

<ins>Examples:</ins>
Simple execution example:
```
$ python cifar10 svm experiments.py –data dir ./cifar-10-batchespy –pca 0.90 –subsample 0.2 –output dir ./cifar results
```
Example for binary problem (e.g. airplane vs rest):
```
$ python cifar10 svm experiments.py –data dir ./cifar-10-batchespy –binary 0 –pca 0.90 –subsample 0.2 –output dir ./cifar results airplane vs rest
```
Example for two classes:
```
$ python cifar10 svm experiments.py –data dir ./cifar-10-batchespy –binary 1,3 –pca 0.90 –output dir ./cifar results
```
