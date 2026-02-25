# Neural-Networks
Implementation of Python program with SMV and  CNN+MLP for CIFAR − 10 classification (with backprop)
## **Python program with CNN +MLP for CIFAR − 10 classification (with backprop)**
* Loads the CIFAR − 10 from the pickled files we have. [https://www.cs.toronto.edu/~kriz/cifar.html] (You can find the CIFAR-10 files here and the code to "unpickle" them)
* Defines a convolutional network (CNN) that is finally connected to a fully connected layer (MLP). The size of the hidden layer of the MLP is parametric, so we can compare performance for different numbers of neurons.
* Trains with back−prop, records training/testaccuracy at each epoch, and measures training time.
* Stores and shows examples of correct and incorrect classification (images + predicted/actual label).
* It implements comparison with classical algorithms: k−NearestNeighbors(kNN) and NearestClassCentroid(NCC) using features (e.g.rawpixels or PCA).
* It sweeps on parameters (number of neurons in the hidden layer, learningrate, batchsize) and outputs a table of results.
* Comments in Greek.

## **Instructions**
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
