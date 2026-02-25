# Neural-Networks
Implementation of Python program with SMV and  CNN+MLP for CIFAR − 10 classification (with backprop)
## **Python program with CNN +MLP for CIF AR − 10 classification (with backprop)**
* Loads the CIFAR − 10 from the pickled files we have. [https://www.cs.toronto.edu/~kriz/cifar.html] (You can find the CIFAR-10 files here and the code to "unpickle" them)
* Defines a convolutional network (CNN) that is finally connected to a fully connected layer (MLP). The size of the hidden layer of the MLP is parametric, so we can compare performance for different numbers of neurons.
* Trains with back−prop, records training/testaccuracy at each epoch, and measures training time.
* Stores and shows examples of correct and incorrect classification (images + predicted/actual label).
* It implements comparison with classical algorithms: k−NearestNeighbors(kNN) and NearestClassCentroid(NCC) using features (e.g.rawpixels or PCA).
* It sweeps on parameters (number of neurons in the hidden layer, learningrate, batchsize) and outputs a table of results.
* Comments in Greek.

## **Instructions**
* Install dependencies (recommended environment Python 3.8+):
$ pip install torch torchvision numpy scikit-learn matplotlib tqdm pillow
* Import CIFAR files (the pickled ones) into a folder:
data/cif ar-10-batches-py/.
* We run the code:
python cif ar cnn mlp.py –data dir data/cif ar-10-batches-py –epochs 30
* The results (plots, CSV with experiments, images of incorrect/correct classifications) will be saved in the folder results/.
