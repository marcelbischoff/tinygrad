import os
import time
import io
import numpy as np
from efficientnet_modified import EfficientNet
from tinygrad.tensor import Tensor
from tinygrad.utils import fetch
from PIL import Image
from tqdm import tqdm, trange
import tinygrad.optim as optim
GPU = os.getenv("GPU", None) is not None

#not tested completely, too many bad URLs: will run ages waiting for timeouts
def fetch_imagenet2():
  url0 = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02084071'
  url1 = 'http://image-net.org/api/text/imagenet.synset.geturls?wnid=n02121808'

  urls0 = fetch(url0).decode('utf-8').splitlines()
  urls1 = fetch(url1).decode('utf-8').splitlines()

  X = []
  Y = []
  for urls, Yn in [[urls0, 0], [urls1, 1]]:
    for url in urls:
      #print(url,Yn)
      try:
        img = Image.open(io.BytesIO(fetch(url)))
        aspect_ratio = img.size[0] / img.size[1]
        img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))
        img = np.array(img)
        y0,x0=(np.asarray(img.shape)[:2]-224)//2
      except:
        continue
      img = img[y0:y0+224, x0:x0+224]
      if img.shape != (3,224,224):
          continue
      # if you want to look at the image
      """ 
      import matplotlib.pyplot as plt
      plt.imshow(img)
      plt.show()
      """

      # low level preprocess
      img = np.moveaxis(img, [2,0,1], [0,1,2])
      img = img.astype(np.float32)[:3].reshape(3,224,224)
      img /= 255.0
      img -= np.array([0.485, 0.456, 0.406]).reshape((-1,1,1))
      img /= np.array([0.229, 0.224, 0.225]).reshape((-1,1,1))
      X.append(img)
      Y.append(Yn)
  X=np.array(X)
  Y=np.array(Y)
  from sklearn.model_selection import train_test_split
  i_train, i_test = train_test_split(range(len(X)), test_size=0.1, random_state=42)
  return X[i_train],Y[i_train],X[i_test],Y[i_test] 

#you need to download the dataset from kaggle and change the folder
def fetch_local(max_no = -1):
  from os import listdir
  #TODO: Download Dogs/Cats testset from kaggle
  folder = 'examples/train/' #/Users/marcel/Downloads/train/'
  photos, labels = list(), list()
  # enumerate files in the directory
  X, Y = [], []
  for file in tqdm(listdir(folder)):
    max_no-=1;
    if max_no == 0:
      break
    #print(file)
    # determine class
    Yn = 0
    if file.startswith('cat'):
        Yn = 1
    img = Image.open(folder+file)
    aspect_ratio = img.size[0] / img.size[1]
    img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))
    img = np.array(img)
    y0,x0=(np.asarray(img.shape)[:2]-224)//2
    img = img[y0:y0+224, x0:x0+224]
#    if img.shape != (3,224,224):
#      continue
    # if you want to look at the image
    """
    import matplotlib.pyplot as plt
    plt.imshow(img)
    plt.show()
    """

    # low level preprocess
    img = np.moveaxis(img, [2,0,1], [0,1,2])
    img = img.astype(np.float32)[:3].reshape(3,224,224)
    img /= 255.0
    img -= np.array([0.485, 0.456, 0.406]).reshape((-1,1,1))
    img /= np.array([0.229, 0.224, 0.225]).reshape((-1,1,1))
    X.append(img)
    Y.append(Yn)
  X=np.array(X)
  Y=np.array(Y)
  print(len(X))
  from sklearn.model_selection import train_test_split
  i_train, i_test = train_test_split(range(len(X)), test_size=0.1, random_state=42)
  return X[i_train],Y[i_train],X[i_test],Y[i_test] 

def train(model, optim, steps, BS=64, gpu=False):
  losses, accuracies = [], []
  for i in (t := trange(steps, disable=os.getenv('CI') is not None)):
    optim.zero_grad()
    samp = np.random.randint(0, X_train.shape[0], size=(BS))

    x = Tensor(X_train[samp], gpu=gpu)
    Y = Y_train[samp]
    y = np.zeros((len(samp),2), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]),Y] = -2.0
    y = Tensor(y, gpu=gpu)

    # NLL loss function
    st = time.time()
    out = model.forward(x)
    et = time.time()
    print("forward %.2f s" % (et-st))

    loss = out.logsoftmax().mul(y).mean()

    st = time.time()
    loss.backward()
    et = time.time()
    print("backward %.2f s" % (et-st))
   
    optim.step()
  
    cat = np.argmax(out.cpu().data, axis=1)
    accuracy = (cat == Y).mean()

    # printing
    lost = np.array(loss.cpu().data)
    losses.append(lost)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (lost, accuracy))


def trainBL(model, optim, steps, BS=64, gpu=False):
  global data_train
  losses, accuracies = [], []
  t = trange(steps, disable=os.getenv('CI') is not None)
  for i in t:
    optim.zero_grad()
    samp = np.random.randint(0, len(data_train), size=(BS))
    
    X, Y = fetch_batch(data_train[samp])

    x = Tensor(X, gpu=gpu)
    y = np.zeros((len(samp),2), np.float32)
    # correct loss for NLL, torch NLL loss returns one per row
    y[range(y.shape[0]),Y] = -2.0
    y = Tensor(y, gpu=gpu)

    # NLL loss function
    st = time.time()
    out = model.forward(x)
    et = time.time()
    #print("forward %.2f s" % (et-st))

    loss = out.logsoftmax().mul(y).mean()

    st = time.time()
    loss.backward()
    et = time.time()
    #print("backward %.2f s" % (et-st))

    optim.step()
  
    cat = np.argmax(out.cpu().data, axis=1)
    accuracy = (cat == Y).mean()

    # printing
    lost = np.array(loss.cpu().data)
    losses.append(lost)
    accuracies.append(accuracy)
    t.set_description("loss %.2f accuracy %.2f" % (lost, accuracy))

def evaluate(model, gpu=False):
  def numpy_eval():
    Y_test_preds_out = model.forward(Tensor(X_test.reshape((-1, 28*28)).astype(np.float32), gpu=gpu)).cpu()
    Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
    return (Y_test == Y_test_preds).mean()

def evaluateBL(model, gpu=False):
  def numpy_eval():
    acc =[]
    for i in tqdm(range(len(data_test)//BS+1)):
        X, Y = fetch_batch(data_test[i:i+BS])
        x = Tensor(X, gpu=gpu)
        Y_test_preds_out = model.forward(x).cpu().data
        Y_test_preds = np.argmax(Y_test_preds_out.data, axis=1)
        acc.append( (Y == Y_test_preds).mean() )
    return np.array(acc).mean()

  accuracy = numpy_eval()
  print("test set accuracy is %f" % accuracy)

def fetch_batch(files):
  #TODO: Download Dogs/Cats testset from kaggle
  global folder
  photos, labels = list(), list()
  # enumerate files in the directory
  X, Y = [], []
  for file in files:
    Yn = 0
    if file.startswith('cat'):
        Yn = 1
    img = Image.open(folder+file)
    aspect_ratio = img.size[0] / img.size[1]
    img = img.resize((int(224*max(aspect_ratio,1.0)), int(224*max(1.0/aspect_ratio,1.0))))
    img = np.array(img)
    y0,x0=(np.asarray(img.shape)[:2]-224)//2
    img = img[y0:y0+224, x0:x0+224]

    # low level preprocess
    img = np.moveaxis(img, [2,0,1], [0,1,2])
    img = img.astype(np.float32)[:3].reshape(3,224,224)
    img /= 255.0
    img -= np.array([0.485, 0.456, 0.406]).reshape((-1,1,1))
    img /= np.array([0.229, 0.224, 0.225]).reshape((-1,1,1))
    assert img.shape == (3,224,224), img.shape
    X.append(img)
    Y.append(Yn)
  X=np.array(X)
  Y=np.array(Y)
  return X, Y





if __name__ == "__main__":
  Tensor.default_gpu = os.getenv("GPU") is not None
  folder = 'examples/train/'
  datafiles = np.array(os.listdir(folder))

  from sklearn.model_selection import train_test_split
  data_train, data_test = train_test_split(datafiles, test_size=0.1, random_state=42)
  BS = 4
  EPOCHS = 1
  STEPS = len(data_train)//BS
    

  model = EfficientNet(categories=2)
  print('Loading weights...')
  model.load_weights_from_torch(GPU)

  optimizer = optim.SGD(model.parameters(), lr=0.0003)
  
  print('Training model...')
  trainBL(model, optimizer, steps=STEPS, BS=BS, gpu=GPU)
  evaluateBL(model)



