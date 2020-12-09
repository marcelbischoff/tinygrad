from tinygrad.tensor import Tensor

class BatchNorm2D:
  def __init__(self, sz, eps=1e-5, track_running_stats=False, training=True):
    self.eps = Tensor([eps], requires_grad=False)
    self.two = Tensor([2], requires_grad=False).reshape(shape=[1,1,1,1])
    self.weight = Tensor.ones(sz)
    self.bias = Tensor.zeros(sz)
    self.track_running_stats = track_running_stats
    self.training = training

    self.running_mean = Tensor.zeros(sz, requires_grad=False)
    self.running_var = Tensor.ones(sz, requires_grad=False)
    self.num_batches_tracked = Tensor.zeros(1, requires_grad=False)

  def __call__(self, x):
    if self.track_running_stats or self.training:
      batch_mean = x.mean(axis=(0,2,3))
      y = (x - batch_mean.reshape(shape=[1, -1, 1, 1])) 
      #batch_var = (y**self.two).mean(axis=(0,2,3)) #**self.two has issues and seems numerical unstable
      batch_var = (y.mul(y)).mean(axis=(0,2,3)) #**self.two has issues and seems numerical unstable
    if self.track_running_stats: #needs momentum
      self.running_mean = self.running_mean.mul(self.num_batches_tracked).add(batch_mean)
      self.running_var = self.running_var.mul(self.num_batches_tracked).add(batch_var)
      self.num_batches_tracked = self.num_batches_tracked.add(Tensor.ones(1, requires_grad=False))
      self.running_mean = self.running_mean.div(self.num_batches_tracked)
      self.running_var = self.running_var.div(self.num_batches_tracked)
    if self.training: 
      return self.normalize(x, batch_mean, batch_var)
    return self.normalize(x, self.running_mean, self.running_var)

  def normalize(self, x, mean, var):
    x = x.sub(mean.reshape(shape=[1, -1, 1, 1]))
    x = x.mul(self.weight.reshape(shape=[1, -1, 1, 1]))
    x = x.div(var.add(self.eps).reshape(shape=[1, -1, 1, 1]).sqrt())
    return x.add(self.bias.reshape(shape=[1, -1, 1, 1]))
