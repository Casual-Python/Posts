import tensorflow as tf

class SimpleModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)
    self.a_variable = tf.Variable(5.0, name="train_me")
    self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")
  def __call__(self, x):
    return self.a_variable * x + self.non_trainable_variable
  
simple_module = SimpleModule(name="simple")
simple_module(tf.constant(5.0))
  
print("trainable variables:", simple_module.trainable_variables)
print("all variables:", simple_module.variables)
  
class Dense(tf.Module):
  def __init__(self, in_features, out_features, name=None):
    super().__init__(name=name)
    self.w = tf.Variable(
      tf.random.normal([in_features, out_features]), name='w')
    self.b = tf.Variable(tf.zeros([out_features]), name='b')
  def __call__(self, x):
    y = tf.matmul(x, self.w) + self.b
    return tf.nn.relu(y)
  
  
class SequentialModule(tf.Module):
  def __init__(self, name=None):
    super().__init__(name=name)

    self.dense_1 = Dense(in_features=3, out_features=3)
    self.dense_2 = Dense(in_features=3, out_features=2)

def __call__(self, x):
    x = self.dense_1(x)
    return self.dense_2(x)
  
my_model = SequentialModule(name="the_model")
  
print("Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))
  
print("Submodules:", my_model.submodules)
  
for var in my_model.variables:
  print(var, "\n")
  
chkp_path = "my_checkpoint"
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint.write(chkp_path)
  
tf.train.list_variables(chkp_path)

new_model = SequentialModule()
new_checkpoint = tf.train.Checkpoint(model=new_model)
new_checkpoint.restore("my_checkpoint")

new_model(tf.constant([[2.0, 2.0, 2.0]]))

tf.saved_model.save(my_model, "the_saved_model")

new_model = tf.saved_model.load("the_saved_model")

import tensorflow_hub as hub

embed = hub.load("https://tfhub.dev/google/nnlm-en-dim50-with-normalization/2")

import numpy as np
def cos_sim(e_w_1,e_w_2):
    return np.dot(e_w_1,e_w_2)\
    /np.linalg.norm(e_w_1)\
    /np.linalg.norm(e_w_2)
  
cos_sim(tf.reshape(embed(["king"]),-1),tf.reshape(embed(["queen"]),-1))

def g_w(name):
    return tf.reshape(embed([name]),-1)
  
cos_sim(g_w("king") - g_w("queen"),g_w("man") - g_w("woman"))

cos_sim(g_w("king") - g_w("man"),g_w("queen") - g_w("woman"))
  
  
  
