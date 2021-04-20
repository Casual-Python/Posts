import tensorflow as tf
import timeit
import tensorboard
import datetime

x = tf.random.uniform(shape=[10, 10], minval=-2, maxval=5, dtype=tf.dtypes.int32)

def power(x):
    result = tf.eye(10, dtype=tf.dtypes.int32)
    for _ in range(10):
        for _ in range(10):
            result = tf.matmul(x, result)
        x = x + 1
    return result
  
print("Eager execution:", timeit.timeit(lambda: power(x), number=1000))
print("Eager execution:", timeit.timeit(lambda: power(x), number=1000))
print("Eager execution:", timeit.timeit(lambda: power(x), number=1000))

res1 = power(x)

@tf.function
def power(x):
    result = tf.eye(10, dtype=tf.dtypes.int32)
    for _ in range(10):
        for _ in range(10):
            result = tf.matmul(x, result)
        x = x + 1
    return result
  
print("Graph execution:", timeit.timeit(lambda: power(x), number=1000))
print("Graph execution:", timeit.timeit(lambda: power(x), number=1000))
print("Graph execution:", timeit.timeit(lambda: power(x), number=1000))

res2 = power(x)

print(res1 == res2)

stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = 'logs/func/%s' % stamp
writer = tf.summary.create_file_writer(logdir)
tf.summary.trace_on(graph=True, profiler=True)

x = tf.random.uniform(shape=[10, 10], minval=-2, maxval=5, dtype=tf.dtypes.int32)

@tf.function
def power(x):
    result = tf.eye(10, dtype=tf.dtypes.int32)
    for _ in range(10):
        for _ in range(10):
            result = tf.matmul(x, result)
        x = x + 1
    return result
  
res = power(x)

with writer.as_default():
    tf.summary.trace_export(
      name="my_func_trace",
      step=0,
      profiler_outdir=logdir)
    
# jupyter notebook

%load_ext tensorboard
%tensorboard --logdir logs/func

