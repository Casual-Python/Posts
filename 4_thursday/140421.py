import tensorflow as tf
import matplotlib.pyplot as plt

x = tf.linspace(-3,2,100)
delta = tf.Variable(0.0,dtype=tf.double)
with tf.GradientTape() as tape1:
    with tf.GradientTape() as tape2: 
        x = x + delta
        f = tf.maximum(x,0)/x*x**3 + tf.minimum(x,0)/x*tf.sin(math.pi * x)
    d_f = tape2.jacobian(f, delta)
d2_f = tape1.jacobian(d_f, delta)

plt.plot(x,f,label='Function')
plt.plot(x,d_f,label='First derivative')
plt.plot(x,d2_f,label='Second derivative')
plt.legend()
plt.show()
