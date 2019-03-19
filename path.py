import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.lines as mlines
from matplotlib.patches import Circle

# Number of nodes
n_observations = 10
fig, ax = plt.subplots(1, 1)

# Coordinates of nodes
xs = np.array([ 1.1,  1.2,  2.7,  3.2,  4.1,  
  4.8,  6.1,  7.5,  9.5,  9.8 ])
ys = np.array([ 0.6,  2.5,  2.6,  3.8,  8.0,  
  5.9,  7.5,  5.5,  7.0,  9.8 ])
# Weights for coverage
ws = np.ones(n_observations)
TW = tf.placeholder(tf.float32)
coverage = 1.0

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Define Hypothesis with random/specific
#Y_pred = tf.Variable(tf.random_normal([1]), name='bias')
Y_pred = tf.Variable([-0.00029278], name='bias')
for pow_i in range(1, 5):
#    W = tf.Variable(tf.random_normal([1]), name='weight_%d' % pow_i)
    if pow_i == 1:
        W = tf.Variable([ 0.46918184], name='weight_%d' % pow_i)
    if pow_i == 2:
        W = tf.Variable([ 1.00643885], name='weight_%d' % pow_i)
    if pow_i == 3:
        W = tf.Variable([-0.23585591], name='weight_%d' % pow_i)
    if pow_i == 4:
        W = tf.Variable([ 0.01418969], name='weight_%d' % pow_i)

    Y_pred = tf.add(tf.multiply(tf.pow(X, pow_i), W), Y_pred)


# Loss function : initialize cost
cost = tf.reduce_sum(tf.multiply(tf.pow(Y_pred - Y, 2), TW))  /  (n_observations - 1)

sess = tf.Session()

cx = xs[3]
cy = ys[3]

learning_rate = 0.1e-8
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

n_epochs = 1
sess.run(tf.global_variables_initializer())
prev_training_cost = 0.0
for epoch_i in range(n_epochs):
    for (x, y, w) in zip(xs, ys, ws):
        sess.run(optimizer, feed_dict={X: x, Y: y, TW: w})

    training_cost = sess.run(
        cost, feed_dict={X: xs, Y: ys, TW: ws})
    print(str(epoch_i) + " : " + str(training_cost))

    if epoch_i % 50 == 0 and epoch_i > 100:

        # Range of Sensor
        normal = np.abs( Y_pred.eval(feed_dict={X: testx}, session=sess) - testy ) > coverage
        if True not in normal:
            break
        ws[normal] *= 1.1

    prev_training_cost = training_cost

xss = np.linspace(0, 10, 1000)

ax.plot(xss, Y_pred.eval(feed_dict={X: xss}, session=sess), color='red', label='CWP')
ax.plot([0,cx], [0, cy], color='blue', linestyle='dashed', label='General Path')
ax.plot(np.concatenate([[cx], xs[3:]]), np.concatenate([[cy], ys[3:]]), color='blue', linestyle='dashed')

# Print
hi = [0,0,0,0,0]
var = [v for v in tf.trainable_variables() if v.name == "bias:0"][0]
hi[4] = np.float32(sess.run(var))
for pow_i in range(1, 5):
    var = [v for v in tf.trainable_variables() if v.name == "weight_"+str(pow_i)+":0"][0]
    hi[(pow_i)-1] = np.float32(sess.run(var))
    print("Weight" + str(pow_i) + " : " + str(hi[(pow_i)-1]))
print("bias : " + str(hi[4]))


fig.show()
plt.draw()
ax.scatter(0.1,0.15, color='black', marker='^', label='H-AP', s=matplotlib.rcParams['lines.markersize'] ** 2 + 50, zorder=10)
ax.scatter(xs[:3], ys[:3], color='black', label='HGN', s=matplotlib.rcParams['lines.markersize'] ** 2 + 10, zorder=10)
ax.scatter(xs[3:], ys[3:], color='black', marker='*', s=matplotlib.rcParams['lines.markersize'] ** 2 + 50, label='LGN', zorder=10)


major_ticks = np.arange(0, 11, 1)
ax.set_xticks(major_ticks)
ax.set_yticks(major_ticks)
ax.grid(which='both')
ax.grid(which='minor', alpha=0.2)
ax.grid(which='major', alpha=0.5)
ax.set_ylim([0, 10])
ax.set_xlim([0, 10])
plt.xlabel('x(m)')
plt.ylabel('y(m)')
plt.legend(loc='lower right', fontsize='small', framealpha=1)
plt.grid(ls='dashed')
plt.savefig('test.eps', format='eps', dpi=1000)
plt.waitforbuttonpress()