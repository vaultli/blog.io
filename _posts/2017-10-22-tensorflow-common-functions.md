---
layout: post
title: TensorFlow 常用函数汇总
date: 2017-10-22
categories: blog
tags: [TensorFlow,常用函数]
description: TensorFlow 使用学习，常用函数。
---

# 1. TensorFLow 简介

![TensorFlow](http://ow7l1fhke.bkt.clouddn.com/my_images/blog4.png "TensorFlow")

TensorFlow 是由 Google 公司发明的一款优秀的深度学习框架，它极大地方便了深度学习的研究和应用，并且社区良好，资料也很多，甚至有些框架还是以 TensorFlow 为后端的，例如 Keras、TFLearn 。由于一些众所周知的原因，TensorFlow 的官网不是很方便浏览，因此，此博客的目的是汇总一些常用函数，一方面当做复习，另一方面方便以后浏览查阅。

# 2. 常用函数

TensorFlow 的基本数据结构是张量 Tensor。0 维tensor（0-D tensor）是标量；1 维tensor（1-D tensor）是向量，可以默认为是行向量；2 维tensor（2-D tensor）是矩阵；N 维tensor（N-D tensor）是一般化的张量。这些可以跟 Numpy 数组 Array 类比，帮助理解。类比之前的 MiniFlow ，TensorFlow 在应用的时候也分为构图和喂数据两个过程，而 Tensor 在定义后并不能直接运行得结果。而需要建立会话 Session ，通过 `sess.run()` 得到运行结果。

TensorFlow 中的 Tensor 有很多特殊的种类，例如常量 tensor ，占位符 tensor ，可变 tensor 等，不同的 Tensor 需要利用不同的函数进行定义。TensorFlow 可以作为一个模块导入 Python 中，在 Python 中使用相应的函数。 TensorFlow 有 CPU 和 GPU 版，但是函数并没有太大区别。

## 简单实例

下面是一段简单代码，可以大概体会下 TensorFlow 在 Python 中的代码风格。

```Python
import tensorflow as tf

hello = tf.constant('Hello, world!') # 定义常量 tensor
with tf.Session() as sess:
    hello_world = sess.run(hello) # 运行常量 tensor
    print(hello_world)
```

输出结果为 `b'Hello, world!'` 。

## 特殊 Tensor 定义

```Python
tf.constant(
    value,
    dtype=None,
    shape=None,
    name='Const',
    verify_shape=False
)
```

创建一个常量 tensor (Constant tensor)。

参数 (Args):

* `value` : A constant value (or list) of output type dtype.
* `dtype` : The type of the elements of the resulting tensor.
* `shape` : Optional dimensions of resulting tensor.
* `name` : Optional name for the tensor.
* `verify_shape` : Boolean that enables verification of a shape of values.

返回 (Returns):

* A Constant Tensor.

---

```Python
tf.placeholder(
    dtype,
    shape=None,
    name=None
)
```

创建一个占位符 tensor (Placeholder tensor)，占位符 tensor 可以用来实现占位，之后用到的占位符 tensor 必须在 `sess.run()` 中的 `feed_dict` 参数中喂数据。参数 `name` 用来给占位符 tensor 命名，之后可以利用加载图的 `loaded_graph.get_tensor_by_name('<name>:0')` 函数通过名字得到指定的占位符 tensor 。其他函数的 `name` 参数具有类似的意义。

官方实例:

```Python
x = tf.placeholder(tf.float32, shape=(1024, 1024))
y = tf.matmul(x, x)

with tf.Session() as sess:
    print(sess.run(y))  # ERROR: will fail because x was not fed.

    rand_array = np.random.rand(1024, 1024)
    print(sess.run(y, feed_dict={x: rand_array}))  # Will succeed.
```

参数 (Args):

* `dtype` : The type of elements in the tensor to be fed.
* `shape` : The shape of the tensor to be fed (optional). If the shape is not specified, you can feed a tensor of any shape.
* `name` : A name for the operation (optional).

返回 (Returns):

* A Tensor that may be used as a handle for feeding a value, but not evaluated directly.

---

```Python
tf.Variable(
    initial_value=None,
    trainable=True,
    collections=None,
    validate_shape=True,
    caching_device=None,
    name=None,
    variable_def=None,
    dtype=None,
    expected_shape=None,
    import_scope=None
)
```

创建一个可变 tensor (Variable tensor)，`tf.Variable` 是一个类，`tf.Variable()` 是相应的构造函数，`initial-value` 参数可以对应的是某种概率分布，`name` 参数用于命名。可变 tensor 常用于定义网络中的可训练参数，并且在开始训练之前必须在 Session 中初始化。

实例:

```Python
w = tf.Variable(tf.truncated_normal((batch_size, feature_size), mean=0.0, stddev=0.1), name='weights')
y = tf.matmul(x, w)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op) # 初始化全局变量
```

参数 (Args):

* `<initial-value>` : Initial value.
* `name` : A name for the operation (optional).

返回 (Returns):

* A Variable Tensor.

## 张量操作

```Python
tf.global_variables_initializer()
```

---

```Python
tf.cast(
    x,
    dtype,
    name=None
)
```

---

```Python
tf.shape(
    input,
    name=None,
    out_type=tf.int32
)
```

---

```Python
tf.reshape(
    tensor,
    shape,
    name=None
)
```

---

```Python
tf.expand_dims(
    input,
    axis=None,
    name=None,
    dim=None
)
```

---

```Python
tf.concat(
    values,
    axis,
    name='concat'
)
```

---

```Python
tf.one_hot(
    indices,
    depth,
    on_value=None,
    off_value=None
    axis=None,
    dtype=None,
    name=None
)
```

---

```Python
tf.identity(
    input,
    name=None
)
```

---

```Python
tf.tile(
    input,
    multiples,
    name=None
)
```

## 算术操作

```Python
tf.add(
    x,
    y,
    name=None
)
```

---

```Python
tf.sub(
    x,
    y,
    name=None
)
```

---

```Python
tf.mul(
    x,
    y,
    name=None
)
```

---

```Python
tf.div(
    x,
    y,
    name=None
)
```

---

```Python
tf.mod(
    x,
    y,
    name=None
)
```

---

```Python
tf.abs(
    x,
    name=None
)
```

---

```Python
tf.neg(
    x,
    name=None
)
```

---

```Python
tf.sign(
    x,
    name=None
)
```

---

```Python
tf.inv(
    x,
    name=None
)
```

---

```Python
tf.square(
    x,
    name=None
)
```

---

```Python
tf.sqrt(
    x,
    name=None
)
```

---

```Python
tf.round(
    x,
    name=None
)
```

---

```Python
tf.pow(
    x,
    y,
    name=None
)
```

---

```Python
tf.exp(
    x,
    name=None
)
```

---

```Python
tf.log(
    x,
    name=None
)
```

---

```Python
tf.maximum(
    x,
    y,
    name=None
)
```

---

```Python
tf.minimum(
    x,
    y,
    name=None
)
```

---

```Python
tf.sin(
    x,
    name=None
)
```

---

```Python
tf.cos(
    x,
    name=None
)
```

---

```Python
tf.tan(
    x,
    name=None
)
```

---

```Python
tf.atan(
    x,
    name=None
)
```

## 矩阵相关操作

```Python
tf.matmul(
    a,
    b,
    transpose_a=False,
    transpose_b=False,
    a_is_sparse=False,
    b_is_sparse=False,
    name=None
)
```

---

```Python
tf.diag(
    diagonal,
    name=None
)
```

---

```Python
tf.diag_part(
    input,
    name=None
)
```

---

```Python
tf.trace(
    x,
    name=None
)
```

## 归约计算操作

```Python
tf.reduce_sum(
    input_tensor,
    reduction_indices=None,
    keep_dims=False,
    name=None
)
```

---

```Python
tf.reduce_mean(
    input_tensor,
    reduction_indices=None,
    keep_dims=False,
    name=None
)
```

---

```Python
tf.reduce_min(
    input_tensor,
    reduction_indices=None,
    keep_dims=False,
    name=None
)
```

---

```Python
tf.reduce_max(
    input_tensor,
    reduction_indices=None,
    keep_dims=False,
    name=None
)
```

## 序列比较与索引提取操作

```Python
tf.argmin(
    input,
    dimension,
    name=None
)
```

---

```Python
tf.argmax(
    input,
    dimension,
    name=None
)
```

---

```Python
tf.equal(
    x,
    y,
    name=None
)
```

## 概率分布相关操作

```Python

tf.truncated_normal(
    shape,
    mean=0.0,
    stddev=1.0,
    dtype=tf.float32,
    seed=None,
    name=None
)
```

---

```Python
tf.zeros(
    shape,
    dtype=tf.float32,
    name=None
)
```

---

```Python
tf.ones(
    shape,
    dtype=tf.float32,
    name=None
)
```

## 神经网络相关操作

```Python
tf.sigmoid(
    x,
    name=None
)
tf.nn.sigmoid
```

---

```Python
tf.tanh(
    x,
    name=None
)
tf.nn.tanh
```

---

```Python
tf.nn.relu(
    features,
    name=None
)
```

----

```Python
tf.nn.softmax(
    logits,
    dim=-1,
    name=None
)
```

---

```Python
tf.nn.dropout(
    x,
    keep_prob,
    noise_shape=None,
    seed=None,
    name=None
)
```

---

```Python
tf.nn.l2_normalize(
    x,
    dim,
    epsilon=1e-12,
    name=None
)
```

---

```Python
tf.contrib.layers.fully_connected(
    inputs,
    num_outputs,
    activation_fn=tf.nn.relu,
    normalizer_fn=None,
    normalizer_params=None,
    weights_initializer=initializers.xavier_initializer(),
    weights_regularizer=None,
    biases_initializer=tf.zeros_initializer(),
    biases_regularizer=None,
    reuse=None,
    variables_collections=None,
    outputs_collections=None,
    trainable=True,
    scope=None
)
```

---

```Python
tf.nn.conv2d(
    input,
    filter,
    strides,
    padding,
    use_cudnn_on_gpu=None,
    data_format=None,
    name=None
)
```

---

```Python
tf.nn.max_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)
```

---

```Python
tf.nn.avg_pool(
    value,
    ksize,
    strides,
    padding,
    data_format='NHWC',
    name=None
)
```

---

```Python
tf.nn.embedding_lookup(
    params,
    ids,
    partition_strategy='mod,
    name=None,
    validate_indices=True
)
```

---

```Python
tf.contrib.layers.embed_sequence(
    ids,
    vocab_size=None,
    embed_dim=None,
    unique=False,
    initializer=None,
    regularizer=None,
    trainable=True,
    scope=None,
    reuse=None

)
```

---

```Python
tf.contrib.rnn.BasicLSTMCell(
    num_units,
    forget_bias=1.0,
    state_is_tuple=True,
    activation=None,
    reuse=None
)
tf.nn.rnn_cell.BasicLSTMCell
```

---

```Python
tf.contrib.rnn.MultiRNNCell(
    cells,
    state_is_tuple=True
)
tf.nn.rnn_cell.MultiRNNCell
```

---

```Python
tf.nn.dynamic_rnn(
    cell,
    inputs,
    sequence_length=None,
    initial_state=None,
    dtype=None,
    parallel_iterations=None,
    swap_memory=False,
    time_major=False,
    scope=None
)
```

---

```Python
tf.nn.softmax_cross_entropy_with_logits(
    _sentinel=None,
    labels=None,
    logits=None,
    dim=-1,
    name=None
)
```

---

```Python
tf.nn.sampled_softmax_loss(
    weights,
    biases,
    labels,
    inputs,
    num_sampled,
    num_classes,
    num_true=1,
    sampled_values=None,
    remove_accidental_hits=True,
    partition_strategy='mod',
    name='sampled_softmax_loss'
)
```

---

```Python
tf.contrib.seq2seq.sequence_loss(
    logits,
    targets,
    weights,
    average_across_timesteps=True,
    average_across_batch=True,
    softmax_loss_function=None,
    name=None
)
```

---

```Python
tf.train.GradientDescentOptimizer(
    learning_rate,
    use_locking=False,
    name='GradientDescent'
)
```

---

```Python
tf.train.AdamOptimizer(
    learning_rate=0.001,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-08,
    use_locking=False,
    name='Adam'
)
```

---

```Python
tf.train.RMSPropOptimizer(
    learning_rate,
    decay=0.9,
    momentum=0.0,
    epsilon=1e-10,
    use_locking=False,
    centered=False,
    name='RMSProp'
)
```

---

```Python
tf.clip_by_value(
    t,
    clip_value_min,
    clip_value_max,
    name=None
)
```

---

```Python
tf.trainable_variables()
```

---

```Python
tf.gradients(
    ys,
    xs,
    grad_ys=None,
    name='gradients',
    colocate_gradients_with_ops=False,
    gate_gradients=False,
    aggregation_method=None
)
```

---

```Python
tf.clip_by_global_norm(
    t_list,
    clip_norm,
    use_norm=None,
    name=None
)
```

---

```Python
tf.Graph()
```

---

```Python
tf.Session(
    target=''
    graph=None,
    config=None
)
```

```Python
train_graph = tf.Graph()
with train_graph.as_default():
    ...
    ...

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    ...
    ...
    train_cost, _ = sess.run([cost, optimizer], feed_dict={inputs: x, targets: y})
    ...
    ...
```

## 保存与恢复相关操作

```Python
tf.train.Saver(
    var_list=None,
    reshape=False,
    sharded=False,
    max_to_keep=5,
    keep_checkpoint_every_n_hours=10000.0,
    name=None,
    restore_sequentially=False,
    saver_def=None,
    builder=None,
    defer_build=False,
    allow_empty=False,
    write_version=tf.train.SaverDef.V2,
    pad_step_number=False,
    save_relative_paths=False,
    filename=None
)
```

第一种保存和恢复方式：

```Python
with tf.Session() as sess:
    ...
    ...
    save_dir = './checkpoints/model.ckpt'
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
```

```Python
with tf.Session() as sess:
    save_dir = './checkpoints/model.ckpt'
    # save_dir = tf.train.get_checkpoint_state('./checkpoints')
    # save_dir = tf.train.latest_checkpoint('./checkpoints')
    saver = tf.train.Saver()
    saver.restore(sess, save_dir)
    ... # sess 可用
    ...
```

第二种保存和恢复方式：

```Python
with tf.Session() as sess:
    ...
    ...
    save_dir = './checkpoints/model'
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
```

```Python
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    save_dir = './checkpoints/model'
    loader = tf.train.import_meta_graph(save_dir + '.meta')
    loader.restore(sess, save_dir)
    ... # sess 和 loaded_graph 均可用
    ...
```

# 3. 小结

TensorFlow 函数众多，不可能在一篇博客内列举完，所以本篇博客也只是列举我见过用过的函数，更加全面和详细的资料还需要想办法查看[官网](https://www.tensorflow.org/)资料和上网查询。PS: 目前博客列举的函数只是写出了参数，后续会陆续补充完上面函数的其余部分。

# 参考链接

[吾知 Tensorflow一些常用基本概念与函数](http://www.cnblogs.com/wuzhitj/p/6431381.html): http://www.cnblogs.com/wuzhitj/p/6431381.html

[TensorFlow 官网资料](https://www.tensorflow.org/): https://www.tensorflow.org/