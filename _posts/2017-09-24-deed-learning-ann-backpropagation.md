---
layout: post
title: Udacity深度学习课程-神经网络及反向传播算法
date: 2017-09-24
categories: blog
tags: [神经网路,反向传播算法]
description: 优达学城深度学习课程博客，神经网络及反向传播算法。

---

<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.2/MathJax.js?config=TeX-MML-AM_CHTML">
</script>
<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {inlineMath: [['$','$'], ['\\(','\\)']]}
});
</script>

近年来，人工智能（Artificial Intelligence，AI）行业迅猛发展，而作为其技术基础的机器学习也备受追捧。机器学习的算法有很多，大致可分为监督学习、非监督学习和强化学习，具体的算法有贝叶斯估计、支持向量机、决策树、聚类、神经网络等等。其中人工神经网络（Artificial Neural Network，ANN）在上世纪五六十年代就已经有人研究，但是几经沉浮，在近几年随着大数据和硬件计算能力的提升，神经网络以深度网络（Deep Neural Network，DNN）的姿态荣耀回归，以其在各领域优异的表现，正逐步成为机器学习领域里的独角兽。而以深度学习（Deep Learning）为契机，人工智能已经开始在各行各业产生颠覆性的影响，属于人工智能的时代已经降临。

## 1. 人工神经网络（Artificial Neural Network，ANN）

人工神经网络是深度学习的基础，因此掌握人工神经网络相关的知识是进一步学习的前提，虽基础但却十分重要。

![ANN](http://ow7l1fhke.bkt.clouddn.com/my_images/ANN.png "ANN")

图1

上图是一个三层前馈神经网络，分别为输入层、隐藏层、输出层，它是一种典型的人工神经网络。图1的第一层是输入层，共包含4个节点单元，第二层是隐藏层，共包含3个节点单元，第三层是输出层，共包含2个节点单元。$\vec x$ 是输入层的输入值，$\vec h$ 是隐藏层的激活值，$\vec o$ 是输出层的输出值，$W^{(1)}$ 是第一层和第二层之间的权重，$W^{(2)}$ 是第二层和第三层之间的权重，$\vec b^{(1)}$ 是针对第二层的偏置，$\vec b^{(2)}$ 是针对第三层的偏置。关于各变量的形状，所有的向量均为行向量，所有的权重均为矩阵，即 $\vec x = [x_{1} \quad x_{2} \quad x_{3} \quad x_{4}]$ ，$\vec h = [h_{1} \quad h_{2} \quad h_{3}]$ ，$\vec o = [o_{1} \quad o_{2}]$ ，$W^{(1)}$ 是$4 \times 3$的矩阵，$W^{(2)}$ 是$3 \times 2$的矩阵。之所以选择行向量为例，是为了令反向传播算法的数学推导与代码中的形式尽量保持一致，方便编写代码。在这里我们假设隐藏层的激活函数为 $f^{(2)}$ ，输出层的激活函数为 $f^{(3)}$ 。接下来，我们以图1的神经网络为例，进行反向传播算法的数学推导。


## 2. 反向传播算法（Backpropagation Algorithm）

反向传播算法目前而言还是深度学习的基础核心算法，简单来讲，就是设定优化目标后，利用梯度下降算法不断调整优化网络中的各项参数，使得网络的输出与目标越来越靠近。现在，[最新的研究已经开始考虑放弃BP算法的束缚](https://mp.weixin.qq.com/s?__biz=MzA3MzI4MjgzMw==&mid=2650731098&idx=1&sn=c7391caee3a567b4b046406d53f022f2&chksm=871b3624b06cbf320f3725fe452d291e04a4a8c1beda8ee9e00f1d10266847be4736090aade3&scene=0#rd)，以此进一步改进深度学习的性能，当然现在还是正处于起步阶段。

针对图1的前馈神经网络，我们接下来进行反向传播算法的相关推导。首先需要设定损失函数Loss function（或者说代价函数Cost function），这里设定的是均方误差（Mean Squared Error，MSE）

$$J(W, \vec{b}) = \frac{1}{2m} \sum_{i=1}^{m} \|\vec{y}_i-\vec{o}_i\|_2^2 \qquad (1)$$

其中，$m$ 为样本容量。均方误差在回归问题里是一个比较常用的损失函数，损失函数在不同问题里可以有不同的定义和形式，例如分类问题里的交叉熵。

损失 $J$ 可以被看作是各层参数（权重和偏置）的函数，选取某个样本的特征 $\vec{x}$ 及其对应的真实目标 $\vec{y}$ ，若此时对应的网络输出为 $\vec{o}$ ，则设定

$$J(W, \vec{b} \ ; \ \vec{x}, \vec{y}) = \frac{1}{2} \|\vec{y}-\vec{o}\|_2^2 \qquad (2)$$

由 $(2)$ 式进一步得

![formula(3)](http://ow7l1fhke.bkt.clouddn.com/my_formulae/blog1_%283%29.PNG "formula(3)")$\qquad (3)$

然后对输出层的偏置求偏导可得

![formula(4)](http://ow7l1fhke.bkt.clouddn.com/my_formulae/blog1_%284%29.PNG "formula(4)")$\qquad (4)$

式 $(3)$ 和 $(4)$ 是针对 $W^{(2)}$ 和 $b^{(2)}$ 求偏导，其中 $z_j^{(3)}$ 是第三层第 $j$ 个节点的输入值。设定误差项 $\delta_j^{(3)} = (y_j - o_j)f'(z_j^{(3)})$ 。接下来对 $W^{(1)}$ 和 $b^{(1)}$ 求偏导，可得式 $(5)$ 和式 $(6)$

![formula(5)](http://ow7l1fhke.bkt.clouddn.com/my_formulae/blog1_%285%29.PNG "formula(5)")$\qquad (5)$

![formula(6)](http://ow7l1fhke.bkt.clouddn.com/my_formulae/blog1_%286%29.PNG "formula(6)")$\qquad (6)$

式 $(5)$ 和 $(6)$ 是针对 $W^{(1)}$ 和 $b^{(1)}$ 求偏导，其中 $z_j^{(2)}$ 是第二层第 $j$ 个节点的输入值。设定误差项 $\delta_j^{(2)} = (\sum_{k=1}^2 w_{jk}^{(2)} \delta_k^{(3)}) f'(z_j^{(2)})$ 。

以上所有推导都是针对某层某节点单元而言的，如果考虑所有节点，则误差项为向量 $\vec{\delta}$ ，权重、偏置也都是矩阵或者向量。在此基础上，考虑所有样本，可以得到反向传播算法的概述如下（针对MSE损失函数和一般的前馈神经网络）：

(1) 对于所有的层 $l$ ，令 $\Delta{W^{(l)}} := 0$ ，$\Delta{\vec{b}^{(l)}} := 0$ （设置为全零矩阵或者全零向量）；初始化 $W^{(l)}$ ，$\vec{b}^{(l)}$ （初始化一次即可）。

(2) 对于训练数据中的每一条记录 $\vec{x}$ ，$\vec{y}$ ：

a) 计算 $\nabla_{W^{(l)}} E(W, \vec{b} \ ; \ \vec{x}, \vec{y})$ 和 $\nabla_{\vec{b}^{(l)}} E(W, \vec{b} \ ; \ \vec{x}, \vec{y})$ ：

>* 前向传播，计算各层激活值 $\vec{h}^{(l)}$（包括 $\vec{x}$ 和 $\vec{o}$）
>* 计算输出层（第 $n_l$ 层）的误差项error term（也可以叫误差）$\vec{\delta}^{(n_l)} = (\vec{y} - \vec{h}^{(n_l)}) * f'(\vec{z}^{(n_l)})$
>* 利用递推公式，计算各隐藏层（第 $l$ 层）的误差项error term（也可以叫误差）$\vec{\delta}^{(l)} = \vec{\delta}^{(l+1)} (W^{(l)})^T * f'(\vec{z}^{(l)})$
>* 计算偏导数值（即梯度）
>
>$$\nabla_{W^{(l)}} E(W, \vec{b} \ ; \ \vec{x}, \vec{y}) = -(\vec{h})^T \vec{\delta}^{(l+1)}$$
>$$\nabla_{\vec{b}^{(l)}} E(W, \vec{b} \ ; \ \vec{x}, \vec{y}) = -\vec{\delta}^{(l+1)}$$

b) 计算 $\Delta{W^{(l)}} := \Delta{W^{(l)}} + \nabla_{W^{(l)}} E(W, \vec{b} \ ; \ \vec{x}, \vec{y})$ ，更新权重步长。

c) 计算 $\Delta{b}^{(l)} = \Delta{b}^{(l)} + \nabla_{\vec{b}^{(l)}} E(W, \vec{b} \ ; \ \vec{x}, \vec{y})$ ，更新偏置步长。

(3) 更新权重和偏置，其中 $\eta$ 是学习率，$m$ 是样本容量（或者说数据点的数量）：

$$W^{(l)} = W^{(l)} - \eta\frac{\Delta{W^{(l)}}}{m}$$
$$b^{(l)} = b^{(l)} - \eta\frac{\Delta{b^{(l)}}}{m}$$

(4) 重复(1)(2)(3)e代(epochs)。

## 3. 小结

按照上面的算法概述即可进行代码编写。上面的反向传播算法是最为基础的，若是网络类型和损失函数发生变化，上面的算法相关公式也要发生变化，但是梯度下降的思想不变。

人工神经网络是借鉴生物神经网络而来的，而深度学习就是人工神经网络的进一步发展。深度学习并不是简单地加深神经网络，这背后伴随着大数据技术的发展、GPU等硬件计算能力的飞速提升以及更多的深度网络技巧，这些都深刻促进了深度学习的发展与崛起。深度学习作为人工智能时代的先锋军，我相信一定会在未来绽放出更为耀眼的光芒。