---
layout: post
title: Python编程语言学习（三）-星号（*）的用法
date: 2017-11-05
categories: blog
tags: [编程学习,Python,星号的用法]
description: Python编程语言学习博客，星号的用法。
---

## 1. 星号（*）的用法

星号 `*` 可以作为运算符，也可以在函数定义和调用阶段使用。如下：

* `*` 乘法运算符，两个数相乘或是返回一个被重复若干次的序列（字符串、列表、元组）。
* `**` 乘幂运算符，`x ** y` 表示 x 的 y 次幂。
* `*` 用在函数定义阶段，可接收多个参数，打包成元组；用在函数调用阶段，解压元组、列表、集合等为多个参数。
* `**` 用在函数定义阶段，可接受多个关键字赋值，打包成字典；函数调用阶段，解压字典为多个关键字赋值。

## 2. 实例

`*` 乘法运算符：

```Python
a = 2
b = 3
c = ['a', 'b', 'c']
d = a * b
e = c * 2
print(d)
print(e)
```

运行结果：

```Shell
6
['a', 'b', 'c', 'a', 'b', 'c']

```

---

`**` 乘幂运算符：

```Python
a = 2
b = 3
c = a ** b
print(c)
```

运行结果：

```Shell
8

```

---

`*` 函数定义阶段：

```Python
def add(*args):
    a = args[0]
    b = args[1]
    c = a + b
    print(a)
    print(b)
    print(c)

add(2, 3)
```

运行结果：

```Shell
2
3
5

```

`*` 函数调用阶段：

```Python
def add(a, b):
    c = a + b
    print(a)
    print(b)
    print(c)

l = [2, 3]
add(*l)
```

运行结果：

```Shell
2
3
5

```

---

`**` 函数定义阶段：

```Python
def add(**kwargs):
    a = kwargs['a']
    b = kwargs['b']
    c = a + b
    print(a)
    print(b)
    print(c)

add(a=2, b=3)
```

运行结果：

```Shell
2
3
5

```

`**` 函数调用阶段：

```Python
def add(a, b):
    c = a + b
    print(a)
    print(b)
    print(c)

d = {'a': 2, 'b': 3}
add(**d)
```

运行结果：

```Shell
2
3
5

```