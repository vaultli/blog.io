---
layout: post
title: Python编程语言学习（一）-基础语法
date: 2017-10-15
categories: blog
tags: [编程学习,Python,基础语法]
description: Python编程语言学习博客，基础语法。
---

## 1. Python 简介

Python 是一种解释型、面向对象、动态数据类型的高级程序设计语言，由荷兰计算机工程师 Guido van Rossum 于 1989 年发明，第一个公开发行版发行于 1991 年。Python 语言目前有 Python 2.x 和 Python 3.x 两个版本，为了不带入太多累赘，Python 3.x 在设计的时候没有考虑向下兼容。[Python 2.x 和 Python 3.x 两个版本语法上有一些区别](http://www.runoob.com/python/python-2x-3x.html)，本博客中涉及的语言版本为 Python 3.x 。

利用下面的命令行可以查看使用的 Python 版本：

```Shell
$ python -V
```

执行结果如下（基于 Anaconda 环境）：

```Shell
Python 3.5.2 :: Anaconda 4.2.0 (64-bit)
```

Python 的 “Hello World!” 程序：

```Python
print("Hello World!")
```

将上面的代码保存成 hello.py 文件，并使用 python 命令执行该脚本文件：

```Shell
$ python hello.py
```

或者

```Shell
$ python3 hello.py
```

以上命令的输出结果为：

```Shell
Hello World!
```

## 2. 基础语法

### 编码

默认情况下，Python 3.x 源码文件以 UTF-8 编码，所有字符串都是 unicode 字符串，此时，下面的代码是可以正确执行的：

```Python
中国 = 'China'
print(中国)
```

当然也可以为源码文件指定不同的编码：

```Python
# -*- coding: cp-1252 -*-
```

### 标识符

与其他编程语言一样，Python 中的标识符是程序员自己规定的具有特定含义的词，比如类名称、属性名称、变量名等。

Python 中标识符需注意以下几点：

* 第一个字符必须是字母表中字母或下划线“_”。
* 标识符其他部分由字母、数字或下划线组成。
* 标识符对大小写敏感。

在 Python 3 中，非 ASCII 标识符也是允许的。

### 关键字

关键字也叫保留字，不能把它们用作任何标识符名称。Python 的标准库提供一个 keyword 模块，可以输出当前版本的所有关键字：

```IPython
>>> import keyword
>>> keyword.kwlist
['False', 'None', 'True', 'and', 'as', 'assert', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
```

### 注释

Python 中的单行注释以 # 开头，如下：

```Python
# 第一行注释
print("Hello World!") # 第二行注释
```

上面代码的执行结果为：

```Shell
Hello World!
```

多行注释可以用多个 # 号，也可以用三对单引号 ' 或者三对双引号 " 包含，如下：
```Python
# 第一行注释
# 第二行注释

'''
三对单引号, 多行注释
三对单引号, 多行注释
'''

"""
三对双引号, 多行注释
三对双引号, 多行注释
"""

print("Hello World!")
```

上面代码的执行结果为：

```Shell
Hello World!
```

在 Linux 系统或者 macOS 系统中，Python 代码文件的第一行一般写一句注释：

```Python
# !/usr/bin/python3
```

这个第一行注释标的是指向 Python 的路径，告诉操作系统执行这个脚本的时候，调用 /usr/bin 下的 Python 解释器。此外还有以下形式（推荐写法）：

```Python
# !/usr/bin/env python3
```

这种用法先在 env (环境变量) 设置里查找 Python 的安装路径，再调用对应路径下的解释器程序完成操作。

需要说明的是，Windows 系统下可以不写这个第一行注释。

### 行与缩进
Python 使用缩进来表示代码块，不需要使用大括号 {} ，这是 Python 的特色之处，同时也是非常需要注意的地方。缩进的空格数是可变的，但是同一个代码块的语句必须包含相同的缩进空格数，缩进可以利用 Tab 键或者空格键实现。实例如下：

```Python
if True:
    print("Example")
    print("True")
else:
    print("Example")
    print("False")
```

以下代码最后一行语句缩进的空格数不一致，会导致运行错误：

```Python
if True:
    print("Example")
    print("True")
else:
    print("Example")
  print("False")
```

以上程序由于缩进不一致，执行时会出现以下错误：

```Shell
  File "C:\XXXX\XXXX\XXXX\test.py", line 6
    print("False")
                 ^
IndentationError: unindent does not match any outer indentation level
```

### 多个语句构成代码组

缩进相同的一组语句构成一个代码块，称之为代码组。例如if、while、def和class这样的复合语句，首行以关键字开始，以冒号 : 结束，该行之后的一行或多行代码构成代码组。将首行及后面的代码组称为一个子句（clause），实例如下（Python伪代码）：

```Python
if expression:
    suite
elif expression:
    suite
else:
    suite
```

### 一条语句分成多行语句

Python通常是一行写完一条语句，但是如果语句很长，需要写成多行时，可以使用反斜杠 \ 来实现多行语句的连接，如下：

```Python
total = item_one + item_two + \
        item_three + item_four + \
        item_five
```

在 ()，[]，{} 中的多行语句，不需要使用反斜杠\来连接如下：

```Python
total = ['item_one', 'item_two',
        'item_three', 'item_four',
        'item_five']
```

### 多条语句写在一行

Python 可以在同一行中使用多条语句，语句之间使用分号 ; 分割，如下：

```Python
import sys; x = 'abcd'; sys.stdout.write(x + '\n')
```

执行以上代码的结果为：

```Shell
$ python test.py
abcd
```

### 数据类型

Python 3 中有六个标准的数据类型：

* Number（数字）
* String（字符串）
* List（列表）
* Tuple（元组）
* Set（集合）
* Dictionary（字典）

#### Number（数字）

Python 3 支持 int、float、bool和complex。

* int（长整型），如 1000，在Python 3里只有一种整数类型int，表示长整型，没有了Python 2中的long。
* float（浮点数），如 1.23、1E-2（或者1e-2）。
* bool（布尔型），关键字 True、False。
* complex（复数），如 1 + 2j、1.1 + 2.2j。

#### String（字符串）

* Python 中利用单引号 ' 和双引号 " 构成字符串，二者成对出现，二者的使用完全相同。
* 使用成对的三个单引号 ''' 或者三个双引号 """ 可以指定一个多行字符串（注意区分与多行注释的情形，形式一样，但当多行注释赋给某变量时就成为了多行字符串）。
* 转义符 \ 。
* 自然字符串，通过在字符串前加 r 或者 R 实现，如 r"this is a line with \n"则\n会显示出来并不是换行。
* Python 允许处理 unicode 字符串，加前缀  u或者 U ，如 u"this is an unicode string"。
* 字符串是不可变的。
* 级联字符串，利用加号 + 实现，或者直接写在一行。

实例如下：

```Python
word = 'string'
sentence = "this is a sentence."
paragraph = """this is a paragraph,
consisting of many sentences."""

# a、b和c是一样的，均是'thisisastring'
a = 'this' + 'is' + 'a' + 'string'
b = 'this''is''a''string'
c = 'this' 'is' 'a' 'string'
```

列表、元组等数据结构留待后续博客详细分析。

### 空行

函数之间或者类的方法之间用空行分隔，表示一段新的代码的开始。类和函数入口之间也用一行空行分隔，以突出函数入口的开始。空行与代码缩进不同，空行不是 Python 语法的一部分，书写程序时不插入空行，Python 解释器运行也不会出错。但是空行的作用在于分隔两段不同功能或含义的代码，便于日后代码的维护或重构。因此，适当的空行是好的程序代码不可或缺的一部分。

### 等待用户输入

执行下面的程序在按回车键后就会等待用户输入：

```Python
input("\n\n按下 enter 键后退出。")
```

运行以上代码，结果如下：

```Shell
$ python test.py


按下 enter 键后退出。
```

按下 enter 键后，结果如下：

```Shell
$ python test.py


按下 enter 键后退出。
$
```

利用 `a = input()` 可以将输入赋值给变量 a 。

### print输出

print() 函数默认输出是换行的，如果要实现不换行，需要给参数 end 重新赋值，例如 end=" " ：

```Python
x = 'a'
y = 'b'
# 换行输出
print(x)
print(y)
print('----------')

# 不换行输出
print(x, end=" ")
print(y, end=" ")
print()
```

以上代码的输出结果为：

```Shell
$ python test.py
a
b
----------
a b
$
```

### import 与 form...import

在 Python 中用 import 或者 from...import 来导入相应的模块。将整个模块（module）导入，格式为 `import somemodule` ；从某个模块中导入某个函数，格式为 `from somemodule import somefunction` ；从某个模块中导入多个函数，格式为 `from somemodule import firstfunc, secondfunc, thirdfunc` ；将某个模块中的全部函数导入，格式为 `from somemodule import *` 。实例如下：

```Python
import sys # 导入sys模块
print('===========Python import mode==========')
print('命令行参数为：')
for i in sys.argv:
    print(i)
print('\n Python 可访问路径为', sys.path)
```

运行以上代码，结果如下：

```Shell
$ python test.py
===========Python import mode==========
命令行参数为：
test.py

 Python 可访问路径为 ['C:\\XXXX\\xxxxx\\xxxx', 'C:\\Program Files\\Anaconda3\\Lib\\site-packages\\libsvm', 'C:\\Program Files\\Anaconda3\\Lib\\site-packages\\libsvm\\python', 'C:\\Program Files\\Anaconda3\\python35.zip', 'C:\\Program Files\\Anaconda3\\DLLs', 'C:\\Program Files\\Anaconda3\\lib', 'C:\\Program Files\\Anaconda3', 'C:\\Program Files\\Anaconda3\\lib\\site-packages', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\Sphinx-1.4.6-py3.5.egg', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\win32', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\win32\\lib', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\Pythonwin', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\setuptools-27.2.0-py3.5.egg']
$
```

```Python
from sys import argv, path # 导入特定的成员
print('===========Python from...import mode==========')
print('Python 可访问路径为', path) # 因为已经导入 path 成员，所以此处调用时不需要写成 sys.path 。
```

运行以上代码，结果如下：

```Shell
$ python test.py
===========Python from...import mode==========
 Python 可访问路径为 ['C:\\XXXX\\xxxxx\\xxxx', 'C:\\Program Files\\Anaconda3\\Lib\\site-packages\\libsvm', 'C:\\Program Files\\Anaconda3\\Lib\\site-packages\\libsvm\\python', 'C:\\Program Files\\Anaconda3\\python35.zip', 'C:\\Program Files\\Anaconda3\\DLLs', 'C:\\Program Files\\Anaconda3\\lib', 'C:\\Program Files\\Anaconda3', 'C:\\Program Files\\Anaconda3\\lib\\site-packages', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\Sphinx-1.4.6-py3.5.egg', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\win32', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\win32\\lib', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\Pythonwin', 'C:\\Program Files\\Anaconda3\\lib\\site-packages\\setuptools-27.2.0-py3.5.egg']
$
```

### 命令行参数

很多程序可以在命令行执行一些操作来查看一些基本信息，Python 可以使用 -h 参数查看各种参数的帮助信息：

```Shell
$ python -h
```

在使用脚本形式执行 Python 程序时，可以接收命令行输入的参数，具体使用可以参照 [Python 3 命令行参数](http://www.runoob.com/python3/python3-command-line-arguments.html)。

### 查看帮助信息

调用 Python 的 `help()` 函数可以打印输出一个函数的帮助文档字符串。也可以利用 `somefunc.__doc__` 查看 `__doc__` 属性只得到帮助文档字符串（注意 doc 前后分别是两个下划线）。

## 3. 参考链接

[菜鸟教程 Python 基础教程](http://www.runoob.com/python/python-tutorial.html): http://www.runoob.com/python/python-tutorial.html

[菜鸟教程 Python 3 基础教程](http://www.runoob.com/python3/python3-tutorial.html): http://www.runoob.com/python3/python3-tutorial.html

[菜鸟教程 Python3 基础语法](http://www.runoob.com/python3/python3-basic-syntax.html): http://www.runoob.com/python3/python3-basic-syntax.html

[菜鸟教程 Python2.x与3​​.x版本区别](http://www.runoob.com/python/python-2x-3x.html): http://www.runoob.com/python/python-2x-3x.html

[爱尔兰时空 Python 3 数据类型](http://www.cnblogs.com/zhanmeiliang/p/5977168.html): http://www.cnblogs.com/zhanmeiliang/p/5977168.html

[玩蛇网 Python 单行、多行注释符号使用方法及规范](http://www.iplaypy.com/jichu/note.html): http://www.iplaypy.com/jichu/note.html

[菜鸟教程 Python 3 命令行参数](http://www.runoob.com/python3/python3-command-line-arguments.html): http://www.runoob.com/python3/python3-command-line-arguments.html