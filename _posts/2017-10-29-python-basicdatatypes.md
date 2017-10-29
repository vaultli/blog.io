---
layout: post
title: Python编程语言学习（二）-基本数据类型
date: 2017-10-29
categories: blog
tags: [编程学习,Python,基本数据类型]
description: Python编程语言学习博客，基本数据类型。

---

掌握基本数据类型是学习任何一种编程语言的基础。Python 的基本数据类型与其他语言类似，不过也有自己的一些特点，本篇博客围绕着 Python 3 的基本数据类型进行介绍。 

## 1. Python 基本数据类型

Python 中的变量不需要声明，每个变量在使用前都必须赋值，变量被赋值以后该变量才会被创建。在 Python 中，变量就是变量，它没有类型，这里所说的“类型”是变量所指的内存中对象的类型。等号（=）用来给变量赋值，等号（=）运算符左边是一个变量名，右边是存储在变量中的值。例如：

```Python
a = 100 # 整型变量
b = 100.0 # 浮点型变量
c = 'abc' # 字符串

print(a)
print(b)
print(c)
```

运行得结果：

```Shell
100
100.0
abc
```

### 多个变量赋值

Python 允许同时为多个变量赋值，例如：

```Python
a = b = c = 1
```

在以上实例中，创建一个整型对象，值为 1，三个变量被分配到相同的内存空间上。此外，也可以为多个对象指定多个变量，例如：

```Python
a, b, c = 100, 100.0, 'abc'
```

在以上实例中，整型对象 100 被分配给变量 a，浮点型对象 100.0 被分配给变量 b，字符串对象 'abc' 被分配给变量c。

### 标准数据类型

Python 3 中有 6 个标准的数据类型：

* 数字（Number）
* 字符串（String）
* 列表（list）
* 元组（Tuple）
* 集合（set）
* 字典（Dictionary）

## 2. 数字（Number）

Python 3 支持int 、float 、bool 、complex 四种数值类型。Python 3 里只有 int 一种整数类型，表示长整型，没有 Python 2 中的 long 。跟大多数编程语言一样，Python 中数值类型的赋值和计算都很直观。利用内置的 `type()` 函数可以用来查询变量所指的对象类型。例如：

```Python
a, b, c, d = 100, 100.0, True, 1+2j
print(type(a), type(b), type(c), type(d))
```

运行结果：

```Shell
<class 'int'> <class 'float'> <class 'bool'> <class 'complex'>
```

除此之外，还可以利用 `isinstance()` 函数来判断某变量是否为某种类型。例如：

```Python
a = 100
print(isinstance(a, int))
```

运行结果为：

```Shell
True
```

`type()` 和 `isinstance()` 的区别在于：

* `type()` 不会认为子类是一种父类类型。
* `isinstance()` 会认为子类是一种父类类型。

```Python
class A:
    pass
class B(A):
    pass


type(A()) == A # return True
isinstance(A(), A) # return True
type(B()) == A # return False
isinstance(B(), A) # return True
```

> PS：在 Python 2 中是没有布尔型的，此时 1（int 型）表示 `True` ，0（int 型）表示 `False` 。但是 Python 3 中，`True` 和 `False` 被定义成了关键字，可它们的值依然还是 1（int 型）和 0（int 型），它们可以和数字进行运算。

可以利用 `del` 语句删除单个或多个变量，例如：

```Python
del var
del var1, var2
```

### 数值运算

`+`（加法）、`-`（减法）、`*`（乘法）、`/`（除法，得到一个浮点数）、`//`（整除，得到一个整数）、`%`（取余）、`**`（乘方）

注意：

* 一个变量可以通过赋值指向不同类型的对象。
* 在混合计算时，Python 会把整数转换为浮点数。

### 数值类型实例

|int   |float |complex        |
|:----:|:----:|:-------------:|
|100   |100.0 |100j           |
|-100  |-100.0|0.100j         |
|0x100 |1e100 |100+100.0j     |
|-0x100|1E100 |100+100.0J     |
|0o12    |10.   |complex(10, 10)|

## 3. 字符串（String）

Python 的字符串用单引号（`'`）或双引号（`"`）括起来，对应的是 str() 函数，同时使用反斜杠（`\`）转义特殊字符，如果不想让反斜杠发生转义，可以在字符串前面添加一个 `r` ，表示原始字符串。字符串支持切片截取，索引值以 0 为开始值，-1 为从末位开始的位置。加号（`+`）是字符串的连接符；星号（`*`）表示重复当前字符串，紧跟的数字为重复的次数。

与 C 语言不同，Python 没有单独的字符类型，一个字符就是长度为 1 的字符串。而且，与 C 字符串不同，Python 字符串不能被修改，向一个索引位置赋值，例如 `str = 'abc'; str[0] = 'd'` ，会导致错误。

构造空字符串的语法如下：

```Python
my_str1 = ''  # 空字符串
my_str1_1 = str()  # 空字符串
```

实例：

```Python
str = 'abcdef'  # 原始字符串

print(str)  # 输出字符串
print(str[0], str[5])  # 输出第一个和最后一个字符
print(str[-1], str[-6])  # 输出最后一个和第一个字符
print(str[0: -1])  # 输出从第一个到倒数第二个的所有字符
print(str[1: 5])  # 输出从第二个到倒数第二个的所有字符
print(str[1:])  # 输出从第二个开始到最后的所有字符
print(str * 2)  # 重复字符串然后输出
print(str + 'ABCDEF')  # 连接字符串然后输出

print('\n', end='')
print('adc\ndef')  # \n 为转义字符，表示换行
print(r'adc\ndef')  # \n 不为转义字符
```

运行结果：

```Shell
abcdef
a f
f a
abcde
bcde
bcdef
abcdefabcdef
abcdefABCDEF

adc
def
adc\ndef

```

## 4. 列表（List）

列表（List）是 Python 中使用最频繁的数据类型，它可以实现大多数集合类的数据结构。列表中元素的类型可以不相同，数字、字符串，甚至是列表都可以作为列表中的元素。

列表是写在方括号之间，用逗号分隔开的元素列表，对应 list() 函数。和字符串一样，列表同样可以被索引和切片截取，列表被截取后返回一个包含所需元素的新列表。索引值以 0 为开始值，-1 为从末位开始的位置。加号（`+`）是列表连接运算符；星号（`*`）是重复操作，紧跟的数字为重复的次数。

与 Python 字符串不同，列表中的元素是可以改变的，如 `my_list = [0, 1, 2, 3]; my_list[0] = 4` ，`my_list` 会变为 `[4, 1, 2, 3]` 。而且，列表还内置了很多方法，例如 list.append() 、list.pop() 等等，详细内容会在之后的博客中提及。

构造包含 0 个或 1 个元素的列表的语法如下：

```Python
my_list1 = []  # 空列表
my_list1_1 = list()  # 空列表
my_list2 = [10]  # 一个元素，注意与元组区分
```

实例：

```Python
my_list = ['abc', 789, 1.23, True, 1+2j, "def"]  # 原始列表

print(my_list)  # 输出列表
print(my_list[0], my_list[5])  # 输出第一个和最后一个元素
print(my_list[-1], my_list[-6])  # 输出最后一个和第一个元素
print(my_list[0: -1])  # 输出从第一个到倒数第二个的所有元素
print(my_list[1: 5])  # 输出从第二个到倒数第二个的所有元素
print(my_list[1:])  # 输出从第二个开始到最后的所有元素
print(my_list * 2)  # 重复列表元素然后输出
print(my_list + [123, 'abc', False])  # 连接列表然后输出
```

运行结果：

```Shell
['abc', 789, 1.23, True, (1+2j), 'def']
abc def
def abc
['abc', 789, 1.23, True, (1+2j)]
[789, 1.23, True, (1+2j)]
[789, 1.23, True, (1+2j), 'def']
['abc', 789, 1.23, True, (1+2j), 'def', 'abc', 789, 1.23, True, (1+2j), 'def']
['abc', 789, 1.23, True, (1+2j), 'def', 123, 'abc', False]

```

## 5. 元组（Tuple）

元组（Tuple）写在小括号里，元素之间用逗号隔开，对应 tuple() 函数。元组同样可以被索引和切片截取，元组被截取后返回一个包含所需元素的新元组。索引值以 0 为开始值，-1 为从末位开始的位置。加号（`+`）是元组连接运算符；星号（`*`）是重复操作，紧跟的数字为重复的次数。

元组与列表类似，不同之处在于元组的元素不能被修改，如 `my_tuple = (0, 1, 2, 3); my_tuple[0] = 4` 是错误的，这一点与字符串相似，事实上，字符串可以被看作是一种特殊的元组。元组中的元素类型可以不相同，甚至包含列表（list）这种可变的对象。

string 、list 和 tuple 都属于序列（sequence）。

构造包含 0 个或 1 个元素的元组的语法如下：

```Python
my_tuple1 = ()  # 空元组
my_tuple1_1 = tuple()  # 空元组
my_tuple2 = (10,)  # 一个元素，注意要有逗号
```

实例：

```Python
my_tuple = ('abc', 789, 1.23, True, 1+2j, "def")  # 原始元组

print(my_tuple)  # 输出元组
print(my_tuple[0], my_tuple[5])  # 输出第一个和最后一个元素
print(my_tuple[-1], my_tuple[-6])  # 输出最后一个和第一个元素
print(my_tuple[0: -1])  # 输出从第一个到倒数第二个的所有元素
print(my_tuple[1: 5])  # 输出从第二个到倒数第二个的所有元素
print(my_tuple[1:])  # 输出从第二个开始到最后的所有元素
print(my_tuple * 2)  # 重复元组元素然后输出
print(my_tuple + (123, 'abc', False))  # 连接元组然后输出
```

运行结果：

```Shell
('abc', 789, 1.23, True, (1+2j), 'def')
abc def
def abc
('abc', 789, 1.23, True, (1+2j))
(789, 1.23, True, (1+2j))
(789, 1.23, True, (1+2j), 'def')
('abc', 789, 1.23, True, (1+2j), 'def', 'abc', 789, 1.23, True, (1+2j), 'def')
('abc', 789, 1.23, True, (1+2j), 'def', 123, 'abc', False)

```

## 6. 集合（Set）

集合（Set）是一个无序不重复元素的序列。基本功能是进行成员关系测试和删除重复元素。集合可以使用大括号进行创建，对应 set() 函数。此处的集合概念和数学中的集合是类似的，集合中的元素必须是无序不重复的。因此，集合不可以被索引和切片截取。连接操作（`+`）和重复操作（`*`）对集合不适用。但是集合可以进行交集、并集、差集等操作。

构造包含 0 个或 1 个元素的集合的语法如下：

```Python
my_set1 = set()  # 空集，不可以使用 {} ，{} 用于创建空字典。
my_set2 = {10}  # 一个元素
```

实例：

```Python
my_set = {'abc', 'abc', 789, 789, 1.23, True, 1+2j, "def"}  # 集合

print(my_set)  # 输出集合，重复元素会被自动去掉

my_set = set(['abc', 'abc', 789, 789, 1.23, True, 1+2j, "def"])  # 集合，另一种创建方式，list() 和 tuple() 有类似用法

print(my_set)  # 输出集合
```

运行结果：

```Shell
{1.23, True, 'abc', (1+2j), 789, 'def'}
{1.23, True, (1+2j), 'abc', 789, 'def'}

```

实例：

```Python
a = set('abababcdefxxyyzz')
b = set('abcdefgghh')

print(a)  # 输出集合 a
print(b)  # 输出集合 b
print(a - b)  # 输出 a 和 b 的差集
print(a | b)  # 输出 a 和 b 的并集
print(a & b)  # 输出 a 和 b 的交集
print(a ^ b)  # 输出 a 和 b 中不同时存在的元素
```

运行结果（集合 a 为 {'a', 'b', 'c', 'd', 'e', 'f', 'x', 'y', 'z'} ；集合 b 为 {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'}）：

```Shell
{'c', 'a', 'x', 'e', 'y', 'd', 'z', 'f', 'b'}
{'c', 'g', 'a', 'e', 'h', 'd', 'f', 'b'}
{'x', 'z', 'y'}
{'c', 'g', 'a', 'x', 'e', 'y', 'h', 'd', 'z', 'f', 'b'}
{'c', 'a', 'e', 'd', 'f', 'b'}
{'g', 'x', 'y', 'h', 'z'}

```

## 7. 字典（Dictionary）

字典（Dictionary）是 Python 中非常有用的内置数据类型。字典是无序的对象集合，字典中的元素是通过键值对来存取的，而不是通过偏移存取的。字典是一种映射类型，用大括号标识，对应的是 dict() 函数，它是一个无序的键(key):值(value)对集合。键（key）必须使用不可变类型，在同一个字典中，键（key）必须是唯一的。字典类型也有一些内置函数，例如 dict.clear() 、dict.keys() 、dict.values() 等等，之后的博客会详细介绍。

构造空字典的语法如下：

```Python
my_dict1 = {}  # 空字典
my_dict1_1 = dict()  # 空字典
```

实例：

```Python
my_dict = {'a': 1, 2: 'b'}  # 字典

print(my_dict)  # 输出字典

# 字典，另一种创建方式
my_dict = {}
my_dict['a'] = 1
my_dict[2] = 'b'

print(my_dict)  # 输出字典
print(my_dict['a'])  # 输出键为 'a' 的值
print(my_dict[2])  # 输出键为 2 的值
print(my_dict.keys())  # 输出所有键
print(my_dict.values())  # 输出所有值
```

运行结果：

```Shell
{'a': 1, 2: 'b'}
{'a': 1, 2: 'b'}
1
b
dict_keys(['a', 2])
dict_values([1, 'b'])

```

构造函数 dict() 可以直接从键值对序列中构造字典如下：

```Python
a = dict([('a', 1), (2, 'b')])

print(a)

a = dict(a=1, b=2)  # 此种形式无法创建类似字典 {'a': 1, 2: 'b'} 的这种形式

print(a)
```

运行结果：

```Shell
{'a': 1, 2: 'b'}
{'a': 1, 'b': 2}

```

## 8. 数据类型转换

编写代码时，有时候需要对数据类型进行转换。进行数据类型转换时，只需要将数据类型作为函数名即可。下表中内置的函数可以执行数据类型之间的转换，这些函数返回一个新的对象，表示转换的值。

|函数|描述|
|:---:|:---:|
|`int(x[, base])`|将 x 转换为一个整数|
|`float(x)`|将 x 转换为一个浮点数|
|`complex(real[, imag])`|创建一个复数|
|`str(x)`|将对象 x 转换为字符串|
|`repr(x)`|将对象 x 转换为表达式字符串|
|`eval(str)`|用来计算在字符串中的有效 Python 表达式，并返回一个对象，'123'-->123|
|`chr(x)`|将一个整数转换为一个字符|
|`ord(x)`|将一个字符转换为其对应的十进制 ASCII 码|
|`hex(x)`|将一个整数转换为一个十六进制字符串|
|`oct(x)`|将一个整数转换为一个八进制字符串|
|`list(s)`|将序列 s 转换为一个列表|
|`tuple(s)`|将序列 s 转换为一个元组|
|`set(s)`|将序列 s 转换为一个可变集合|
|`dict(d)`|创建一个字典，d 可以是一个 (key, value) 元组序列|
|`frozenset(s)`|将序列 s 转换为一个不可变集合|

## 9. 小结

字符串、列表、元组都是有序的，集合和字典都是无序的，因此前三个都是可以通过索引值进行索引的，后两个不可以通过索引值进行索引。成员运算符 `in` 这五个数据类型都可以用。而且因为 Python 中的字典用到了一个称为散列表（Hashtable）的算法，其特点是不管字典中有多少项，`in` 操作花费的时间都差不多，因此将字典用于 `in` 操作，效率会提升不少。如果把一个字典对象作为 `for` 的迭代对象，那么这个操作将会遍历字典的键。

## 10. 参考链接

[菜鸟教程 Python 3 基本数据类型](http://www.runoob.com/python3/python3-data-type.html): http://www.runoob.com/python3/python3-data-type.html