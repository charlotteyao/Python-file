# -*- coding: utf-8 -*-
"""
Created on Mon Jun 12 15:01:42 2017

@author: yaohaiying
"""

from numpy import * 
import operator
import matplotlib.pyplot as plt

#创建数据集，以后拿数据集和这些点距离进行比较，得到最近的点，进行预测分类
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

#inX，用于分类的输入向量
#dataSet，训练样本集
#labels，标签向量
#k，选择最近邻居的数目
#【标签向量元素数数目和矩阵dataSet的行数相同】    
def classify0(inX,dataSet,labels,k):
    #----------------距离计算-------------------
    #读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度
    dataSetSize = dataSet.shape[0]
    #tile(A,N),功能是将数组A重复n次，构成一个新的数组
    #tile(A,(m,n)),功能是将数组A重复m行，n列，构成一个新的数组
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    #以下3行代码执行的是欧式距离的计算
    sqDiffMat = diffMat**2 #这个是求幂次方，power方法一样的 5**2 = 25
    #axis=0表示列，axis=1表示行 axis=1以后就是将一个矩阵的每一行向量相加
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5 #开方，就是根号，这里开方之后就是每个点到inX的距离
    #----------------距离计算-------------------
    
    #----------------选择距离最小的k个点-------------------
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        #dict.get(key, default=None)key 为字典中要查找的键，
        #default如果指定键的值不存在时，返回该默认值值。此句代码用于统计标签出现的次数  
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    #----------------选择距离最小的k个点-------------------
        
    #对结果进行排序
    #sorted函数参数解释，sorted(iterable, cmp=None, key=None, reverse=False)  
    #iterable：是可迭代类型;  
    #cmp：用于比较的函数，比较什么由key决定;  
    #key：用列表元素的某个属性或函数进行作为关键字，有默认值，迭代集合中的一项;  
    #reverse：排序规则. reverse = True  降序 或者 reverse = False 升序，有默认值。  
    #返回值：是一个经过排序的可迭代类型，与iterable一样。  
    ######  
    #operator模块提供的itemgetter函数用于获取对象的哪些维的数据，参数为一些序号（即需要获取的数据在对象中的序号）  
    ######      
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),reverse=True)
    #返回最符合的标签 
    return sortedClassCount[0] [0]

#将文本记录转换为Numpy的解析程序
def file2matrix(filename):
    #打开文件
    fr = open(filename) 
    print fr
    #获取文件所有行
    arrayOLines = fr.readlines() 
    print arrayOLines
    #得到文件行数
    numberOfLines = len(arrayOLines) 
    print numberOfLines
    #先用零元素创建需要返回的numpy矩阵（行数，列数）
    returnMat= zeros((numberOfLines,3)) 
    print returnMat
    #创建空的标签列表
    classLabelVector = [] 
   
    index = 0
    
    for line in arrayOLines:
        #截取掉尾部的回车字符
        line = line.strip() 
        #用‘\t’作为分隔符将整行元素分割成元素列表，将一行数据按空进行分割
        listFromLine = line.split('\t') 
        #选取列表前三个元素到矩阵中
        returnMat[index,:] = listFromLine[0:3] 
        #将列表的最后一列存储到向量中
        classLabelVector.append(listFromLine[-1])
        
        index += 1
    #返回数据集矩阵和对应的标签向量
    return returnMat,classLabelVector 
    
#归一化特征值    
def autoNorm(dataSet):
    #找到数据集中的最小值（实际上应该是样本数据中的一列中的最小值，参数0就代表这个，下同），
    #这样说的话minVals和maxVals都应该是一个行向量（1*n）
    minVals = dataSet.min(0)
    #找到数据集中的最大值
    maxVals = dataSet.max(0)
    #得到数据的范围差值
    ranges = maxVals - minVals
    #数据集与最小值相减（title（）函数将按照括号中的参数制作对应大小的矩阵，
    #用给定的minVals内容来填充）
    normDataSet = dataSet - tile(minVals,(m,1))
    #除以范围值之后就是归一化的值了（注意是矩阵除法）
    normDataSet = normDataSet/tile(ranges,(m,1))
    return normDataSet,ranges,minVals

#分类器针对约会网站的测试代码    
def datingClassTest():
    hoRatio = 0.10
    datingDataMat,datingLabels = file2matrix('datingTestSet2.txt')
    normMat,ranges,minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %s,the real answer is: %s" % (classifierResult,datingLabels[i])        
        if(int(classifierResult) != int(datingLabels[i])):errorCount += 1.0
        
        print "the total error rate is: %f" % (errorCount/float(numTestVecs))
        print errorCount































#创建数据集
group,labels = createDataSet()

#计算相邻值
result = classify0([0,0],group,labels,3)

print result