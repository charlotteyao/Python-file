# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#本项目所需包
import sys
print("python version:{}".format(sys.version))

import numpy as np
print("numpy version:{}".format(np.__version__))

import scipy as sp
print("scipy version:{}".format(sp.__version__))

import pandas as pd
print("pandas version:{}".format(pd.__version__))

import matplotlib
print("matplotlib version:{}".format(matplotlib.__version__))

import IPython
print("IPython version:{}".format(IPython.__version__))

import sklearn
print("skit-learn version:{}".format(sklearn.__version__))



import pandas as pd
from sklearn import datasets
iris_data=datasets.load_iris()

#data对应了样本的4个特征，150行4列
print iris_data.data.shape

#显示样本特征的前5行
print iris_data.data[:5]

#target对应了样本的类别（目标属性），150行1列
print iris_data.target.shape

#显示所有样本的目标属性
print iris_data.target

iris_data.pd.head()
iris_data.describe()