# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 14:19:27 2017

@author: yaohaiying
"""
#1、前期准备
# Imports

# Numpy,Pandas
import numpy as np
import pandas as pd

# matplotlib,seaborn,pyecharts

import matplotlib.pyplot as plt
plt.style.use('ggplot')  #风格设置近似R这种的ggplot库
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline
import missingno as msno

#  忽略弹出的warnings
import warnings
warnings.filterwarnings('ignore')  

pd.set_option('display.float_format', lambda x: '%.5f' % x)

#2、数据获取与解析
#读取数据
data = pd.read_csv('D:\python file\lending club\LoanStats_2017Q2.csv' , encoding='latin-1',skiprows = 1) 
data.head() #查看表格默认前5行
#统计每列属性缺失值的数量。
#查看缺失值比例
check_null = data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(data))
# 查看缺失比例大于20%的属性。
print(check_null[check_null > 0.2]) 

# 设定阀值
thresh_count = len(data)*0.4 

#若某一列数据缺失的数量超过阀值就会被删除
data = data.dropna(thresh=thresh_count, axis=1 ) 

#再次检查缺失值情况，发现缺失值比较多的数据列已被我们删除
data.isnull().sum(axis=0).sort_values(ascending=False)/float(len(data)) 

# 将初步预处理后的数据转化为csv
data.to_csv('D:\python file\lending club\loans_2017q2_ml.csv', 
            index = False,encoding='utf-8') 

#再次用pandas解析数据
loans = pd.read_csv('D:\python file\lending club\loans_2017q2_ml.csv',encoding='utf-8') 
loans.dtypes.value_counts() # 分类统计数据类型

#通过Pandas的nunique方法来筛选属性分类为一的变量，剔除分类数量只有1的变量，
#Pandas方法nunique()返回的是变量的分类数量（除去非空值）
loans = loans.loc[:,loans.apply(pd.Series.nunique) != 1]

#查看数据的行列，发现数据已比之前少了3列
loans.shape 

#3、缺失值处理——分类型变量
#查看分类变量缺失值的情况
objectColumns = loans.select_dtypes(include=["object"]).columns
loans[objectColumns].isnull().sum().sort_values(ascending=False)

#分类变量中，"int_rate"、"revol_util"、“annual_inc”的属性实质意义是数值，
#但pandas因为它们含有“%”符号或数字间有逗号而误识别为字符。
#为了方便后续处理，我们先将他们的数据类型重分类
loans['int_rate'] = loans['int_rate'].str.rstrip('%').astype('float')
loans['revol_util'] = loans['revol_util'].str.rstrip('%').astype('float')
loans['annual_inc'] = loans['annual_inc'].str.replace(",","").astype('float')
# 对objectColumns重新赋值
objectColumns = loans.select_dtypes(include=["object"]).columns  

#对分类型变量缺失值来个感性认知
msno.matrix(loans[objectColumns]) #缺失值可视化

#查看缺失值之间的相关性
#当相关性为0时，说明一个变量与另一个变量之间没有影响。
#相关性接近1或-1说明变量之间呈现正相关或负相关
msno.heatmap(loans[objectColumns])
#使用pandas.fillna()处理文本变量缺失值，为分类变量缺失值创建一个分类“Unknown”
# 筛选数据类型为object的数据
objectColumns = loans.select_dtypes(include=["object"]).columns 
#以分类“Unknown”填充缺失值
loans[objectColumns] = loans[objectColumns].fillna("Unknown") 

#再次查看分类变量缺失值的情况，发现缺失值已被清洗干净
msno.bar(loans[objectColumns]) #可视化

#4、缺失值处理——数值型变量
#查看数值型变量的缺失值情况
loans.select_dtypes(include=[np.number]).isnull().sum().sort_values(ascending=False)

numColumns = loans.select_dtypes(include=[np.number]).columns
msno.matrix(loans[numColumns]) #缺失值可视化

pd.set_option('display.max_columns', len(loans.columns))
loans[numColumns]

#从表格发现，第105,451行至105,454行的属性值全为NaN，
#这些空行对我们预测模型的构建没有任何意义，在此先单独删除这些行
loans.drop([105451,105452,105453,105454], inplace = True)
loans[numColumns].tail() # 默认查看表格倒数5行

#对数值型变量的缺失值，我们采用均值插补的方法来填充缺失值，
#这里使用可sklearn的Preprocessing模块，
#参数strategy可选项有median或most_frequent以及median
from sklearn.preprocessing import Imputer
imr = Imputer(missing_values='NaN', strategy='mean', axis=0)  # 针对axis=0 列来处理
imr = imr.fit(loans[numColumns])
loans[numColumns] = imr.transform(loans[numColumns])
msno.matrix(loans) # 再次检查缺失值情况

#5、数据过滤
#冗余特征重复了包含在一个或多个其他属性中的许多或所有信息。
#例如，zip_code对于我们借款人的偿债能力并没有任何意义。
#grade和sub_grade是重复的属性信息。下一步，我们对数据进行过滤。
objectColumns = loans.select_dtypes(include=["object"]).columns
var = loans[objectColumns].columns
for v in var:
    print('\nFrequency count for variable {0}'.format(v))
    print(loans[v].value_counts())
loans[objectColumns].shape

#sub_grade：与Grade的信息重复
#emp_title ：缺失值较多，同时不能反映借款人收入或资产的真实情况
#zip_code：地址邮编，邮编显示不全，没有意义
#addr_state：申请地址所属州，不能反映借款人的偿债能力
#last_credit_pull_d ：LendingClub平台最近一个提供贷款的时间，没有意义
#policy_code ： 变量信息全为1
#pymnt_plan 基本是n
#title： title与purpose的信息重复，同时title的分类信息更加离散
#next_pymnt_d : 下一个付款时间，没有意义
#policy_code : 没有意义
#collection_recovery_fee: 全为0，没有意义
#earliest_cr_line : 记录的是借款人发生第一笔借款的时间
#issue_d ： 贷款发行时间，这里提前向模型泄露了信息
#last_pymnt_d、collection_recovery_fee、last_pymnt_amnt： 
#预测贷款违约模型是贷款前的风险控制手段，
#这些贷后信息都会影响我们训练模型的效果，在此将这些信息删除 
drop_list = ['sub_grade', 'emp_title',  'title', 'zip_code', 'addr_state', 
             'mths_since_last_delinq' ,'initial_list_status','title',
             'issue_d','last_pymnt_d','last_pymnt_amnt','next_pymnt_d',
             'last_credit_pull_d','policy_code','collection_recovery_fee',
             'earliest_cr_line']
loans.drop(drop_list, axis=1, inplace = True)

#分类型变量从28列被精减至11列
loans.select_dtypes(include = ['object']).shape
loans.select_dtypes(include = ['object']).head() # 再次概览数据

#不同算法模型需要不同的数据类型来建立。例如逻辑回归只支持数值型的数据，
#而随机森林通常对字符型和数值型都支持。由于在场景分析中，
#我们判定本项目预测贷款违约是一个二元分类问题，我们选择的算法是逻辑回归算法模型，
#从数据预处理的过程中也发现数据的结构是半结构化，因此需要对特征数据作进一步转换

#5.1 特征衍生
#特征衍生是指利用现有的特征进行某种组合生成新的特征。
#在风险控制方面，传统银行获得企业的基本财务报表（资产负债表、利润表以及现金流量表），
#借助于现代成熟的财务管理体系，在不同业务场景的需求下，利用企业财务报表各种项目之间的组合，
#就可以衍生不同新特征反映企业不同的财务状况，
#例如资产与负债项目组合能够生成反映企业债务情况的特征，
#收入与应收账款组合能生成反映应收账款周转率（资金效率）特征等，
#同时还能利用企业财务报表之间的勾稽关系生成新特征来佐证企业报表的质量。
#在金融风险控制中，要做好以上工作的前提是，你必须熟悉各种业务场景同时精通财务知识。

#"installment"代表贷款每月分期的金额，
#将'annual_inc'除以12个月获得贷款申请人的月收入金额，
#然后再把"installment"（月负债）与（'annual_inc'/12）（月收入）相除
#生成新的特征'installment_feat'，
#新特征'installment_feat'代表客户每月还款支出占月收入的比，
#'installment_feat'的值越大，意味着贷款人的偿债压力越大，违约的可能性越大。
loans['installment_feat'] = loans['installment'] / (loans['annual_inc'] / 12)

#5.2 特征抽象（feature abstraction）
#特征抽象是指将数据转换成算法可以理解的数据。
#使用Pandas replace函数定义新函数：

def coding(col, codeDict):

    colCoded = pd.Series(col, copy=True)
    for key, value in codeDict.items():
        colCoded.replace(key, value, inplace=True)

    return colCoded

#把贷款状态LoanStatus编码为违约=1, 正常=0:

pd.value_counts(loans["loan_status"])

loans["loan_status"] = coding(loans["loan_status"],
     {'Current':0,
      'Fully Paid':0,
      'In Grace Period':1,
      'Late (31-120 days)':1,
      'Late (16-30 days)':1,
      'Charged Off':1})

print( '\nAfter Coding:')

pd.value_counts(loans["loan_status"])

# 贷款状态分布可视化
fig, axs = plt.subplots(1,2,figsize=(14,7))
sns.countplot(x='loan_status',data=loans,ax=axs[0])
axs[0].set_title("Frequency of each Loan Status")
loans['loan_status'].value_counts().plot(x=None,y=None,
     kind='pie',ax=axs[1],autopct='%1.2f%%')
axs[1].set_title("Percentage of each Loan status")
plt.show()

#筛选数据类型为object的变量
object_columns_df =loans.select_dtypes(include=["object"]) 
print(object_columns_df.iloc[0])

#对变量“delinq_2yrs”、“total_acc”、“last_pymnt_amnt”、“revol_bal”的数据类型重分类
loans['delinq_2yrs'] = loans['delinq_2yrs'].apply(lambda x: float(x))
loans['total_acc'] = loans['total_acc'].apply(lambda x: float(x))
loans['revol_bal'] = loans ['revol_bal'].apply(lambda x: float(x))
loans.select_dtypes(include=["object"]).describe().T # 再次检查数据
#将变量类型为"object"的数量从30个缩减至7个

#有序特征的映射，对变量“emp_length”、"grade"进行特征抽象化，
#使用的方法是先构建一个mapping，再用pandas的replace( )进行映射转换
# 构建mapping，对有序变量"emp_length”、“grade”进行转换
mapping_dict = {
    "emp_length": {
        "10+ years": 10,
        "9 years": 9,
        "8 years": 8,
        "7 years": 7,
        "6 years": 6,
        "5 years": 5,
        "4 years": 4,
        "3 years": 3,
        "2 years": 2,
        "1 year": 1,
        "< 1 year": 0,
        "n/a": 0
    },
    "grade":{
        "A": 1,
        "B": 2,
        "C": 3,
        "D": 4,
        "E": 5,
        "F": 6,
        "G": 7
    }
}

loans = loans.replace(mapping_dict) #变量映射
loans[['emp_length','grade']].head() #查看效果

#对多值无序变量进行独热编码（one-hot encoding）
#使用pandas的get_dummies( )方法创建虚拟特征，虚拟特征的每一列各代表变量属性的一个分类。
#然后再使用pandas的concat()方法将新建虚拟特征和原数据进行拼接
#get_dummies返回的一组数据是一个稀疏矩阵，但这组数据已经可以带到算法中进行计算
n_columns = ["home_ownership", "verification_status",
             "application_type","purpose", "term"] 
# 用get_dummies进行one hot编码
dummy_df = pd.get_dummies(loans[n_columns])
#当axis = 1的时候，concat就是行对齐，然后将不同列名称的两张表合并
loans = pd.concat([loans, dummy_df], axis=1) 

#查看变量“home_ownership”经过独热编码处理的前后对比，
#发现“home_ownership”每一个分类均创建了一个新的虚拟特征，
#虚拟特征的每一列代表变量“home_ownership”属性的一个值。
#从下表可以看出，我们已成功将n1_columns的变量转化成算法可理解的数据类型，这里就不逐个展示
#筛选包含home_ownership的所有变量
loans.loc[:,loans.columns.str.contains("home_ownership")].head() 

loans = loans.drop(n_columns, axis=1)  #清除原来的分类变量

loans.info() #查看数据信息

#5.3 特征缩放（peature scaling）
















