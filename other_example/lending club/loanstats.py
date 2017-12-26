# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#探索性数据分析(Exploratory Data Analysis，以下简称EDA)
#1、前期准备
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('ggplot') #风格设置近似R这种的ggplot库
import seaborn as sns
sns.set_style('whitegrid')
#%matplotlib inline
from pyecharts import Pie

#忽略弹出的warnings
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei'] #指定默认字体
plt.rcParams['axes.unicode_minus'] = False #解决保存图像是负号‘-’显示为方块的问题

#2、获取数据
data = pd.read_csv('D:\python file\lending club\LoanStats_2017Q2.csv',
                   encoding = 'latin-1',skiprows = 1)
#探索分析数据（EDA）
data.shape #查看表格行数和列数
data.head() #默认前5行
data.iloc[0] #取矩阵当中第一行样本

#处理缺失值
#统计每列属性缺失值的数量
def not_null_count(column):
    column_null = pd.isnull(column) #判断某列属性是否存在缺失值 series
    null = column[column_null] #array
    return len(null)

column_null_count = data.apply(not_null_count)
print(column_null_count)
half_count = len(data)/2 #设定阀值
data = data.dropna(thresh = half_count,axis = 1) #若某一列数据缺失的数量超过阀值被删除
#data = data.drop(['desc','url'],axis = 1) #删除某些加载了网址的URL和描述的列
data.to_csv('D:\python file\lending club\loans_2017q2.csv',
            index = False,encoding='utf-8')

#再次用pandas解析预处理过的数据文件并预览基本信息
loans = pd.read_csv('D:\python file\lending club\loans_2017q2.csv',encoding='utf-8')
loans.shape
loans.head()
#查看数据类型
loans.dtypes

loans.describe()
#数据集的属性较多，我们初步聚焦几个重要特征展开分析，特别是我们最关心的属性贷款状态。
#贷款金额，贷款期限，贷款利率，信用评级，业务发生时间，业务发生所在州，贷款状态，贷款用途
used_col = ['loan_amnt','term','int_rate','grade','issue_d','addr_state',
            'loan_status','purpose','annual_inc','emp_length']
used_data = loans[used_col]
used_data.head()

def not_null_count(column):
    column_null = pd.isnull(column) #判断某列属性是否存在缺失值 series
    null = column[column_null] #array
    return len(null)
column_null_count = used_data.apply(not_null_count)
print(column_null_count)

###单变量分析
#1、贷款状态分布
#处理异常值
#由于loan_status异常值为n的数量和贷款金额较小，因此我们直接删异常值所对应的行
used_data[pd.isnull(used_data.loan_status) == True]
used_data = used_data.drop([105451,105452])
used_data.describe()

#为了更方便分析，将贷款状态进行分类变量编码，主要将贷款状态分为正常和违约
#使用pandas replace函数定义新函数：
def coding(col,codeDict):
    colCoded = pd.Series(col,copy=True)
    for key,value in codeDict.items():
        colCoded.replace(key,value,inplace=True)
        
    return colCoded
 #把贷款状态loanstatus编码为 违约=1，正常=0：
 
pd.value_counts(used_data["loan_status"])
 
used_data["loan_status_code"] = coding(used_data["loan_status"],
         {'Current':0,
          'Fully Paid':0,
          'Late (31-120 days)':1,
          'In Grace Period':1,
          'Late (16-30 days)':1,
          'Charged Off':1,
          'Default':1})

print('\nAfter Coding:')

pd.value_counts(used_data["loan_status_code"])

#Pyecharts 目前支持Numpy 和 Pandas的数据类型，因此需要做数据类型转换
[i for i in pd.value_counts(used_data['loan_status_code'])] 

#用Pyecharts作图
attr = ["正常","违约"]
pie = Pie("贷款状态占比")
pie.add("",attr,
        [int(i) for i in pd.value_counts(used_data["loan_status_code"])],
        is_label_show=True)
pie
pie.show_config()
pie.render('D:\\python file\\out.html')   

#2、贷款金额分布
plt.figure(figsize=(18, 9))
sns.set()
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth":2 } )
sdisplot_loan = sns.distplot(used_data['loan_amnt'] )
plt.xticks(rotation=90)
plt.xlabel('Loan amount')
plt.title('Loan amount\'s distribution')
sdisplot_loan.figure.savefig("Loan_amount")

#3、贷款期限分布
pd.value_counts(loans["term"])  # 分类统计贷款期限
[i for i in pd.value_counts(loans["term"])]  #数据转换
# 贷款期限占比可视化
attr = ["36个月", "60个月"]
pie = Pie("贷款期限占比")
pie.add("",attr,
        [float(i) for i in pd.value_counts(loans["term"])],
        is_label_show=True)
pie
pie.render('D:\\python file\\out2.html')  

#4、贷款产品用途种类比较
used_data['purpose'].value_counts()# 按借款用途统作统计
plt.figure(figsize=(18, 9))
sns.set()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
total = float(len(loans.index))
ax = sns.countplot(x="purpose", data=used_data, palette="Set2")
plt.xticks(rotation=90)
plt.title('Purpose')
plt.show()
ax.figure.savefig("Purpose")

#5、客户信用等级占比
used_data['grade'].value_counts()
attr = ["C", "B","A","D","E","F","G"]
pie = Pie("信用等级比例")
pie.add("",attr,
        [float(i) for i in pd.value_counts(loans["grade"])],
        is_label_show=True)
pie
pie.render('D:\\python file\\lending club\\out3.html') 

#6、贷款利率种类分布
#数据转换
used_data['int_rate_num']=used_data['int_rate'].str.rstrip("%").astype("float") 
used_data.tail() #发现空值
#used_data[pd.isnull(used_data.int_rate) == True]
used_data.dropna(inplace=True) #处理空值
used_data.tail() #再次检查
used_data.describe()

plt.figure(figsize=(18, 9))
sns.set()
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth":2 } )
sdisplot_loan = sns.distplot(used_data['int_rate_num'] )
plt.xticks(rotation=90)
plt.xlabel('Interest Rate')
plt.title('Interest Rate\'s distribution')
sdisplot_loan.figure.savefig("Interest Rate")

###多维变量分析
#1、探索贷款与时间的关系
used_data['issue_d2'] = pd.to_datetime(used_data['issue_d'])
used_data.head()

data_group_by_date = used_data.groupby(['issue_d2']).sum()
data_group_by_date.reset_index(inplace=True)
# 新增月份列
data_group_by_date['issue_month'] = data_group_by_date['issue_d2'].apply(lambda x: x.to_period('M'))
#按月份统计贷款金额 
loan_amount_group_by_month = data_group_by_date.groupby('issue_month')['loan_amnt'].sum()
# 输出结果转成DataFrame
loan_amount_group_by_month_df = pd.DataFrame(loan_amount_group_by_month).reset_index() 
loan_amount_group_by_month_df

# 可视化
plt.figure(figsize=(15, 9))
sns.set()
sns.set_context("notebook", font_scale=1, rc={"lines.linewidth": 2})
plot1 = sns.barplot(x='issue_month', y= 'loan_amnt', data = loan_amount_group_by_month_df,)
plt.xlabel('Month')
plt.ylabel('Loan_amount')
plt.title('Mounth VS Loan_amount')
plot1.figure.savefig("Mounth VS Loan_amount.png")

#2、探索贷款金额与州之间的关系
# 按州统计贷款金额
data_group_by_state = used_data.groupby(['addr_state'])['loan_amnt'].sum() 
# 将结果转为 dataframe
data_group_by_state_df= data_group_by_state.reset_index() 

sns.set()
plt.figure(figsize=(15, 9))
sns.set_context("notebook",font_scale=1,rc={"lines.linewidth":5})
sbarplot=sns.barplot(y='loan_amnt',x='addr_state',data=data_group_by_state_df)
plt.xlabel('State')
plt.ylabel('Loan_amount')
plt.xticks(rotation=90)
plt.title('State VS Loan_amount')
sbarplot.figure.savefig("State VS Loan_amount")

#3、探索信用评级、贷款期限和利率的关系
used_data['int_rate_num']= used_data['int_rate'].str.rstrip("%").astype("float")
data_group_by_grade_term = used_data.groupby(['grade', 'term'])['int_rate_num'].mean()
data_group_by_grade_term_df = pd.DataFrame(data_group_by_grade_term).reset_index()
data_group_by_grade_term_pivot = data_group_by_grade_term_df.pivot(index='grade',
                                                                   columns='term', values='int_rate_num')
data_group_by_grade_term_pivot  #  输出数据透视表

# 查看信用评级的分布
used_data['grade'].value_counts() 

#4、探索贷款用途与利率的关系
plt.figure(figsize=(15, 9))
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})
loans['int_rate_num'] = loans['int_rate'].str.rstrip("%").astype("float")
sboxplot = sns.boxplot(y="purpose", x="int_rate_num", data=loans)
sns.despine(top=True)
plt.xlabel('Interest_Rate')
plt.ylabel('Purpose')
plt.xticks(rotation=90)
plt.show()
sboxplot.figure.savefig("Purpose VS Rate")

#5、探索贷款金额与利率之间的关系
plt.figure(figsize=(15, 9))
j_plot = sns.jointplot("loan_amnt","int_rate_num",
                       data=used_data, kind="reg",size=10)
j_plot.savefig("Loan amount VS Interest Rate")

#6、探索贷款利率与违约次数之间的关系
plt.figure(figsize=(15, 9))
sns.set_context("notebook", font_scale=1.2, rc={"lines.linewidth": 2.5})
sboxplot2 = sns.boxplot(x="delinq_2yrs", y="int_rate_num", data=loans)
sns.despine(top=True)
plt.xticks(rotation=90)
plt.title('Interest Rate VS Delinq_2yrs')
sboxplot2.figure.savefig("Interest Rate VS Delinq_2yrs")

#7、探索利率、收入、工作年限以及贷款状态之间的关系
#替换变量的第二种方法，创建mapping
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
    }
}

used_data.head()
used_data['annual_inc'].value_counts() 
used_data = used_data.replace(mapping_dict)
# 数据转换
used_data["annual_inc"]=used_data["annual_inc"].replace(",","").astype("float").dropna()

#数据可视化
sns.set_context("notebook",font_scale=3,rc={"lines.linewidth": 2.5})
p_plot = sns.pairplot(used_data, vars=["int_rate_num","annual_inc", "emp_length"], hue="loan_status_code", diag_kind="kde" ,kind="reg", size = 7)
p_plot.savefig("Interest Rate VS Annual Income VS Emp_length")

###总结
used_data.corr() #计算相关系数

# 相关系数图
names = ['loan_amnt','annual_inc','emp_length','Loan_Status_Coded',
         'int_rate' ] #设置变量名
correlations = used_data.corr()
# plot correlation matrix
plt.figure(figsize=(19, 9))
fig = plt.figure() #调用figure创建一个绘图对象
ax = fig.add_subplot(111)
cax = ax.matshow(correlations, vmin=-1, vmax=1)  #绘制热力图，从-1到1
fig.colorbar(cax)  #将matshow生成热力图设置为颜色渐变条
ticks = np.arange(0,5,1) #生成0-5，步长为1
ax.set_xticks(ticks)  #生成刻度
ax.set_yticks(ticks)
ax.set_xticklabels(names) #生成x轴标签
ax.set_yticklabels(names)
plt.xticks(rotation=90)
fig.savefig("Corr")
plt.show()