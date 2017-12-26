# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 22:13:13 2017

@author: yaohaiying
"""

from pylab import *

#1：普通图
subplot(3,3,1)

n = 256
X = np.linspace(-np.pi,np.pi,n,endpoint=True)
Y = np.sin(2*X)

#plt.axes([0.025,0.025,0.95,0.95])

plot (X, Y+1, color='blue', alpha=1.00)
plt.fill_between(X, 1, Y+1, color='blue', alpha=.25)

plot (X, Y-1, color='blue', alpha=1.00)
plt.fill_between(X, -1, Y-1, (Y-1) > -1, color='blue', alpha=.25)
plt.fill_between(X, -1, Y-1, (Y-1) < -1, color='red',  alpha=.25)

plt.xlim(-np.pi,np.pi), plt.xticks([])
plt.ylim(-2.5,2.5), plt.yticks([])
# savefig('../figures/plot_ex.png',dpi=48)

#2：散点图
subplot(3,3,2)
n = 20
X = np.random.normal(0,1,n)#给出均值为0，标准差为1的n个高斯随机数
Y = np.random.normal(0,1,n)
colors = np.random.rand(n)#Random values in a given shape.
#print(colors)
area = np.pi * (15 * np.random.rand(n)) ** 2

scatter(X,Y,s=area,c=colors,alpha=0.5)

#3：条形图
subplot(3,3,3)
n = 12
X = np.arange(n)
Y1 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)
Y2 = (1-X/float(n)) * np.random.uniform(0.5,1.0,n)

bar(X,+Y1,facecolor = '#9999ff',edgecolor = 'white')
bar(X,-Y2,facecolor = '#ff9999',edgecolor = 'white')

for x,y in zip(X,Y1):
    text(x+0.4,y+0.05,'%.2f' % y,ha='center',va='bottom')
    
for x,y in zip(X,-Y2):
    text(x+0.4,y+0.05,'%.2f' % y,ha='center',va='top')    
ylim(-1.25,+1.25)

#4:等高线图
subplot(3,3,4)
def f(x,y):return(1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 256
x = np.linspace(-3,3,n)
y = np.linspace(-3,3,n)
X,Y = np.meshgrid(x,y)

contourf(X,Y,f(X,Y),8,alpha=.75,cmap='jet')
C = contour(X,Y,f(X,Y),8,colors = 'black',linewidth = .5)

#5:灰度图
subplot(3,3,5)
def f(x,y):return(1-x/2+x**5+y**3)*np.exp(-x**2-y**2)

n = 10
x = np.linspace(-3,3,4*n)
y = np.linspace(-3,3,3*n)
X,Y = np.meshgrid(x,y)
imshow(f(X,Y))

#6:饼状图
subplot(3,3,6)
# Pie chart, where the slices will be ordered and plotted counter-clockwise:
labels = 'Frogs', 'Hogs', 'Dogs', 'Logs'
sizes = [15, 30, 45, 10]
explode = (0, 0.1, 0, 0)  # only "explode" the 2nd slice (i.e. 'Hogs')

pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
        shadow=True, startangle=90)
axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

#7：量场图
subplot(3,3,7)
n = 8
X,Y = np.mgrid[0:n,0:n]
quiver(X,Y)

#8：网格
subplot(3,3,8)
axes = gca()
axes.set_xlim(0,4)
axes.set_ylim(0,3)
axes.xaxis.set_major_locator(plt.MultipleLocator(1.0))
axes.xaxis.set_minor_locator(plt.MultipleLocator(0.1))
axes.yaxis.set_major_locator(plt.MultipleLocator(1.0))
axes.yaxis.set_minor_locator(plt.MultipleLocator(0.1))
axes.grid(which='major', axis='x', linewidth=0.75, linestyle='-', color='0.75')
axes.grid(which='minor', axis='x', linewidth=0.25, linestyle='-', color='0.75')
axes.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.75')
axes.grid(which='minor', axis='y', linewidth=0.25, linestyle='-', color='0.75')
axes.set_xticklabels([])
axes.set_yticklabels([])

#
subplot(3,3,9)
#ax = plt.axes([0.025,0.025,0.95,0.95], polar=True)

N = 20
theta = np.arange(0.0, 2*np.pi, 2*np.pi/N)
radii = 10*np.random.rand(N)
width = np.pi/4*np.random.rand(N)
bars = plt.bar(theta, radii, width=width, bottom=0.0)

for r,bar in zip(radii, bars):
    bar.set_facecolor( plt.cm.jet(r/10.))
    bar.set_alpha(0.5)

ax.set_xticklabels([])
ax.set_yticklabels([])

show()


