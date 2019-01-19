# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as py
import matplotlib.pyplot as plt

#a plot is something which has x axis and y axis, and we plot something on it

#Example with hard coded values for x and y
plt.plot([1,2,3],[5,7,6])
plt.show()


#Eample with labels and variables having values for x and y
x = [1,2,3]
y = [5,7,6]
plt.plot(x,y)
plt.xlabel = "x-->"
plt.ylabel = "y-->"
plt.show()


#Example with multiple lines
x1 = [1,2,3]
y1 = [5,7,6]

x2 = [2,3,4]
y2 = [1,9,6]

plt.plot(x1,y1, label = "first line")
plt.plot(x2,y2, label = "second line")
plt.xlabel = "x-->"
plt.ylabel = "y-->"
plt.legend()            #Just need to invoke the legend and pass the parameters in plot()
plt.show()


#Example for bar chart
x = [1,2,3,4,5,6,7,8,9]
y = [11,44,34,6,78,342,67,12,9]
plt.bar(x,y)
plt.show()

#Example of bar chart with legend
x = [1,2,3,4,5,6,7,8,9]
y1 = [11,44,34,6,78,342,67,12,9]
y2 = [33,12,65,132,74,45,96,54,19]

plt.bar(x,y1, label = "y1")
plt.bar(x,y2, label = "y2")

plt.legend()
plt.show()


#Exmple of scatter plot
x = [1,2,3,4,5,6,7,8,9]
y = [11,44,34,6,78,342,67,12,9]
plt.scatter(x, y)
plt.scatter(x, y, color = 'black', label = "scatterplot")
plt.show()

#Example of stack charts
x = [1,2,3,4,5,6,7,8,9]
y1 = [11,44,34,6,78,342,67,12,9]
y2 = [33,12,65,132,74,45,96,54,19]
y3 = [9,82,0,897,27,8,9,69,86]

plt.stackplot(x, y1,y2,y3, colors=['red','green','magenta'])
plt.show()

#Example of stack charts with legtends (which is kind a hack to show what is what)
x = [1,2,3,4,5,6,7,8,9]
y1 = [11,44,34,6,78,342,67,12,9]
y2 = [33,12,65,132,74,45,96,54,19]
y3 = [9,82,0,897,27,8,9,69,86]

plt.plot([], [], color='red', label = 'RED', linewidth = 10)
plt.plot([], [], color='green', label = 'GREEN')
plt.plot([], [], color='magenta', label = 'MAGENTA')

plt.stackplot(x, y1,y2,y3, colors=['red','green','magenta'])
plt.legend()
plt.show()


#Example of pie chart
status = ['pass', 'fail','pass with grade','witheld','absent']
values = [12,45,10,9, 10]
cols = ['red','green','blue','magenta','cyan']

plt.pie(values, labels=status, 
        colors=cols, 
        explode=(0,0,1,0,0),
        startangle=90,
        autopct='%1.1f%%') 
plt.show()