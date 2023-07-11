# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 14:24:43 2023

@author: Pooja
CHapter 5
Codes in a sequence as mentioned in the chapter
"""
#subplot and plot function
import matplotlib.pyplot as plt
import numpy as np
x=[1,2,3,4]
y=[2,4,6,8]
fig, ax = plt.subplots()
ax.plot(x,y, color ='blue', linestyle ='dashed', marker='o', linewidth=2.0)
ax.set(xlim=(0,5), xticks=np.arange(1,5),
       ylim=(0,10),yticks=np.arange(1,10))
plt.show()

#------------------------------
#adding xlabel, ylabel and title
import matplotlib.pyplot as plt
import numpy as np
x=[1,2,3,4]
y=[2,4,6,8]
fig, ax = plt.subplots()
ax.plot(x,y, color ='blue', linestyle ='dashed', marker='o', linewidth=2.0)
ax.set(xlim=(0,5), xticks=np.arange(1,5),
       ylim=(0,10),yticks=np.arange(1,10),
       title="Plot of x versus y",
       xlabel="Values of x", ylabel="Values of y")
plt.show()
#-----------------------------------
#adding face color and annotation
import matplotlib.pyplot as plt
import numpy as np
x=[1,2,3,4]
y=[2,4,6,8]
fig, ax = plt.subplots()
ax.plot(x,y, color ='blue', linestyle ='dashed', marker='o', linewidth=2.0)
ax.set(xlim=(0,5), xticks=np.arange(1,5),
       ylim=(0,10),yticks=np.arange(1,10),
       title="Plot of x versus y",
       xlabel="Values of x", ylabel="Values of y", facecolor="yellow")
ax.annotate('3,6 point', xy=(3.1, 5.9), xytext=(3.5,5.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()
#-------------------------------------
#including grid and tick parameters
import matplotlib.pyplot as plt
import numpy as np
x=[1,2,3,4]
y=[2,4,6,8]
fig, ax = plt.subplots()
ax.plot(x,y, color ='blue', linestyle ='dashed', marker='o', linewidth=2.0)
ax.set(xlim=(0,5), xticks=np.arange(1,5),
       ylim=(0,10),yticks=np.arange(1,10),
       title="Plot of x versus y",
       xlabel="Values of x", ylabel="Values of y")
ax.annotate('3,6 point', xy=(3.1, 5.9), xytext=(3.5,5.5),
            arrowprops=dict(facecolor='black', shrink=0.05))
ax.grid(True, linestyle='--')
ax.tick_params(labelcolor='r', labelsize='medium',width=3)
plt.show()
#------------------------------------------
#Line Plot
import matplotlib.pyplot as plt
import numpy as np

#Sample data
x=np.linspace(10,20,100)
y=np.cos(x)

#Plot the data
plt.plot(x,y)

#Add labels and title
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Cosine wave')

#Show the plot
plt.show()

#-------------------------------------------
#Bar plot
import matplotlib.pyplot as plt

#Sample data
categories = ['A', 'B', 'C', 'D']
values = [1, 4, 2, 5]

#Plot the data
plt.bar(categories, values)

#Add labels and title
plt.xlabel('Categories')
plt.ylabel('Values')
plt.title('Bar Chart Example')

#Show the plot
plt.show()

#-------------------------------------------
#Histogram
import matplotlib.pyplot as plt
import pandas as pd

#Load the Titanic dataset
df = pd.read_csv('titanic.csv')

#Plot the histogram
plt.hist(df.age, bins=20, edgecolor='black', alpha =0.7)

#Add labels and title
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram of Titanic Passengers \'Age\'')

#Show the plot
plt.show()

#-------------------------------------------
#Scatter Plot
import matplotlib.pyplot as plt
import pandas as pd

#Load the Titanic dataset
df = pd.read_csv('titanic.csv')

#Select the 'Age' and 'Fare' columns from dataset
age = df.age.fillna(df.age.max())
fare = df.fare.fillna(df.fare.max())

#Plot the Scatter Plot
plt.scatter(age, fare, alpha =0.7)

#Add labels and title
plt.xlabel('Age')
plt.ylabel('Fare')
plt.title('Scatter Plot of Titanic Passengers \'Age\' and \'Fare\'')

#Show the plot
plt.show()

#-------------------------------------------
#Pie Plot
import matplotlib.pyplot as plt
import pandas as pd

#Load the Titanic dataset
df = pd.read_csv('titanic.csv')

#Calculate the proportion of Survivors
survived = df['survived'].value_counts()
proportion = survived / df['survived'].sum()

#Plot the Pie Chart
plt.pie(proportion, labels =["Died", "Survived"], autopct ='%1.1f%%')

#Add title
plt.title('Proportion of Passengers Who Survived the Titanic')

#Show the plot
plt.show()

#-------------------------------
#Area Plot
import matplotlib.pyplot as plt
import numpy as np

#Sample Data
categories =['Category 1', 'Category 2', 'Category 3']
quantities = [10,20,30]

#Plot the data as an Area Plot
plt.fill_between(categories, quantities, alpha=0.5)

#Add labels and title
plt.xlabel('Categories')
plt.ylabel('Quantities')
plt.title('Area plot Example')

#Show the plot
plt.show()

#-------------------------------------------
#Box Plot
import matplotlib.pyplot as plt
import pandas as pd

#Load the Titanic dataset
df = pd.read_csv('titanic.csv')

#Plot Box Plot on Age of Passengers, Grouped by their Survival Status
x=df[df['survived']==0]['age'].dropna()
x=df[df['survived']==1]['age'].dropna()
plt.boxplot([x,y], labels = ['Did not Survive', 'Survived'])

#Add labels and title
plt.xlabel('Survived Status')
plt.ylabel('Age')
plt.title('Box Plot of Age by Survival Status')

#Show the plot
plt.show()

#-------------------------------------
#modified appearance of Box Plot
plt.boxplot([x,y], labels = ['Did not Survive', 'Survived'],
            patch_artist=True, sym='+', notch=True)


#---------------------------------
#3D plotting - Scatter Plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Sample Data
x = np.random.standard_normal(100)
y = np.random.standard_normal(100)
z = np.random.standard_normal(100)

#Plot the Data
ax.scatter(x, y, z)

#Set Labels for the axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

#Show the plot
plt.show()

#---------------------------------
#3D plotting - Surface plot
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

#Sample Data
x = np.linspace(-5,5,100)
y = np.linspace(-5,5,100)
x, y  = np.meshgrid(x, y)
z = np.sin(np.sqrt(x**2 + y**2))

#Plot the Surface plot
ax.plot_surface(x, y, z, cmap = 'viridis')

#Set Labels for the axes
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

#Show the plot
plt.show()
#------------------------
#Surface plot with cmap as Blues
ax.plot_surface(x, y, z, cmap = 'Blues')


#-----------------------------------------
#Using mesgrid function code
import numpy as np

# Generate two 1D arrays
x = np.linspace(-5, 5, 10)
y = np.linspace(-5, 5, 10)

# Use np.meshgrid to generate a 2D grid
X, Y = np.meshgrid(x, y)

print(X)
print(Y)

#--------------------------------
#Saving Plots syntax
plt.savefig(filename, dpi=Nonee, facecolor='w', edgecolor='w',
            orientation='portrait', papertype =None, format= None,
            transparent=False, bbox_inches=None, pad_inches=0.1,
            frameon=None, metadata=None)

#--------------------------------
#Saving Plot as png
plt.plot([1,2,3,4])
plt.savefig("example.png", facecolor='red', edgecolor='blue')
 
#--------------------------------
#Saving Plot as png with landscape orientation
plt.plot([1,2,3,4])
plt.savefig("example1.png", orientation='landscape',
            facecolor='red', edgecolor='blue')

#--------------------------------
#Saving Plot as png with papertype as legal
plt.plot([1,2,3,4])
plt.savefig("example2.png",papertype='legal')

#--------------------------------
#Saving Plot as png with transparent background
plt.plot([1,2,3,4])
plt.savefig("example3.png",transparent = True)

#-----------------------
#saving as png
plt.savefig("plot.png")

#-----------------------
#saving as pdf
plt.savefig("plot.pdf")

#--------------------------------
#Saving Plot as png - sepecifing  dpi, bbox_inches, pad_inches
plt.plot([1,2,3,4])
plt.savefig("example4.png",dpi=300,bbox_inches='tight', 
            pad_inches=0.5,transparent = True)


#--------------------------------------------------
#ANNEXURE PANDAS
#--------------------------------------------------
#creating a series of integers
import pandas as pd
s= pd.Series([1, 3, 5, np.nan, 6, 8])
print(s)

#----------------
#creating DataFrame
import pandas as pd

#create a dictoinary of data
data ={'Name' : ['John', 'Jane', 'Jim', 'Joan'],
       'Age' : [32, 28, 41, 37],
       'Country' : ['USA', 'UK', 'Canada', 'Australia']}

#creating dataframe from dictionary
df = pd.DataFrame(data)
print(df)


#---------------------
#Accessing a column from the dataframe
age = df['Age']
print(age)

#------------------------
#Access a row in a dataframe using  indexing
row = df.loc[0]
print(row)

#----------------------------
#Filter rows in a dataframe based on a condition
filtered_df = df[df['Age']>35]
print(filtered_df)


#------------------------------------
#Group data in a dataframe by a column and aggregate the results
grouped_df = df.groupby(['Country']).mean()
print(grouped_df)

#-----------------------------------
#Merge two dataframes based on a common column

df1 = pd.DataFrame ({'Key' : ['K0','K1','K2','K3'],
                     'A' : ['A0','A1','A2','A3'],
                     'B' : ['B0', 'B1','B2','B3']})

df2 = pd.DataFrame ({'Key' : ['K0','K1','K2','K3'],
                     'C' : ['C0','C1','C2','C3'],
                     'D' : ['D0', 'D1','D2','D3']})

merged_df = pd.merge(df1, df2, on='Key')
print(merged_df)

#--------------------------------------------------
#ANNEXURE NUMPY
#--------------------------------------------------

import numpy as np

#creat two arrays
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])

print("a:\n",a)
print("b:\n",b)

#perform element-wise addition
c = a + b
print("element-wise addition of a and b is : \n", c)


#perform element-wise multiplication
d = a * b
print("element-wise multiplication of a and b is : \n", d)

#----------------------
#matrix multiplication with scalar, transpose, determinant

import numpy as np

#create a 2X2 matrix
A = np.array([[1,2],[3,4]])

#Multiply the matrix iwth a scalar
B = 2* A
print("B: \n", B)

# Transpose the matrix
C = A.T
print("C : \n", C)

#Calculate the determinat of the matrix
det = np.linalg.det(A)
print("determinant of A is :\n",det)

#creating an array of ones
ones = np.ones((3,3))
print(ones)

#creating an array of zeros
zeros = np.zeros((3,3))
print(zeros)

#Create an array with range of values
rang = np.arange(0,10)
print(rang)
