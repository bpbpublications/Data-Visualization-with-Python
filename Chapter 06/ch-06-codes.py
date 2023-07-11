# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 17:54:34 2023

@author: Pooja
Chapter 6 
codes as per their sequence in the chapter
"""
#importing seaborn library
import seaborn as sns

#documentation
help(sns)

#help on scatterplots from seaborn
help(sns.scatterplot)

#seaborn datasets
import seaborn as sns
# Load the "tips" dataset
tips = sns.load_dataset("tips")
#---------------------------
#get_dataset_names()
import seaborn as sns
sns.get_dataset_names()
#---------------------------

#figure and subplots
import seaborn as sns
import matpotlib.pyplot as plt

#Create a figure and a set of subplots
fig, ax = plt.subplots()
#---------------------------

#Line Plot
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Create a figure and a set of subplots
fig, ax = plt.subplots()

#Synthetic data creation
data_dict = {'year': [2000, 2001, 2002, 2003, 2004],
             'pop': [6.1, 6.3, 6.5, 6.7, 6.9]}
# Create a DataFrame from the dictionary
my_data = pd.DataFrame(data_dict)
# Create a dark color palette
palette = sns.dark_palette("red", as_cmap=True)

# Plot the data using seaborn lineplot
sns.lineplot(x='year', y='pop', data=my_data, ax=ax, palette=palette)

# Set the title of the figure
ax.set_title('Population Growth Over Time')

# Set the ticks, labels for the x and y axes
ax.set_xticks([2000, 2001, 2002, 2003, 2004])
ax.set_xlabel('Year')
ax.set_ylabel('Population (millions)')

# Display the plot
plt.show()



#----------------------------
#Two plots in one figure using subplot#
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# Load the Titanic dataset
titanic = pd.read_csv('titanic.csv')
# Create a figure with two subplots
fig = plt.figure(figsize=(10, 5))
# First subplot: survival rate by sex
ax1 = fig.add_subplot(1, 2, 1)
sns.barplot(x='sex', y='survived', data=titanic, ax=ax1)
ax1.set_title('Survival Rate by Sex')
# Second subplot: survival rate by passenger class
ax2 = fig.add_subplot(1, 2, 2)
sns.barplot(x='pclass', y='survived', data=titanic, ax=ax2)
ax2.set_title('Survival Rate by Passenger Class')
# Adjust the layout
fig.tight_layout()
# Show the plot
plt.show()

#-----------------------------
#palettes
import seaborn as sns
import matplotlib.pyplot as plt
# Create a matrix of values to display
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# Create a dark color palette
palette = sns.dark_palette("red", as_cmap=True)
# Plot the heatmap with the matrix and color palette
sns.heatmap(matrix, cmap=palette)
# Display the plot
plt.show()

#-----------------------------
#bar plot
import seaborn as sns
import matplotlib.pyplot as plt
# Load the tips dataset
tips = sns.load_dataset("tips")
# Create a bar plot using Seaborn
sns.barplot(x="day", y="total_bill", data=tips, 
            hue="sex", ci=None, palette="Set2")

# Set the title and axis labels of the plot
plt.title("Total Bill Amount by Day of the Week and Gender")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill Amount ($)")
# Show the plot
plt.show()

#bar plot with othr parametrs 
sns.barplot(x='day', y='total_bill', data=tips, 
            capsize=0.5, estimator=sum, 
            order=['Thur', 'Fri', 'Sat', 'Sun'],)

# Set the title and axis labels of the plot
plt.title('Average total bill amount by day of the week')
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill Amount ($)")
# Show the plot
plt.show()


#--------------------------------
#count plot
import seaborn as sns
import matplotlib.pyplot as plt
# Load the Tips dataset
tips = sns.load_dataset("tips")
# Create a count plot of the number of customers by day of the week
sns.countplot(x="day", data=tips)
# Set plot title and axes labels
plt.title("Number of Customers by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Customers")
# Show the plot
plt.show()

# Create a count plot of the number of customers by day of the week
sns.countplot(x="day", hue="sex", data=tips, palette="pastel",
             order=["Thur", "Fri", "Sat", "Sun"])

# Set plot title and axes labels
plt.title("Number of Customers by Day of the Week")
plt.xlabel("Day of the Week")
plt.ylabel("Number of Customers")
# Show the plot
plt.show()
#------------------------------
# boxplot
import seaborn as sns
import matplotlib.pyplot as plt
# Load the tips dataset
tips = sns.load_dataset("tips")
# Create a box plot using Seaborn
sns.boxplot(x="day", y="total_bill", data=tips, 
            hue="sex", palette="Set2", notch=True)

# Set the title and axis labels of the plot
plt.title("""Distribution of Total Bill Amount 
          by Day of the Week and Gender""")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill Amount ($)")
# Show the plot
plt.show()

# Create a box plot with custom parameters
sns.boxplot(x="day", y="total_bill", hue="sex", data=tips,
            showfliers=False, whis=[5, 95], width=0.5)

# Set the title and axis labels of the plot
plt.title("""Distribution of Total Bill Amount 
          by Day of the Week and Gender""")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill Amount ($)")
# Show the plot
plt.show()

#------------------------------
# Voilin plot
import seaborn as sns
import matplotlib.pyplot as plt
# Load the tips dataset
tips = sns.load_dataset("tips")
# Create a violin plot using Seaborn
sns.violinplot(x="day", y="total_bill", data=tips, 
               hue="sex", split=True, inner="stick", palette="Set2")
 
# Set the title and axis labels of the plot
plt.title("Distribution of Total Bill Amount by Day of the Week and Gender")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill Amount ($)")
# Show the plot
plt.show()

 
# Create a violin plot with custom parameters
sns.violinplot(x="day", y="total_bill", hue="sex", data=tips,
                bw=0.3, cut=0, scale="width")


# Set the title and axis labels of the plot
plt.title("Distribution of Total Bill Amount by Day of the Week and Gender")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill Amount ($)")
# Show the plot
plt.show()
# ------------------------
# strip plot 
import seaborn as sns
import matplotlib.pyplot as plt
# Load the tips dataset
tips = sns.load_dataset("tips")
# Create a strip plot using Seaborn
sns.stripplot(x="day", y="total_bill", 
              data=tips, jitter=True, hue="sex", palette="Set2")
# Set the title and axis labels of the plot
plt.title("Total Bill Amount by Day of the Week and Gender")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill Amount ($)")
# Show the plot
plt.show()

#------------------------------
#Swarm Plot
import seaborn as sns
import matplotlib.pyplot as plt
# Load the tips dataset
tips = sns.load_dataset("tips")
# Create a swarm plot using Seaborn
sns.swarmplot(x="day", y="total_bill", 
              data=tips, hue="sex", palette="Set2")

# Set the title and axis labels of the plot
plt.title("Total Bill Amount by Day of the Week and Gender")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill Amount ($)")
# Show the plot
plt.show()

# swarm plot with customizations
sns.swarmplot(x="day", y="total_bill", data=tips, 
              hue="sex", palette="Set2", dodge=True,
              order=["Sun", "Sat", "Fri", "Thur"])
# Set the title and axis labels of the plot
plt.title("Total Bill Amount by Day of the Week and Gender")
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill Amount ($)")
# Show the plot
plt.show()
#------------------------------
#catplot()
import seaborn as sns
import matplotlib.pyplot as plt
# Load the tips dataset
tips = sns.load_dataset("tips")
# Create a box plot using Seaborn
sns.catplot(x="day", y="total_bill", 
            hue="sex", kind="box", data=tips)

# Show the plot
plt.show()

# Create a box plot using Seaborn
sns.catplot(x="day", y="total_bill", 
            hue="sex", kind="boxen", data=tips)
#------------------------------
#facetgrid
import seaborn as sns
import matplotlib.pyplot as plt
# Load the tips dataset
tips = sns.load_dataset("tips")
# Create a grid of scatter plots, one for each day of the week
g = sns.FacetGrid(data=tips, col="day", col_wrap=2,)
# Add the scatter plots to the grid
g = g.map(sns.scatterplot, "total_bill", "tip")
# Add titles to each subplot
g = g.set_titles("{col_name}")
# Add margin titles to the grid
g = g.set_axis_labels("Total Bill", "Tip")
# Show the plot
plt.show()
#------------------------------
#Scatter plot
import seaborn as sns
import matplotlib.pyplot as plt
# Load the tips dataset
tips = sns.load_dataset("tips")
# Create a scatter plot of total_bill vs tip
sns.scatterplot(x="total_bill", y="tip", data=tips, 
                hue="sex", style="time", 
                size="size", alpha=0.8)
# Set the title and axes labels
plt.title("Total Bill vs Tip")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
# Customize the legend
legend_labels = ["Male - Lunch", "Male - Dinner", 
                 "Female - Lunch", "Female - Dinner"]
plt.legend(title="Legend", labels=legend_labels, 
           loc="upper left", frameon=True)
# Show the plot
plt.show()
#------------------------------
#-Line plot
import seaborn as sns
import matplotlib.pyplot as plt

# Load the tips dataset
tips = sns.load_dataset("tips")

# Create a bar plot of total bill by day of the week
sns.lineplot(x="day", y="total_bill", data=tips, 
             hue='sex',hue_order=(['Male','Female']),alpha=0.8)

# Add labels to the plot
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill")
plt.title("Total Bill by Day of the Week")
plt.show()

# Create a line plot of total_bill vs tip, 
#with separate lines for each gender
sns.lineplot(x="total_bill", y="tip", hue="sex", 
             hue_order=["Male", "Female"], data=tips)
# Add a title and axis labels
plt.title("Tip vs Total Bill by Gender")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
# Show the plot
plt.show()
#------------------------------
#Heatmap
import seaborn as sns
import matplotlib.pyplot as plt
import pandas
#Fetching tips dataset from the seaborn
tips = sns.load_dataset("tips")
#creating correlation matrix on whole dataset
corr_matrix = tips.corr()
#Developing heatmap plot on correlation matrix
sns.heatmap(corr_matrix, annot=True, cmap="YlGnBu")
plt.show()
#------------------------------
#joint plot
import seaborn as sns
import matplotlib.pyplot as plt
#loading data
tips = sns.load_dataset("tips")
#creating join on total_bill and tip
sns.jointplot(x="total_bill", y="tip", data=tips)
plt.show()

# Create a joint plot between "total_bill" and "tip" 
#with a regression line
sns.jointplot(data=tips, x="total_bill", y="tip",kind='reg')

# Show the plot
plt.show()

# Create a joint plot between "total_bill" and "tip" 
#with a regression line and with different plot and line color
sns.jointplot(data=tips, x="total_bill", y="tip",color ='Purple',kind='reg', line_kws={'color':'red'})

# Show the plot
plt.show()

# Create a joint plot between "total_bill" and "tip" 
#with a regression line and with different plot and line color an iffrnt histogram color
sns.jointplot(data=tips, x="total_bill", y="tip",color ='Purple',kind='reg', line_kws={'color':'red'},
              marginal_kws = {'color':'yellow'})

# Show the plot
plt.show()
#------------------------------
#PAir Plot
import seaborn as sns
import matplotlib.pyplot as plt

# Load the tips dataset
tips = sns.load_dataset("tips")
# Create a pairplot
sns.pairplot(tips)
plt.show()
# Create a pairplot with KDE at diagnol
sns.pairplot(tips, diag_kind ='kde')
plt.show()

# Create a pairplot with KDE at diagnol and regression plot for others
sns.pairplot(tips, kind='reg', diag_kind ='kde')
plt.show()

# Create a pairplot with vars 
sns.pairplot(tips, kind = 'reg', diag_kind="kde", 
             vars={"total_bill", "tip"}, x_vars=["total_bill"], 
             y_vars=["tip"],plot_kws={'color':'green'})
plt.show()


# Create a pairplot with vars an hue_order
sns.pairplot(tips, kind = 'reg', diag_kind="kde", 
             hue="sex", hue_order=['Male','Female'],
             vars={"total_bill", "tip"}, x_vars=["total_bill"], 
             y_vars=["tip"],plot_kws={'color':'green'})
plt.show()

#------------------------------
#Distribution plot
import seaborn as sns
import matplotlib.pyplot as plt
# load the tips dataset
tips = sns.load_dataset('tips')
# create a distribution plot of the total bill
sns.displot(tips['total_bill'], kde=True,color = 'purple',)
# show the plot
plt.show()

# Create a distribution plot with row_order, and log_scale
displ = sns.displot(data=tips, x="total_bill", col="sex",
    row="day", hue="time", kind="kde",log_scale=True,
    row_order=["Thur", "Fri", "Sat", "Sun"],
    facet_kws=dict(margin_titles=True),)
# Add titles and axis labels
displ.fig.suptitle("Distribution of Total Bill by Day and Sex")
displ.set_xlabels("Total Bill")
displ.set_ylabels("Density")
# Show the plot
plt.show()

#------------------------------
#Regression PLOT
import seaborn as sns
import matplotlib.pyplot as plt
# load the tips dataset
tips = sns.load_dataset("tips")
# create a regression plot using the total bill 
#as the x-variable and the tip amount as the y-variable
sns.regplot(x="total_bill", y="tip", data=tips, 
            fit_reg=True, ci=95, scatter_kws={"s": 50, "alpha": 0.5},
            line_kws={"color": "red", "linewidth": 3})

# set plot title and axes labels
plt.title("Regression plot for tips dataset")
plt.xlabel("Total bill")
plt.ylabel("Tip amount")
# show the plot
plt.show()

#------------------------------
###SAVING plot
import seaborn as sns
import matplotlib.pyplot as plt

# Load the tips dataset
tips = sns.load_dataset("tips")

# Create a bar plot of total bill by day of the week
sns.barplot(x="day", y="total_bill", data=tips)

# Add labels to the plot
plt.xlabel("Day of the Week")
plt.ylabel("Total Bill")
plt.title("Total Bill by Day of the Week")

# Save the plot to a file
plt.savefig("total_bill_by_day.png")
#------------------------------