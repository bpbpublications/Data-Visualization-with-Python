# -*- coding: utf-8 -*-
"""
Created on Thu Apr 27 21:37:56 2023

@author: Pooja
CChapter 07 codes as per the sequence inn the chapter
"""
#importing library
import bokeh

#Documentation
help('bokeh.plotting')

#Figure
from bokeh.plotting import figure

p = figure(plot_width=800, plot_height=600, title="My Plot",
       	x_axis_label="X Axis Label", y_axis_label="Y Axis Label",
       	x_range=(0, 10), y_range=(0, 1),
       	x_axis_type="linear", y_axis_type="log",
       	background_fill_color="#f5f5f5", border_fill_color="#cccccc",
       	tools=["pan", "box_zoom", "wheel_zoom", "reset", "save", "hover"],
       	toolbar_location="above")

#Glyphs
import bokeh
from bokeh.plotting import figure, show
from bokeh.sampledata.iris import flowers

# Create a ColumnDataSource from the Iris dataset
source = bokeh.plotting.ColumnDataSource(flowers)

# Create a figure with a title and x/y axis labels
p = figure(title="Iris Dataset", x_axis_label="Petal Length", y_axis_label="Petal Width")

# Add glyphs for each type of glyph to the first five rows of the iris dataset
p.circle(x=flowers["sepal_length"][:5], y=flowers["sepal_width"][:5], size=10, color="blue")
# Show the plot
show(p)
p.square(x=flowers["sepal_length"][:5], y=flowers["sepal_width"][:5], size=10, color="red")
# Show the plot
show(p)
p.triangle(x=flowers["sepal_length"][:5], y=flowers["sepal_width"][:5], size=10, color="green")
# Show the plot
show(p)
p.diamond(x=flowers["sepal_length"][:5], y=flowers["sepal_width"][:5], size=10, color="purple")
# Show the plot
show(p)
p.cross(x=flowers["sepal_length"][:5], y=flowers["sepal_width"][:5], size=10, color="orange")
# Show the plot
show(p)

#----------------------------------
#Scatter plot
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource

#Create sample data
X = [1, 2, 3, 4, 5]
Y = [6, 7, 2, 4, 5]

#Create a ColumnDataSource Object
source = ColumnDataSource(data = dict(x=X,y=Y))

# Create a new plot with a title and axis labels
p = figure(title ="Scatter Plot", x_axis_label='X - Axis', y_axis_label='Y - Axis')

# Add a circle glyph to the plot using the data from ColumnDataSource
p.circle('x','y', size=10, source=source)

#Save th eplot to an HTML file and display it
output_file("scatter.html")
show(p)

#----------------------------------
#example of creating a ColumnDataSource in Bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource
import numpy as np

#Sample Data
x=np.linspace(0,10,100)
y=np.sin(x)

#Create a ColumnDataSource Object
source = ColumnDataSource(data = dict(x=X,y=Y))

#----------------------------------
#Line plot
from bokeh.plotting import figure, output_file, show
###Sample data
X = [1, 2, 3, 4, 5]
Y = [6, 7, 2, 4, 5]

# Create a new plot with a title and axis labels
p = figure(title ="Line Plot", x_axis_label='X - Axis', y_axis_label='Y - Axis')

# Add a line glyph to the plot using the data from ColumnDataSource
p.line(X,Y, line_width=2)

#Save the plot to an HTML file and display it
output_file("line.html")
show(p)

#----------------------------------
#Barplot
from bokeh.plotting import figure, output_file, show
###Sample data
categories = ['Apple', 'Orange', 'Banana']
Values = [5, 3, 4]

# Create a new plot with a title and axis labels
p = figure(title ="Bar Plot", x_axis_label='Category', y_axis_label='Value', 
           x_range=categories)

# Add a vbar glyph to the plot using the data from ColumnDataSource
p.vbar(categories,top=Values,bottom=0, width =0.6)

#Save th eplot to an HTML file and display it
output_file("bar.html")
show(p)

#---------------------------------------------------
#Heat Map
import bokeh
from bokeh.plotting import figure, output_file, show
from bokeh.models import ColumnDataSource, LinearColorMapper
from bokeh.transform import transform
# Create some data
x = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
y = ['Morning', 'Afternoon', 'Evening']
data = {'x': x*len(y), 'y': [i for i in y for _ in x], 
        'values': [5, 3, 2, 7, 6, 4, 8, 1, 9, 2, 5, 7, 6, 2, 3]}
# Create a new plot with a title and axis labels
p = figure(title="Heatmap", x_range=x, y_range=y)
# Create a color mapper to map the values to colors
color_mapper = LinearColorMapper(palette="Viridis256", 
                                 low=min(data['values']), high=max(data['values']))
# Add a Rect glyph to the plot using the data and color mapper
source = ColumnDataSource(data)
p.rect(x='x', y='y', width=1, height=1, source=source, 
       line_color=None, fill_color=transform('values', color_mapper))
# Add a color bar to the plot
color_bar = bokeh.models.ColorBar(color_mapper=color_mapper, location=(0, 0))
p.add_layout(color_bar, 'right')
# Save the plot to an HTML file and display it
output_file("heatmap.html")
show(p)

#---------------------------------------------------
#Histogram
from bokeh.plotting import figure, output_file, show
from bokeh.sampledata.iris import flowers
import numpy as np

# Create a new plot with a title and axis labels
p = figure(title="Histogram", 
           x_axis_label="Sepal Length", y_axis_label="Count")

# Get the data
values, edges = np.histogram(flowers['sepal_length'], bins=20)

# Add the histogram to the plot
p.quad(top=values, bottom=0, left=edges[:-1], right=edges[1:], 
       fill_color="navy", line_color="white", alpha=0.5)

# Save the plot to an HTML file and display it
output_file("histogram.html")
show(p)

#---------------------------------
#Patch plot
from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

# create a sample polygon
x = [1, 2, 3, 4, 5]
y = [1, 3, 4, 2, 1]
polygon = [(xi, yi) for xi, yi in zip(x, y)]

# create a data source for the polygon
source = ColumnDataSource(data=dict(x=x, y=y))

# create a figure and add a patch glyph
p1 = figure(title="Patch plot")
p1.patch(x='x', y='y', fill_alpha=0.4, line_width=2, source=source)

show(p1)


#-------------------------------
#Area Plot
x = [1, 2, 3, 4, 5]
y1 = [2, 4, 3, 6, 4]
y2 = [1, 3, 2, 4, 3]

p2 = figure(title="Area Plot", 
           x_axis_label="X-axis", y_axis_label="Y-axis")
p2.varea(x=x, y1=y1, y2=y2, fill_color="blue")
#with hatch patch as red dot
p2.varea(x=x, y1=y1, y2=y2, fill_color="blue", hatch_color="red", hatch_pattern="dot",fill_alpha=0.3)
#varea_stacked plot
source = ColumnDataSource(data=dict(x=x,y1=y1,y2=y2))
p2.varea_stack(['y1', 'y2'], x='x', 
              fill_color=["lightblue", "lightgreen"], 
              source=source, legend_label=["Area 1", "Area 2"])

show(p2)

#----------------------------
#vbar stacked plot

from bokeh.plotting import figure, show
from bokeh.palettes import Spectral6
x = ["Books", "Magazine", "Posters"]
data = {'Product': x,
        '2020': [20, 30, 40],
        '2021': [10, 15, 20],
        '2022': [5,10,15]}
color=Spectral6[0:3]
# Convert the data to a ColumnDataSource
source = ColumnDataSource(data=data)
p = figure(x_range=x, plot_height=350, title="Product Sales by Year",
           toolbar_location=None, tools="")

p.vbar_stack(['2020','2021','2022'], x='Product', color=color,
             source=source)

show(p)


#---------------------------------------
#interactive scatter plot
from bokeh.plotting import figure, show, output_file
from bokeh.models import HoverTool
import pandas as pd
#Sample data
df = pd.DataFrame({
    'x':[1,2,3,4,5],
    'y':[4,7,1,6,3],
    'size':[10,20,30,40,50],
    'color':['red','green','blue','orange','purple']})
#create Bokeh figure and add scatter plot
p=figure(title='Interactive Scatter Plot',
         tools='box_select,lasso_select,reset')
p.scatter('x','y',size='size',color='color',alpha=0.5, source=df)
#Add hover tooltip
hover = HoverTool(tooltips=[('x', '@x'),('y','@y')])
p.add_tools(hover)
#Show plot in output file or in notebook
output_file('interactive_Scatter.html')
show(p)


#---------------Grid of plots

from bokeh.layouts import gridplot
from bokeh.plotting import figure, show

#create two figures
p1=figure(title="Plot 1",plot_width=250, plot_height=250)
p1.circle([1,2,3],[4,5,6])

p2=figure(title="Plot 2",plot_width=250, plot_height=250)
p2.square([1,2,3],[4,5,6])

#create grid og plots
grid=gridplot([[p1,p2]])

show(grid)

# two plots in row---------------------

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

# create a sample data
x = [1, 2, 3, 4, 5]
y = [1, 3, 4, 2, 1]
y1 = [2, 4, 3, 6, 4]
y2 = [1, 3, 2, 4, 3]

#create polygon with x and y
polygon = [(xi, yi) for xi, yi in zip(x, y)]

# create a First plot as patch glyph
source = ColumnDataSource(data=dict(x=x, y=y))
p1 = figure(title="Patch plot")
p1.patch(x='x', y='y', fill_alpha=0.4, line_width=2, source=source)

# create a Second plot as varea
p2 = figure(title="Area Plot", 
           x_axis_label="X-axis", y_axis_label="Y-axis")
source = ColumnDataSource(data=dict(x=x,y1=y1,y2=y2))
p2.varea_stack(['y1', 'y2'], x='x', 
              fill_color=["lightblue", "lightgreen"], 
              source=source, legend_label=["Area 1", "Area 2"])
#Create Row of Plots
from bokeh.layouts import row
row=row(p1,p2)
show(row)

# two plots in column---------------------
#Create column of Plots
from bokeh.layouts import column
col=column(p1,p2)
show(col)
output_file("Plot_incol.html",col)

#---------------------------------
#PLOTS in TABS

from bokeh.plotting import figure, show
from bokeh.models import ColumnDataSource

# create a sample data
x = [1, 2, 3, 4, 5]
y = [1, 3, 4, 2, 1]
y1 = [2, 4, 3, 6, 4]
y2 = [1, 3, 2, 4, 3]

#create polygon with x and y
polygon = [(xi, yi) for xi, yi in zip(x, y)]

# create a First plot as patch glyph
source = ColumnDataSource(data=dict(x=x, y=y))
p1 = figure(title="Patch plot")
p1.patch(x='x', y='y', fill_alpha=0.4, line_width=2, source=source)

# create a Second plot as varea
p2 = figure(title="Area Plot", 
           x_axis_label="X-axis", y_axis_label="Y-axis")
source = ColumnDataSource(data=dict(x=x,y1=y1,y2=y2))
p2.varea_stack(['y1', 'y2'], x='x', 
              fill_color=["lightblue", "lightgreen"], 
              source=source, legend_label=["Area 1", "Area 2"])

#Create tabs
from bokeh.models import Panel, Tabs
tab1 = Panel(child=p1,title="PATCH PLOT")
tab2=Panel(child=p2,title="VAREA PLOT")

tabs=Tabs(tabs=[tab1,tab2])
output_file("Plots_in_tab.html", tabs)
show(tabs)





