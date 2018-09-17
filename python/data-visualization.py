import numpy as np  # useful for many scientific computing in Python
import pandas as pd # primary data structure library
from PIL import Image # converting images into arrays
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from wordcloud import WordCloud, STOPWORDS
import seaborn as sns
import folium as folium

#
# Area Plots
#
df = pd.read_csv('titanic.csv')
df.dropna(subset=['PClass'], inplace=True)
df.dropna(subset=['Survived'], inplace=True)
df.dropna(subset=['Age'], inplace=True)
for i in df.index:
    if df.at[i, 'PClass'] == "1st":
        df.at[i, 'PClass']  = 1
    elif df.at[i, 'PClass']  == "2nd":
        df.at[i, 'PClass']  = 2
    elif df.at[i, 'PClass']  == "3rd":
        df.at[i, 'PClass']  = 3

def Show_Area_Plot_Figure():
    df = df[['PClass', 'Survived']]
    df_area = df.groupby(['PClass']).sum()
    df_area = df_area.head(20)
    df_area = df_area['Survived'].transpose()
    df_area.plot(kind='area', 
            stacked=False,
            figsize=(20, 10),
            y = 'Survived'
        )
    plt.title('PClass ~ Survived')
    plt.ylabel('Survived Volumn')
    plt.xlabel('PClass')
    plt.show()

#
# Histogram
#
def Show_Histogram_Figure():
    df_hist = df[['PClass', 'Survived']]
    df_hist = df_hist.groupby(['PClass']).sum()
    df_hist = df_hist['Survived']
    # count, bin_edges = np.histogram(df_hist['Survived'], 3)
    df_hist.plot(kind='hist', 
            stacked=False,
            figsize=(20, 10)
        )
    plt.title('PClass ~ Survived')
    plt.ylabel('Survived Volumn')
    plt.xlabel('PClass')
    plt.show()

#
# Bar Chart
#
def Show_Bar_Chart_Figure():
    df_bar = df[['PClass', 'Survived']]
    df_bar = df_bar.groupby(['PClass']).sum()
    df_bar = df_bar['Survived']
    df_bar.plot(kind='bar')
    plt.annotate('',
         xy=(2, 140),
         xytext=(1, 120),
         xycoords='data',
         arrowprops=dict(arrowstyle='->', connectionstyle='arc3', color='blue', lw=2)
        )

    plt.title('PClass ~ Survived')
    plt.ylabel('Survived Volumn')
    plt.xlabel('PClass')
    plt.show()

#
# Pie Chart
#
def Show_Pie_Chart_Figure():
    df_pie = df[['PClass', 'Survived']]
    df_pie = df_pie.groupby(['PClass']).sum()
    df_pie = df_pie['Survived']
    df_pie.plot(kind='pie',
            figsize=(5, 5),
            autopct='%1.1f%%',
            startangle=90,
            shadow=True,   
        )
    plt.title('PClass ~ Survived')
    plt.show()

#
# Box Plots
#
def Show_Box_Plot_Figure():
    df_box = df[['Age', 'Survived']]
    df_box = df_box.loc[df['Survived'].isin([1])]
    df_box = df_box[['Age']]
    df_box.plot(kind='box', figsize=(8,6))
    plt.title('Age ~ Survived')
    plt.ylabel('Age')
    plt.show()

#
# Scatter Plot
#
def Show_Scatter_Plot_Figure():
    df_scatter = df[['Age', 'Survived', 'PClass']]
    df_scatter = df_scatter.loc[df['Survived'].isin([1])]
    df_scatter = df[['Age', 'PClass']].astype('float64')
    df_scatter.plot(kind='scatter', x='Age', y='PClass', figsize=(10,6))
    plt.title('Age ~ PClass ~ Survived')
    plt.xlabel('Age')
    plt.ylabel('PClass')
    plt.show()

#
# Waffle Charts
#
def Waffle_Chart(categories, values, height, width, colormap, value_sign=''):

    # compute the proportion of each category with respect to the total
    total_values = sum(values)
    category_proportions = [(float(value) / total_values) for value in values]

    # compute the total number of tiles
    total_num_tiles = width * height # total number of tiles
    print ('Total number of tiles is', total_num_tiles)
    
    # compute the number of tiles for each catagory
    tiles_per_category = [round(proportion * total_num_tiles) for proportion in category_proportions]
    
    # initialize the waffle chart as an empty matrix
    waffle_chart = np.zeros((height, width))

    # define indices to loop through waffle chart
    category_index = 0
    tile_index = 0

    # populate the waffle chart
    for col in range(width):
        for row in range(height):
            tile_index += 1

            # if the number of tiles populated for the current category 
            # is equal to its corresponding allocated tiles...
            if tile_index > sum(tiles_per_category[0:category_index]):
                # ...proceed to the next category
                category_index += 1       
            
            # set the class value to an integer, which increases with class
            waffle_chart[row, col] = category_index
    
    # instantiate a new figure object
    fig = plt.figure()

    # use matshow to display the waffle chart
    colormap = plt.cm.coolwarm
    plt.matshow(waffle_chart, cmap=colormap)
    plt.colorbar()

    # get the axis
    ax = plt.gca()

    # set minor ticks
    ax.set_xticks(np.arange(-.5, (width), 1), minor=True)
    ax.set_yticks(np.arange(-.5, (height), 1), minor=True)
    
    # add dridlines based on minor ticks
    ax.grid(which='minor', color='w', linestyle='-', linewidth=2)

    plt.xticks([])
    plt.yticks([])

    # compute cumulative sum of individual categories to match color schemes between chart and legend
    values_cumsum = np.cumsum(values)
    total_values = values_cumsum[len(values_cumsum) - 1]

    # create legend
    legend_handles = []
    for i, category in enumerate(categories):
        if value_sign == '%':
            label_str = category + ' (' + str(values[i]) + value_sign + ')'
        else:
            label_str = category + ' (' + value_sign + str(values[i]) + ')'
            
        color_val = colormap(float(values_cumsum[i])/total_values)
        legend_handles.append(mpatches.Patch(color=color_val, label=label_str))

    # add legend to chart
    plt.legend(
        handles=legend_handles,
        loc='lower center', 
        ncol=len(categories),
        bbox_to_anchor=(0., -0.2, 0.95, .1)
    )

    plt.show()

width = 40
height = 10

categories = ["TestA", "TestB", "TestC"]
values = [50, 30, 20]

colormap = plt.cm.coolwarm
#Waffle_Chart(categories, values, height, width, colormap)

#
# Word Cloud
#
def Word_Cloud():
    novel = open('chill.txt', 'r').read()
    stopwords = set(STOPWORDS)
    chill_wc = WordCloud(background_color='white', max_words=2000, stopwords=stopwords)
    chill_wc.generate(novel)
    plt.imshow(chill_wc, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# Word_Cloud()

#
# Folium
#
canada_map = folium.Map(
    location=[40.4637, -3.7492],
    zoom_start=6,
    tiles='Stamen Terrain'
)

ontario = folium.map.FeatureGroup()
ontario.add_child(
    folium.CircleMarker(
        [51.25, -85.32], radius = 5,
        color = "red", fill_color = "Red"
    )
)

canada_map.add_child(ontario)

folium.Marker([51.25, -85.32], popup='Ontario').add_to(canada_map)

#canada_map.save("Map.html")