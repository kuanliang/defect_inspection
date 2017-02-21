import numpy as np
import matplotlib.pyplot as plt
# from tsne import bh_sne

from bokeh.plotting import figure, output_file, show, ColumnDataSource
from bokeh.models import HoverTool



def _create_column_source(vis_data, labels, sns, images):
    '''
    '''
    sourceDict = {}
    for label in list(set(labels)):
        # print label
        sourceDict[label] = ColumnDataSource(
                            data=dict(
                                x=vis_data[labels==label][:, 0],
                                y=vis_data[labels==label][:, 1],
                                desc=sns[labels==label],
                                imgs=images[labels==label]
                            ))
    return sourceDict
    
    

def plot_tsne(features, labels, sns, perplexity, interactive=True, **kwargs):
    '''plot 2-dimensional t-sne plot 
    
    Notes:
    
    Args:
        features:
        labels:
        images:
        perplexity:
        ColumnDataSource:
    
    Return:
        None
    
    '''
    vis_data = bh_sne(features, perplexity=perplexity)
    vis_x = vis_data[:, 0]
    vis_y = vis_data[:, 1]
    plt.style.use('ggplot')
    
    if not interactive:
        # plot with matplotlib
        plt.scatter(vis_x, vis_y, c=labels, cmap=plt.cm.get_cmap('jet', 10))
        plt.colorbar(ticks=range(10))
        plt.clim(-0.5, 9.5)
        plt.show()
    else:
        # plot with BOKEH
        
        sourceDict = _create_column_source(vis_data, labels, sns, kwargs['images'])
        
        colorList = ['red', 'blue', 'yellow', 'blue']
    
        hover = HoverTool(
            tooltips=
            """
                <div>
                    <div>
                        <img
                            src="@imgs" height="80" alt="@imgs" width="80"
                            style="float: left; margin: 0px 15px 15px 0px;"
                            border="2"
                        ></img>
                    </div>
                    <div>
                        <span style="font-size: 17px; font-weight: bold;">@desc</span>
                    </div>
                </div>
            """
        )

        p = figure(plot_width=800, plot_height=800, tools=[hover],
                   title="Mouse over the dots")

        label_remove0 = list(set(labels))
        label_remove0.remove('0')
    
        #print label_remove0
    
        for key, value in sourceDict.iteritems():
            # print key
            if key == '0':
                p.circle('x', 'y', size=20, alpha=0.5, color='green', source=sourceDict[key])
            else:
                p.circle('x', 'y', size=20, alpha=0.5, 
                        color=colorList[label_remove0.index(key)],
                      source=sourceDict[key])
    
        show(p)