import altair as alt
import numpy as np
import pickle
import pandas as pd

#from vega_datasets import data

top_term_year = pickle.load(open('trending_ratio.pkl', 'rb'))
df = pd.read_pickle('trending_df.p')
for year in top_term_year:
    sub_df = df[year[1:] + ['year']]
    long_df = sub_df.melt('year', var_name='term', value_name='Term Popularity')
    # define selection
    click = alt.selection_single(fields=['term'], on='click')

    color = alt.condition(click, alt.Color('term:N', legend=None),
                          alt.value('lightgray'))

    opacity = alt.condition(click, alt.value(1),
                          alt.value(0.5))
    size = alt.condition(click, alt.value(4),
                          alt.value(1))
    # scatter plots of points
    scatter = alt.Chart(long_df, height=400, width=600).mark_line().encode(
        x='year:O',
        y='Term Popularity:Q',
        color=color,
        opacity=opacity,
        size=size
    )#.transform_filter(
    #    click
    #).interactive()

    scatter.encoding.x.title = 'Year'
    
    # legend
    legend = alt.Chart(long_df).mark_rect().encode(
        y=alt.Y('term:O', axis=alt.Axis(title='Select a Term:'), sort=[]),#, sort=list(range(len(year)-1))),
        color=color,
    ).add_selection(
        click
    )

    
    
    chart = (legend | scatter)
    chart.save('trending_'+str(year[0]) + '.json')

