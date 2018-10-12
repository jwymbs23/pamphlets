import altair as alt
import numpy as np
import pickle
import pandas as pd

from vega_datasets import data

top_term_year = pickle.load(open('trending_ratio.pkl', 'rb'))
df = pd.read_pickle('trending_df.p')
for year in top_term_year:
    sub_df = df[year[1:] + ['year']]
    long_df = sub_df.melt('year', var_name='term', value_name='popularity')
    # define selection
    click = alt.selection_single(fields=['term'])
    color = alt.condition(click, alt.Color('term:N', legend=None),
                          alt.value('lightgray'))

    opacity = alt.condition(click, alt.value(1),
                          alt.value(0.5))
    size = alt.condition(click, alt.value(4),
                          alt.value(1))
    # scatter plots of points
    scatter = alt.Chart(long_df).mark_line().encode(
        x='year:Q',
        y='popularity:Q',
        color=color,
        opacity=opacity,
        size=size
    )#.transform_filter(
    #    click
    #).interactive()
    
    # legend
    legend = alt.Chart(long_df).mark_rect().encode(
        y=alt.Y('term:N', axis=alt.Axis(title='Select Origin')),
        color=color
    ).add_selection(
        click
    )
    
    chart = (scatter | legend)
    chart.save(str(year[0]) + '.html')

