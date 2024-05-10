import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')
df['BMI'] = df['weight'] / (df['height'] / 100) ** 2

# 2
df['overweight'] = df['BMI'].apply(lambda x: 1 if x > 25 else 0)
df.drop(columns='BMI', inplace=True)

# 3
df['cholesterol'] = df['cholesterol'].apply(lambda x: 0 if x == 1 else x)
df['cholesterol'] = df['cholesterol'].apply(lambda x: 1 if x > 1 else x)

df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else x)
df['gluc'] = df['gluc'].apply(lambda x: 1 if x > 1 else x)


# 4
def draw_cat_plot():
    # 5
    df_cat = df.melt(id_vars=['cardio'], value_vars=['cholesterol', 'gluc', 'smoke', 'alco', 'active', 'overweight'])

    # 6
    df_cat = df_cat.groupby('cardio').value_counts().reset_index()
    df_cat.rename(columns={'count': 'total'}, inplace=True)

    # 7

    # 8

    fig = sns.catplot(data=df_cat, kind='bar', x='variable', y='total', col='cardio',
                      hue='value', order=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']).fig

    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[df['height'] >= df['height'].quantile(0.025)]
    df_heat = df_heat[df_heat['height'] <= df['height'].quantile(0.975)]
    df_heat = df_heat[df_heat['weight'] <= df['weight'].quantile(0.975)]
    df_heat = df_heat[df_heat['weight'] >= df['weight'].quantile(0.025)]
    df_heat = df_heat[df_heat['ap_lo'] <= df_heat['ap_hi']]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots()
    ax = sns.heatmap(corr, mask=mask, annot=True, fmt='1.1f')
    # 15

    # 16
    fig.savefig('heatmap.png')
    return fig
