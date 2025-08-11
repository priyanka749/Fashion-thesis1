import pandas as pd

# Load regional dataset
df2 = pd.read_csv('data/raw/regional_fashion_trends.csv')

print(' REGIONAL FASHION DATASET SUCCESS!')
print('='*50)
print(f' Total Data Points: {len(df2):,}')
print(f' Regions Covered: {df2["region"].nunique()}')
print(f' Keywords: {df2["keyword"].nunique()}')

print('\n REGIONS ANALYZED:')
for region in sorted(df2["region"].unique()):
    print(f'• {region}')

print('\n TOP TRENDING ITEMS BY REGION:')
for region in ['US', 'UK', 'France', 'Japan', 'Germany']:
    region_data = df2[df2["region"] == region].groupby('keyword')["trend_score"].mean().sort_values(ascending=False).head(3)
    print(f'\n{region}:')
    for item, score in region_data.items():
        print(f'  {item.title()}: {score:.1f}')

