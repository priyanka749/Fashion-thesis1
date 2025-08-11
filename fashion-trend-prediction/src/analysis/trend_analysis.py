import json
from datetime import datetime

def analyze_fashion_trends():
    with open('data/raw/google_trends.json', 'r') as f:
        data = json.load(f)

    print(' FASHION TREND ANALYSIS REPORT')
    print('='*60)
    print(f'Analysis Date: {datetime.now().strftime("%Y-%m-%d %H:%M")}')
    print(f'Target Demographic: Women aged 20-25')
    print()

    all_trends = []

    for group_name, group_data in data['trend_data'].items():
        category = group_name.replace('group_', '').replace('_', ' ').title()
        print(f' {category}')
        print('-' * 40)
        
    
        avg_trends = {}
        for keyword in group_data['keywords']:
            total = sum(week_data[keyword] for week_data in group_data['data'].values())
            avg_trends[keyword] = total / len(group_data['data'])
        
        # Sort by popularity
        sorted_trends = sorted(avg_trends.items(), key=lambda x: x[1], reverse=True)
        
        for i, (item, score) in enumerate(sorted_trends, 1):
            status = '' if score >= 90 else '' if score >= 80 else ''
            print(f'  {i}. {item.title()}: {score:.1f} {status}')
            all_trends.append((item, score, category))
        print()

    print(' TOP 10 OVERALL TRENDING ITEMS:')
    print('-' * 40)
    top_trends = sorted(all_trends, key=lambda x: x[1], reverse=True)[:10]
    
    for i, (item, score, category) in enumerate(top_trends, 1):
        status = '' if score >= 95 else '' if score >= 90 else ''
        print(f'{i:2d}. {item.title()} ({category}): {score:.1f} {status}')

if __name__ == "__main__":
    analyze_fashion_trends()
