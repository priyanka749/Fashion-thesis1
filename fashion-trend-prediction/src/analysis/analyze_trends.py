"""
Fashion Trends Visualization - Display all trending clothes
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from collections import defaultdict
import os

def analyze_fashion_trends():
    """Analyze and visualize the fashion trends data"""
    
    # Load the trends data
    with open('data/raw/google_trends.json', 'r') as f:
        data = json.load(f)
    
    print("🎨 Fashion Trends Analysis")
    print("=" * 50)
    
    # Extract all trending items and their categories
    categories = {}
    all_trends = []
    
    for group_name, group_data in data['trend_data'].items():
        category = group_name.replace('group_', '').replace('_', ' ').title()
        keywords = group_data['keywords']
        categories[category] = keywords
        
        # Calculate average trend scores
        trend_data = group_data['data']
        for keyword in keywords:
            scores = []
            for date, values in trend_data.items():
                if keyword in values:
                    scores.append(values[keyword])
            
            avg_score = np.mean(scores) if scores else 0
            all_trends.append({
                'category': category,
                'item': keyword,
                'avg_trend_score': avg_score,
                'max_score': max(scores) if scores else 0,
                'min_score': min(scores) if scores else 0
            })
    
    # Create DataFrame
    df = pd.DataFrame(all_trends)
    
    print(f"Total trending fashion items: {len(df)}")
    print(f"Categories covered: {len(categories)}")
    print()
    
    # Display top trending items
    print("🔥 TOP 15 TRENDING FASHION ITEMS:")
    print("-" * 50)
    top_items = df.nlargest(15, 'avg_trend_score')
    for i, row in top_items.iterrows():
        print(f"{row['avg_trend_score']:.1f} - {row['item']} ({row['category']})")
    
    print()
    print("📊 TRENDING ITEMS BY CATEGORY:")
    print("-" * 50)
    for category, items in categories.items():
        print(f"\n{category}:")
        for item in items:
            trend_score = df[df['item'] == item]['avg_trend_score'].values[0]
            print(f"  • {item} (score: {trend_score:.1f})")
    
    # Create visualizations
    create_trend_visualizations(df, categories)
    
    return df, categories

def create_trend_visualizations(df, categories):
    """Create comprehensive visualizations"""
    
    os.makedirs('outputs', exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # 1. Top trending items chart
    plt.figure(figsize=(14, 8))
    top_20 = df.nlargest(20, 'avg_trend_score')
    
    bars = plt.barh(range(len(top_20)), top_20['avg_trend_score'], 
                    color=sns.color_palette("viridis", len(top_20)))
    
    plt.yticks(range(len(top_20)), top_20['item'])
    plt.xlabel('Average Trend Score')
    plt.title('Top 20 Trending Fashion Items (2024-2025)', fontsize=16, fontweight='bold')
    plt.gca().invert_yaxis()
    
    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, top_20['avg_trend_score'])):
        plt.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2, 
                f'{score:.1f}', va='center', ha='left', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/top_trending_fashion_items.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. Category comparison
    plt.figure(figsize=(14, 8))
    category_avg = df.groupby('category')['avg_trend_score'].mean().sort_values(ascending=False)
    
    bars = plt.bar(range(len(category_avg)), category_avg.values, 
                   color=sns.color_palette("Set2", len(category_avg)))
    
    plt.xticks(range(len(category_avg)), category_avg.index, rotation=45, ha='right')
    plt.ylabel('Average Trend Score')
    plt.title('Fashion Trend Scores by Category', fontsize=16, fontweight='bold')
    
    # Add value labels
    for bar, score in zip(bars, category_avg.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{score:.1f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('outputs/category_trends_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. Heat map of trends by category
    plt.figure(figsize=(16, 10))
    
    # Create pivot table for heatmap
    category_items = []
    trend_scores = []
    item_names = []
    
    for category, items in categories.items():
        for item in items:
            score = df[df['item'] == item]['avg_trend_score'].values[0]
            category_items.append(category)
            trend_scores.append(score)
            item_names.append(item)
    
    # Create heatmap data
    unique_categories = list(categories.keys())
    max_items = max(len(items) for items in categories.values())
    
    heatmap_data = np.zeros((len(unique_categories), max_items))
    heatmap_labels = [[''] * max_items for _ in range(len(unique_categories))]
    
    for i, (category, items) in enumerate(categories.items()):
        for j, item in enumerate(items):
            score = df[df['item'] == item]['avg_trend_score'].values[0]
            heatmap_data[i, j] = score
            heatmap_labels[i][j] = item
    
    # Mask zeros
    mask = heatmap_data == 0
    
    ax = sns.heatmap(heatmap_data, 
                     annot=False,
                     cmap='YlOrRd',
                     mask=mask,
                     cbar_kws={'label': 'Trend Score'},
                     yticklabels=unique_categories)
    
    plt.title('Fashion Trends Heatmap by Category', fontsize=16, fontweight='bold')
    plt.xlabel('Items in Category')
    plt.ylabel('Fashion Categories')
    plt.tight_layout()
    plt.savefig('outputs/fashion_trends_heatmap.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 4. Trend score distribution
    plt.figure(figsize=(12, 6))
    plt.hist(df['avg_trend_score'], bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(df['avg_trend_score'].mean(), color='red', linestyle='--', 
                label=f'Average: {df["avg_trend_score"].mean():.1f}')
    plt.xlabel('Trend Score')
    plt.ylabel('Number of Fashion Items')
    plt.title('Distribution of Fashion Trend Scores', fontsize=16, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('outputs/trend_score_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✅ All visualizations saved to 'outputs/' folder!")
    print("   - top_trending_fashion_items.png")
    print("   - category_trends_comparison.png") 
    print("   - fashion_trends_heatmap.png")
    print("   - trend_score_distribution.png")

def create_trend_report(df):
    """Create a comprehensive text report"""
    
    report = f"""
# Fashion Trends Report - 2024/2025
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary
This report analyzes {len(df)} trending fashion items across {df['category'].nunique()} different categories for women aged 20-25.

## Top Trending Items Overall
"""
    
    top_10 = df.nlargest(10, 'avg_trend_score')
    for i, row in top_10.iterrows():
        report += f"{row['avg_trend_score']:.1f} - **{row['item']}** ({row['category']})\n"
    
    report += "\n## Category Analysis\n"
    
    category_stats = df.groupby('category').agg({
        'avg_trend_score': ['mean', 'max', 'count']
    }).round(1)
    
    for category in df['category'].unique():
        cat_data = df[df['category'] == category]
        top_item = cat_data.loc[cat_data['avg_trend_score'].idxmax()]
        avg_score = cat_data['avg_trend_score'].mean()
        
        report += f"\n### {category}\n"
        report += f"- **Top item**: {top_item['item']} (score: {top_item['avg_trend_score']:.1f})\n"
        report += f"- **Category average**: {avg_score:.1f}\n"
        report += f"- **Items in category**: {len(cat_data)}\n"
    
    report += f"""
## Key Insights
1. **Highest trending category**: {df.groupby('category')['avg_trend_score'].mean().idxmax()}
2. **Most versatile trend**: Items scoring above 90 points
3. **Emerging trends**: {len(df[df['avg_trend_score'] > 85])} items with high trend scores
4. **Fashion diversity**: {df['category'].nunique()} distinct fashion categories tracked

## Recommendations for Fashion Brands
- Focus on top-scoring items for immediate production
- Monitor emerging trends (scores 80-90) for future collections  
- Consider category diversification based on trend distribution
- Target demographic: Women aged 20-25 show strong preference for versatile, mix-and-match pieces
"""
    
    # Save report
    with open('outputs/fashion_trends_report.md', 'w', encoding='utf-8') as f:
        f.write(report)
    
    print("📋 Comprehensive report saved to 'outputs/fashion_trends_report.md'")

def main():
    """Main execution function"""
    print("🚀 Starting Fashion Trends Analysis...")
    print()
    
    df, categories = analyze_fashion_trends()
    create_trend_report(df)
    
    print()
    print("🎉 Analysis Complete!")
    print(f"📊 Analyzed {len(df)} trending fashion items")
    print(f"📈 {len(categories)} fashion categories covered")
    print("🎯 Perfect for women aged 20-25 fashion trend research!")

if __name__ == "__main__":
    main()
