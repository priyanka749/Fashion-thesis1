"""
Fashion Trend Analysis and Visualization
Analyze fashion trends and create visualizations
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from wordcloud import WordCloud
from collections import Counter
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any

class FashionTrendAnalyzer:
    def __init__(self):
        """Initialize the fashion trend analyzer"""
        self.output_dir = "outputs"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Set plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """
        Load all processed data files
        
        Returns:
            Dictionary of DataFrames by source
        """
        data_files = {
            'pinterest': "data/processed/pinterest_processed.csv",
            'blogs': "data/processed/fashion_blogs_processed.csv",
            'trends': "data/processed/google_trends_processed.csv"
        }
        
        datasets = {}
        for source, file_path in data_files.items():
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                datasets[source] = df
                print(f"Loaded {len(df)} records from {source}")
            else:
                print(f"File not found: {file_path}")
        
        return datasets
    
    def analyze_keyword_trends(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze fashion keyword trends across all data sources
        
        Args:
            datasets: Dictionary of DataFrames by source
            
        Returns:
            Analysis results
        """
        all_keywords = []
        
        # Extract keywords from all sources
        for source, df in datasets.items():
            if 'fashion_keywords' in df.columns:
                for keywords in df['fashion_keywords'].dropna():
                    if isinstance(keywords, str):
                        # Handle string representation of lists
                        keywords = eval(keywords) if keywords.startswith('[') else [keywords]
                    if isinstance(keywords, list):
                        all_keywords.extend(keywords)
        
        # Count keyword frequencies
        keyword_counts = Counter(all_keywords)
        
        # Top keywords
        top_keywords = dict(keyword_counts.most_common(20))
        
        # Create visualizations
        self.create_keyword_visualizations(keyword_counts)
        
        return {
            'total_keywords': len(all_keywords),
            'unique_keywords': len(keyword_counts),
            'top_keywords': top_keywords,
            'keyword_counts': keyword_counts
        }
    
    def analyze_temporal_trends(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze trends over time
        
        Args:
            datasets: Dictionary of DataFrames by source
            
        Returns:
            Temporal analysis results
        """
        temporal_data = []
        
        # Process Google Trends data
        if 'trends' in datasets:
            trends_df = datasets['trends']
            if 'date' in trends_df.columns:
                trends_df['date'] = pd.to_datetime(trends_df['date'])
                temporal_data.append(trends_df[['date', 'keyword', 'trend_value']])
        
        # Process Pinterest data
        if 'pinterest' in datasets:
            pinterest_df = datasets['pinterest']
            if 'created_at' in pinterest_df.columns:
                pinterest_df['date'] = pd.to_datetime(pinterest_df['created_at'])
                # Create trend metrics for Pinterest
                pinterest_temporal = pinterest_df.groupby('date').size().reset_index(name='pinterest_posts')
                temporal_data.append(pinterest_temporal)
        
        # Create temporal visualizations
        if temporal_data:
            self.create_temporal_visualizations(temporal_data)
        
        return {'temporal_data': temporal_data}
    
    def analyze_color_trends(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Analyze color trends in fashion
        
        Args:
            datasets: Dictionary of DataFrames by source
            
        Returns:
            Color analysis results
        """
        color_keywords = [
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple',
            'orange', 'brown', 'gray', 'grey', 'navy', 'beige', 'nude', 'gold',
            'silver', 'rose', 'coral', 'mint', 'lavender', 'turquoise'
        ]
        
        color_counts = Counter()
        
        # Extract color mentions from all sources
        for source, df in datasets.items():
            text_columns = ['description_clean', 'full_text', 'keywords_text']
            
            for col in text_columns:
                if col in df.columns:
                    for text in df[col].dropna():
                        if isinstance(text, str):
                            text_lower = text.lower()
                            for color in color_keywords:
                                if color in text_lower:
                                    color_counts[color] += text_lower.count(color)
        
        # Create color trend visualization
        self.create_color_visualizations(color_counts)
        
        return {'color_trends': dict(color_counts.most_common(15))}
    
    def create_keyword_visualizations(self, keyword_counts: Counter):
        """
        Create keyword trend visualizations
        
        Args:
            keyword_counts: Counter object with keyword frequencies
        """
        # Word Cloud
        wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='viridis',
            max_words=100
        ).generate_from_frequencies(keyword_counts)
        
        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title('Fashion Keywords Word Cloud', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/fashion_wordcloud.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Top keywords bar chart
        top_keywords = dict(keyword_counts.most_common(20))
        
        plt.figure(figsize=(12, 8))
        keywords = list(top_keywords.keys())
        counts = list(top_keywords.values())
        
        bars = plt.barh(keywords, counts, color='skyblue', alpha=0.8)
        plt.xlabel('Frequency')
        plt.title('Top 20 Fashion Keywords', fontsize=16, fontweight='bold')
        plt.gca().invert_yaxis()
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                    str(count), va='center', ha='left')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/top_keywords.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_temporal_visualizations(self, temporal_data: List[pd.DataFrame]):
        """
        Create temporal trend visualizations
        
        Args:
            temporal_data: List of DataFrames with temporal data
        """
        # Google Trends over time
        if len(temporal_data) > 0 and 'trend_value' in temporal_data[0].columns:
            trends_df = temporal_data[0]
            
            # Top trending keywords
            top_keywords = trends_df.groupby('keyword')['trend_value'].mean().nlargest(10).index
            
            plt.figure(figsize=(15, 8))
            for keyword in top_keywords:
                keyword_data = trends_df[trends_df['keyword'] == keyword]
                plt.plot(keyword_data['date'], keyword_data['trend_value'], 
                        marker='o', label=keyword, linewidth=2)
            
            plt.xlabel('Date')
            plt.ylabel('Trend Value')
            plt.title('Fashion Keyword Trends Over Time', fontsize=16, fontweight='bold')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f'{self.output_dir}/temporal_trends.png', dpi=300, bbox_inches='tight')
            plt.show()
        
        # Interactive Plotly visualization
        if len(temporal_data) > 0:
            self.create_interactive_trends(temporal_data[0])
    
    def create_interactive_trends(self, trends_df: pd.DataFrame):
        """
        Create interactive trend visualization with Plotly
        
        Args:
            trends_df: DataFrame with trend data
        """
        if 'trend_value' not in trends_df.columns:
            return
        
        # Get top keywords
        top_keywords = trends_df.groupby('keyword')['trend_value'].mean().nlargest(8).index
        
        fig = go.Figure()
        
        for keyword in top_keywords:
            keyword_data = trends_df[trends_df['keyword'] == keyword]
            fig.add_trace(go.Scatter(
                x=keyword_data['date'],
                y=keyword_data['trend_value'],
                mode='lines+markers',
                name=keyword,
                line=dict(width=3),
                marker=dict(size=6)
            ))
        
        fig.update_layout(
            title='Interactive Fashion Keyword Trends',
            xaxis_title='Date',
            yaxis_title='Trend Value',
            hovermode='x unified',
            width=1000,
            height=600
        )
        
        fig.write_html(f'{self.output_dir}/interactive_trends.html')
        print(f"Interactive trends chart saved to {self.output_dir}/interactive_trends.html")
    
    def create_color_visualizations(self, color_counts: Counter):
        """
        Create color trend visualizations
        
        Args:
            color_counts: Counter object with color frequencies
        """
        if not color_counts:
            return
        
        top_colors = dict(color_counts.most_common(15))
        
        # Color palette mapping
        color_map = {
            'black': '#000000', 'white': '#FFFFFF', 'red': '#FF0000',
            'blue': '#0000FF', 'green': '#00FF00', 'yellow': '#FFFF00',
            'pink': '#FFC0CB', 'purple': '#800080', 'orange': '#FFA500',
            'brown': '#A52A2A', 'gray': '#808080', 'grey': '#808080',
            'navy': '#000080', 'beige': '#F5F5DC', 'nude': '#F0D0A0',
            'gold': '#FFD700', 'silver': '#C0C0C0', 'coral': '#FF7F50'
        }
        
        # Create color trend chart
        colors = list(top_colors.keys())
        counts = list(top_colors.values())
        bar_colors = [color_map.get(color, '#1f77b4') for color in colors]
        
        plt.figure(figsize=(12, 8))
        bars = plt.bar(colors, counts, color=bar_colors, alpha=0.8, edgecolor='black')
        
        plt.xlabel('Colors')
        plt.ylabel('Frequency')
        plt.title('Fashion Color Trends', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                    str(count), ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/color_trends.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_dashboard(self, datasets: Dict[str, pd.DataFrame]):
        """
        Create a comprehensive dashboard with multiple visualizations
        
        Args:
            datasets: Dictionary of DataFrames by source
        """
        # Analyze all aspects
        keyword_analysis = self.analyze_keyword_trends(datasets)
        temporal_analysis = self.analyze_temporal_trends(datasets)
        color_analysis = self.analyze_color_trends(datasets)
        
        # Create summary dashboard
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Data Sources Overview', 'Top Keywords', 
                          'Color Trends', 'Trend Summary'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "table"}]]
        )
        
        # Data sources pie chart
        source_counts = {source: len(df) for source, df in datasets.items()}
        fig.add_trace(
            go.Pie(labels=list(source_counts.keys()), 
                  values=list(source_counts.values()),
                  name="Data Sources"),
            row=1, col=1
        )
        
        # Top keywords
        top_kw = keyword_analysis['top_keywords']
        fig.add_trace(
            go.Bar(x=list(top_kw.values())[:10], 
                  y=list(top_kw.keys())[:10],
                  orientation='h',
                  name="Keywords"),
            row=1, col=2
        )
        
        # Color trends
        color_trends = color_analysis['color_trends']
        fig.add_trace(
            go.Bar(x=list(color_trends.keys())[:10],
                  y=list(color_trends.values())[:10],
                  name="Colors"),
            row=2, col=1
        )
        
        # Summary table
        summary_data = [
            ['Total Records', sum(len(df) for df in datasets.values())],
            ['Unique Keywords', keyword_analysis['unique_keywords']],
            ['Top Color', list(color_analysis['color_trends'].keys())[0] if color_analysis['color_trends'] else 'N/A'],
            ['Data Sources', len(datasets)]
        ]
        
        fig.add_trace(
            go.Table(
                header=dict(values=['Metric', 'Value']),
                cells=dict(values=list(zip(*summary_data)))
            ),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, 
                         title_text="Fashion Trend Analysis Dashboard")
        
        fig.write_html(f'{self.output_dir}/fashion_dashboard.html')
        print(f"Comprehensive dashboard saved to {self.output_dir}/fashion_dashboard.html")
    
    def generate_report(self, datasets: Dict[str, pd.DataFrame]) -> str:
        """
        Generate a comprehensive analysis report
        
        Args:
            datasets: Dictionary of DataFrames by source
            
        Returns:
            Report content as string
        """
        # Perform all analyses
        keyword_analysis = self.analyze_keyword_trends(datasets)
        temporal_analysis = self.analyze_temporal_trends(datasets)
        color_analysis = self.analyze_color_trends(datasets)
        
        # Generate report
        report = f"""
# Fashion Trend Analysis Report
Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Data Overview
- Total data sources: {len(datasets)}
- Total records: {sum(len(df) for df in datasets.values())}

## Source Breakdown
"""
        
        for source, df in datasets.items():
            report += f"- {source.title()}: {len(df)} records\n"
        
        report += f"""

## Keyword Analysis
- Total keywords found: {keyword_analysis['total_keywords']}
- Unique keywords: {keyword_analysis['unique_keywords']}

### Top 10 Fashion Keywords:
"""
        
        for i, (keyword, count) in enumerate(list(keyword_analysis['top_keywords'].items())[:10], 1):
            report += f"{i}. {keyword}: {count} mentions\n"
        
        report += f"""

## Color Trends
Top fashion colors mentioned:
"""
        
        for i, (color, count) in enumerate(list(color_analysis['color_trends'].items())[:10], 1):
            report += f"{i}. {color}: {count} mentions\n"
        
        report += f"""

## Key Insights
1. Most popular fashion keyword: {list(keyword_analysis['top_keywords'].keys())[0]}
2. Most trending color: {list(color_analysis['color_trends'].keys())[0] if color_analysis['color_trends'] else 'N/A'}
3. Data richness: {keyword_analysis['unique_keywords']} unique fashion terms identified
"""
        
        # Save report
        with open(f'{self.output_dir}/fashion_trend_report.md', 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Analysis report saved to {self.output_dir}/fashion_trend_report.md")
        return report

def main():
    """
    Main function to run fashion trend analysis
    """
    analyzer = FashionTrendAnalyzer()
    
    # Load data
    datasets = analyzer.load_data()
    
    if not datasets:
        print("No data found. Please run data collection and preprocessing first.")
        return
    
    print("Starting comprehensive fashion trend analysis...")
    
    # Create all visualizations and analyses
    analyzer.create_comprehensive_dashboard(datasets)
    
    # Generate report
    report = analyzer.generate_report(datasets)
    
    print("\nAnalysis complete! Check the 'outputs' folder for:")
    print("- fashion_wordcloud.png")
    print("- top_keywords.png") 
    print("- temporal_trends.png")
    print("- color_trends.png")
    print("- interactive_trends.html")
    print("- fashion_dashboard.html")
    print("- fashion_trend_report.md")

if __name__ == "__main__":
    main()
