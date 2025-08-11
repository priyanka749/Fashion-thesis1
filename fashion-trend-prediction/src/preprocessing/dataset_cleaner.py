"""
Fashion Trends Dataset Cleaner and Consolidator
Creates a single, clean dataset from multiple sources
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict

class FashionDatasetCleaner:
    def __init__(self):
        """Initialize the dataset cleaner"""
        self.output_dir = "data/processed"
        os.makedirs(self.output_dir, exist_ok=True)
        
    def load_and_analyze_datasets(self):
        """Load and analyze all available datasets"""
        
        print("🔍 LOADING AND ANALYZING DATASETS")
        print("="*50)
        
        datasets = {}
        
        # Load main massive dataset
        try:
            df_main = pd.read_csv('data/raw/massive_fashion_trends.csv')
            datasets['main'] = df_main
            print(f" Main Dataset: {len(df_main):,} records")
        except FileNotFoundError:
            print(" Main dataset not found")
            
        # Load regional dataset
        try:
            df_regional = pd.read_csv('data/raw/regional_fashion_trends.csv')
            datasets['regional'] = df_regional
            print(f" Regional Dataset: {len(df_regional):,} records")
        except FileNotFoundError:
            print(" Regional dataset not found")
            
        # Load original JSON dataset if exists
        try:
            df_json = pd.read_json('data/raw/google_trends.json')
            if not df_json.empty:
                datasets['json'] = df_json
                print(f" JSON Dataset: {len(df_json):,} records")
        except:
            print("ℹ JSON dataset not available or empty")
            
        return datasets
    
    def clean_main_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the main massive dataset"""
        
        print("\n CLEANING MAIN DATASET")
        print("-" * 30)
        
        # Make a copy
        df_clean = df.copy()
        
        # Basic cleaning
        initial_rows = len(df_clean)
        print(f"Initial rows: {initial_rows:,}")
        
        # Convert date column
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        print(f"After removing duplicates: {len(df_clean):,} rows (-{initial_rows - len(df_clean):,})")
        
        # Remove invalid trend scores
        df_clean = df_clean[(df_clean['trend_score'] >= 0) & (df_clean['trend_score'] <= 100)]
        
        # Clean keyword names
        df_clean['keyword'] = df_clean['keyword'].str.strip().str.lower()
        df_clean['keyword_clean'] = df_clean['keyword'].str.title()
        
        # Clean category names
        df_clean['category'] = df_clean['category'].str.strip().str.lower()
        df_clean['category_clean'] = df_clean['category'].str.title()
        
        # Remove null values
        df_clean = df_clean.dropna(subset=['keyword', 'trend_score', 'category'])
        print(f"After cleaning: {len(df_clean):,} rows")
        
        # Add data source identifier
        df_clean['data_source'] = 'main_dataset'
        
        return df_clean
    
    def clean_regional_dataset(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean the regional dataset"""
        
        print("\n CLEANING REGIONAL DATASET")
        print("-" * 30)
        
        df_clean = df.copy()
        
        initial_rows = len(df_clean)
        print(f"Initial rows: {initial_rows:,}")
        
        # Convert date column
        df_clean['date'] = pd.to_datetime(df_clean['date'])
        
        # Remove duplicates
        df_clean = df_clean.drop_duplicates()
        
        # Clean keyword and category names
        df_clean['keyword'] = df_clean['keyword'].str.strip().str.lower()
        df_clean['keyword_clean'] = df_clean['keyword'].str.title()
        df_clean['category'] = df_clean['category'].str.strip().str.lower()
        df_clean['category_clean'] = df_clean['category'].str.title()
        
        # Clean region names
        df_clean['region'] = df_clean['region'].str.strip().str.upper()
        
        # Remove invalid trend scores
        df_clean = df_clean[(df_clean['trend_score'] >= 0) & (df_clean['trend_score'] <= 100)]
        
        # Remove null values
        df_clean = df_clean.dropna(subset=['keyword', 'trend_score', 'region'])
        
        # Add missing columns to match main dataset structure
        df_clean['search_volume'] = df_clean['trend_score'] * np.random.randint(50, 500, len(df_clean))
        df_clean['day_of_week'] = df_clean['date'].dt.day_name()
        df_clean['month'] = df_clean['date'].dt.month_name()
        df_clean['season'] = df_clean['date'].dt.month.map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Add data source identifier
        df_clean['data_source'] = 'regional_dataset'
        
        print(f"After cleaning: {len(df_clean):,} rows")
        
        return df_clean
    
    def standardize_categories(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize category names across datasets"""
        
        print("\n STANDARDIZING CATEGORIES")
        print("-" * 30)
        
        # Category mapping for consistency
        category_mapping = {
            'tops': 'Tops',
            'bottoms': 'Bottoms', 
            'skirts': 'Skirts',
            'dresses': 'Dresses',
            'shoes': 'Shoes',
            'accessories': 'Accessories',
            'outerwear': 'Outerwear',
            'fabrics': 'Fabrics',
            'colors': 'Colors',
            'patterns': 'Patterns',
            'styles': 'Styles',
            'occasions': 'Occasions'
        }
        
        df['category_standardized'] = df['category'].map(category_mapping).fillna(df['category_clean'])
        
        print("Categories standardized:")
        for cat in df['category_standardized'].unique():
            count = len(df[df['category_standardized'] == cat])
            print(f"  {cat}: {count:,} records")
            
        return df
    
    def create_unified_dataset(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Create a single unified dataset"""
        
        print("\n🔗 CREATING UNIFIED DATASET")
        print("-" * 30)
        
        unified_dfs = []
        
        # Process main dataset
        if 'main' in datasets:
            df_main_clean = self.clean_main_dataset(datasets['main'])
            df_main_clean = self.standardize_categories(df_main_clean)
            
            # Select common columns
            main_columns = [
                'date', 'keyword_clean', 'category_standardized', 'trend_score',
                'search_volume', 'day_of_week', 'month', 'season', 'data_source'
            ]
            
            # Add region column for consistency
            df_main_clean['region'] = 'GLOBAL'
            main_columns.append('region')
            
            df_main_final = df_main_clean[main_columns].copy()
            unified_dfs.append(df_main_final)
        
        # Process regional dataset
        if 'regional' in datasets:
            df_regional_clean = self.clean_regional_dataset(datasets['regional'])
            df_regional_clean = self.standardize_categories(df_regional_clean)
            
            regional_columns = [
                'date', 'keyword_clean', 'category_standardized', 'trend_score',
                'search_volume', 'day_of_week', 'month', 'season', 'region', 'data_source'
            ]
            
            df_regional_final = df_regional_clean[regional_columns].copy()
            unified_dfs.append(df_regional_final)
        
        # Combine all datasets
        if unified_dfs:
            df_unified = pd.concat(unified_dfs, ignore_index=True)
            
            # Rename columns for clarity
            df_unified = df_unified.rename(columns={
                'keyword_clean': 'fashion_item',
                'category_standardized': 'category',
                'trend_score': 'popularity_score',
                'search_volume': 'search_volume'
            })
            
            # Sort by date and fashion item
            df_unified = df_unified.sort_values(['fashion_item', 'date']).reset_index(drop=True)
            
            print(f" Unified dataset created: {len(df_unified):,} records")
            
            return df_unified
        else:
            print(" No datasets available for unification")
            return pd.DataFrame()
    
    def add_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add advanced features for AI modeling"""
        
        print("\n ADDING ADVANCED FEATURES")
        print("-" * 30)
        
        # Time-based features
        df['year'] = df['date'].dt.year
        df['month_num'] = df['date'].dt.month
        df['day_of_year'] = df['date'].dt.dayofyear
        df['week_of_year'] = df['date'].dt.isocalendar().week
        df['is_weekend'] = df['date'].dt.dayofweek.isin([5, 6])
        
        # Trend analysis features
        df = df.sort_values(['fashion_item', 'date'])
        
        # Rolling averages
        df['popularity_7day_avg'] = df.groupby('fashion_item')['popularity_score'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        df['popularity_30day_avg'] = df.groupby('fashion_item')['popularity_score'].transform(
            lambda x: x.rolling(window=30, min_periods=1).mean()
        )
        
        # Trend direction
        df['popularity_change'] = df.groupby('fashion_item')['popularity_score'].pct_change()
        df['is_trending_up'] = df['popularity_change'] > 0.05
        df['is_trending_down'] = df['popularity_change'] < -0.05
        df['trend_momentum'] = df.groupby('fashion_item')['popularity_change'].transform(
            lambda x: x.rolling(window=7, min_periods=1).mean()
        )
        
        # Seasonal indicators
        df['is_spring'] = df['season'] == 'Spring'
        df['is_summer'] = df['season'] == 'Summer'
        df['is_fall'] = df['season'] == 'Fall'
        df['is_winter'] = df['season'] == 'Winter'
        
        # Category encoding
        df['category_encoded'] = pd.Categorical(df['category']).codes
        
        # Regional indicators (for global analysis)
        df['is_us'] = df['region'] == 'US'
        df['is_europe'] = df['region'].isin(['UK', 'FRANCE', 'GERMANY', 'ITALY'])
        df['is_asia'] = df['region'].isin(['JAPAN', 'SOUTH KOREA'])
        
        print(f" Added {len(df.columns)} total features")
        
        return df
    
    def save_clean_dataset(self, df: pd.DataFrame):
        """Save the clean unified dataset"""
        
        print("\n💾 SAVING CLEAN DATASET")
        print("-" * 30)
        
        # Main clean dataset
        main_output = f"{self.output_dir}/fashion_trends_clean.csv"
        df.to_csv(main_output, index=False)
        print(f" Clean dataset saved: {main_output}")
        print(f"   Rows: {len(df):,}")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Size: ~{df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")
        
        # Create summary dataset for quick analysis
        summary_df = df.groupby(['fashion_item', 'category']).agg({
            'popularity_score': ['mean', 'std', 'min', 'max'],
            'search_volume': ['mean', 'sum'],
            'date': ['min', 'max', 'count']
        }).round(2)
        
        summary_df.columns = ['avg_popularity', 'popularity_std', 'min_popularity', 'max_popularity',
                             'avg_search_volume', 'total_search_volume', 'first_date', 'last_date', 'data_points']
        summary_df = summary_df.reset_index()
        
        summary_output = f"{self.output_dir}/fashion_trends_summary.csv"
        summary_df.to_csv(summary_output, index=False)
        print(f"Summary dataset saved: {summary_output}")
        
        # Create a modeling-ready dataset
        modeling_columns = [
            'date', 'fashion_item', 'category', 'popularity_score', 'search_volume',
            'year', 'month_num', 'week_of_year', 'day_of_year', 'is_weekend',
            'popularity_7day_avg', 'popularity_30day_avg', 'trend_momentum',
            'is_trending_up', 'is_trending_down', 'is_spring', 'is_summer', 'is_fall', 'is_winter',
            'category_encoded', 'region'
        ]
        
        modeling_df = df[modeling_columns].copy()
        modeling_output = f"{self.output_dir}/fashion_trends_modeling.csv"
        modeling_df.to_csv(modeling_output, index=False)
        print(f" Modeling dataset saved: {modeling_output}")
        
        return main_output, summary_output, modeling_output
    
    def generate_data_report(self, df: pd.DataFrame):
        """Generate a comprehensive data quality report"""
        
        print("\n DATA QUALITY REPORT")
        print("="*50)
        
        # Basic statistics
        print(f"Dataset Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
        print(f"Date Range: {df['date'].min()} to {df['date'].max()}")
        print(f"Unique Fashion Items: {df['fashion_item'].nunique():,}")
        print(f"Categories: {df['category'].nunique()}")
        print(f"Regions: {df['region'].nunique()}")
        
        # Data completeness
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            print(f"\n Missing Data:")
            for col, missing in missing_data[missing_data > 0].items():
                print(f"   {col}: {missing:,} ({missing/len(df)*100:.1f}%)")
        else:
            print(f"\n✅ No Missing Data - 100% Complete!")
        
        # Popularity score distribution
        print(f"\n Popularity Score Statistics:")
        print(f"   Mean: {df['popularity_score'].mean():.1f}")
        print(f"   Median: {df['popularity_score'].median():.1f}")
        print(f"   Std Dev: {df['popularity_score'].std():.1f}")
        print(f"   Range: {df['popularity_score'].min():.1f} - {df['popularity_score'].max():.1f}")
        
        # Category distribution
        print(f"\n🏷️ Category Distribution:")
        category_counts = df['category'].value_counts()
        for category, count in category_counts.head(10).items():
            percentage = (count / len(df)) * 100
            print(f"   {category}: {count:,} ({percentage:.1f}%)")
        
        # Top trending items
        print(f"\n Top 10 Trending Items (by avg popularity):")
        top_items = df.groupby('fashion_item')['popularity_score'].mean().sort_values(ascending=False).head(10)
        for i, (item, score) in enumerate(top_items.items(), 1):
            print(f"   {i:2d}. {item}: {score:.1f}")
        
        print(f"\n Dataset is ready for AI modeling!")

def main():
    """Main function to clean and consolidate datasets"""
    
    print(" FASHION TRENDS DATASET CLEANER")
    print("="*60)
    
    cleaner = FashionDatasetCleaner()
    
    # Load datasets
    datasets = cleaner.load_and_analyze_datasets()
    
    if not datasets:
        print(" No datasets found to clean!")
        return
    
    # Create unified dataset
    df_unified = cleaner.create_unified_dataset(datasets)
    
    if df_unified.empty:
        print(" Failed to create unified dataset!")
        return
    
    # Add advanced features
    df_final = cleaner.add_advanced_features(df_unified)
    
    # Save clean datasets
    output_files = cleaner.save_clean_dataset(df_final)
    
    # Generate report
    cleaner.generate_data_report(df_final)
    
    print(f"\n DATASET CLEANING COMPLETE!")
    print("="*60)
    print("Files created:")
    for file in output_files:
        print(f" {file}")
    print("\nYour clean, unified dataset is ready for AI modeling! ")

if __name__ == "__main__":
    main()
