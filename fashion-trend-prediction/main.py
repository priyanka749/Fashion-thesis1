"""
Main execution script for Fashion Trend Prediction System
Orchestrates the entire pipeline from data collection to analysis
"""

import os
import sys
import argparse
from datetime import datetime

# Add src to path
sys.path.append('src')

from data_collection.google_trends_collector import GoogleTrendsCollector
from data_collection.fashion_blog_scraper import FashionBlogScraper
from preprocessing.data_preprocessor import FashionDataPreprocessor
from modeling.trend_predictor import FashionTrendPredictor
from analysis.trend_analyzer import FashionTrendAnalyzer

def run_data_collection():
    """Run data collection from all sources"""
    print("=" * 60)
    print("PHASE 1: DATA COLLECTION")
    print("=" * 60)
    
  
    print("\n1. Collecting Google Trends data...")
    try:
        trends_collector = GoogleTrendsCollector()
        trends_collector.collect_comprehensive_fashion_trends()
        print("✓ Google Trends data collection completed")
    except Exception as e:
        print(f"✗ Google Trends collection failed: {e}")
    
    # Fashion Blogs
    print("\n2. Collecting fashion blog data...")
    try:
        blog_scraper = FashionBlogScraper()
        print("Note: Blog scraping requires careful consideration of robots.txt and ToS")
 
        print("✓ Blog scraping setup completed (disabled for demo)")
    except Exception as e:
        print(f"✗ Blog scraping failed: {e}")
    
    # Pinterest (requires API token)
    print("\n3. Pinterest data collection...")
    print("Note: Pinterest collection requires API access token")
    print("Visit: https://developers.pinterest.com/docs/getting-started/authentication/")
    print("✓ Pinterest setup instructions provided")

def run_preprocessing():
    """Run data preprocessing"""
    print("\n" + "=" * 60)
    print("PHASE 2: DATA PREPROCESSING")
    print("=" * 60)
    
    try:
        preprocessor = FashionDataPreprocessor()
        preprocessor.combine_all_data()
        print("✓ Data preprocessing completed")
    except Exception as e:
        print(f"✗ Data preprocessing failed: {e}")

def run_modeling():
    """Run machine learning modeling"""
    print("\n" + "=" * 60)
    print("PHASE 3: MACHINE LEARNING MODELING")
    print("=" * 60)
    
    try:
        # Check if processed data exists
        processed_files = [
            "data/processed/google_trends_processed.csv",
            "data/processed/fashion_blogs_processed.csv",
            "data/processed/pinterest_processed.csv"
        ]
        
        existing_files = [f for f in processed_files if os.path.exists(f)]
        
        if not existing_files:
            print("No processed data found. Skipping modeling phase.")
            print("Please run data collection and preprocessing first.")
            return
        
        from modeling.trend_predictor import main as modeling_main
        modeling_main()
        print("✓ Machine learning modeling completed")
    except Exception as e:
        print(f"✗ Modeling failed: {e}")

def run_analysis():
    """Run trend analysis and visualization"""
    print("\n" + "=" * 60)
    print("PHASE 4: TREND ANALYSIS & VISUALIZATION")
    print("=" * 60)
    
    try:
        from analysis.trend_analyzer import main as analysis_main
        analysis_main()
        print("✓ Trend analysis and visualization completed")
    except Exception as e:
        print(f"✗ Analysis failed: {e}")

def setup_environment():
    """Setup the project environment"""
    print("Setting up project environment...")
    
    # Create necessary directories
    directories = [
        "data/raw",
        "data/processed", 
        "data/external",
        "models",
        "outputs",
        "config"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✓ Project directories created")

def install_dependencies():
    """Install required Python packages"""
    print("\nInstalling dependencies...")
    print("Run the following command to install required packages:")
    print("pip install -r requirements.txt")
    print("\nNote: Some packages might require additional setup (e.g., spaCy models)")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Fashion Trend Prediction System')
    parser.add_argument('--phase', choices=['setup', 'collect', 'preprocess', 'model', 'analyze', 'all'],
                       default='all', help='Phase to run')
    parser.add_argument('--install-deps', action='store_true', help='Show dependency installation instructions')
    
    args = parser.parse_args()
    
    print(" Fashion Trend Prediction System")
    print("AI-Based Fashion Trend Analysis for Women Aged 20-25")
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    if args.install_deps:
        install_dependencies()
        return
    
    if args.phase in ['setup', 'all']:
        setup_environment()
    
    if args.phase in ['collect', 'all']:
        run_data_collection()
    
    if args.phase in ['preprocess', 'all']:
        run_preprocessing()
    
    if args.phase in ['model', 'all']:
        run_modeling()
    
    if args.phase in ['analyze', 'all']:
        run_analysis()
    
    print("\n" + "=" * 80)
    print(" EXECUTION COMPLETED!")
    print("=" * 80)
    
    # Show output summary
    print("\nGenerated Outputs:")
    output_files = [
        "data/raw/google_trends.json",
        "data/processed/google_trends_processed.csv",
        "models/random_forest_model.pkl",
        "outputs/fashion_wordcloud.png",
        "outputs/fashion_dashboard.html",
        "outputs/fashion_trend_report.md"
    ]
    
    for file_path in output_files:
        if os.path.exists(file_path):
            print(f"✓ {file_path}")
        else:
            print(f"- {file_path} (not generated)")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()
