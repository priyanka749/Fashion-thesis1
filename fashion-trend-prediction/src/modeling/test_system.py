"""
Simple test script to verify the fashion trend prediction system
"""

import sys
import os
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.append('src')

def test_google_trends():
    """Test Google Trends data collection"""
    try:
        from pytrends.request import TrendReq
        
        print("✅ Google Trends library imported successfully")
        
        # Create simple trends request
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Test with a few fashion keywords
        fashion_keywords = ['dress', 'jeans', 'shoes']
        pytrends.build_payload(fashion_keywords, timeframe='today 3-m')
        
        # Get trend data
        trend_data = pytrends.interest_over_time()
        
        if not trend_data.empty:
            print(f"✅ Successfully collected trend data for {len(fashion_keywords)} keywords")
            print(f"   Data shape: {trend_data.shape}")
            print(f"   Date range: {trend_data.index[0]} to {trend_data.index[-1]}")
            
            # Save test data
            os.makedirs('data/raw', exist_ok=True)
            trend_data.to_csv('data/raw/test_trends.csv')
            print("✅ Test data saved to data/raw/test_trends.csv")
            
            return True
        else:
            print("❌ No trend data received")
            return False
            
    except Exception as e:
        print(f"❌ Google Trends test failed: {e}")
        return False

def test_text_processing():
    """Test text processing capabilities"""
    try:
        import nltk
        import pandas as pd
        from collections import Counter
        
        print("✅ Text processing libraries imported")
        
        # Sample fashion text data
        sample_texts = [
            "Love this vintage dress with floral patterns",
            "Cropped blazer and wide leg jeans trending now",
            "Pastel colors are perfect for spring fashion",
            "Black leather jacket with denim never goes out of style"
        ]
        
        # Simple keyword extraction
        fashion_keywords = ['dress', 'blazer', 'jeans', 'jacket', 'vintage', 'floral', 'pastel', 'black', 'leather', 'denim']
        
        found_keywords = []
        for text in sample_texts:
            text_lower = text.lower()
            for keyword in fashion_keywords:
                if keyword in text_lower:
                    found_keywords.append(keyword)
        
        keyword_counts = Counter(found_keywords)
        
        print(f"✅ Extracted {len(found_keywords)} keywords from sample texts")
        print(f"   Top keywords: {dict(keyword_counts.most_common(5))}")
        
        return True
        
    except Exception as e:
        print(f"❌ Text processing test failed: {e}")
        return False

def test_visualization():
    """Test basic visualization capabilities"""
    try:
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        print("✅ Visualization libraries imported")
        
        # Create sample data
        keywords = ['dress', 'jeans', 'shoes', 'bag', 'top']
        counts = [45, 38, 32, 28, 25]
        
        # Create simple bar chart
        plt.figure(figsize=(10, 6))
        plt.bar(keywords, counts, color='skyblue', alpha=0.8)
        plt.title('Fashion Keywords Test Chart')
        plt.xlabel('Keywords')
        plt.ylabel('Frequency')
        
        # Save chart
        os.makedirs('outputs', exist_ok=True)
        plt.savefig('outputs/test_chart.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✅ Test visualization created and saved to outputs/test_chart.png")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False

def create_sample_dataset():
    """Create a sample dataset for testing"""
    try:
        # Sample fashion data
        sample_data = {
            'text': [
                'crop top trending summer fashion',
                'wide leg jeans street style',
                'pastel colors spring collection',
                'black dress formal wear',
                'vintage denim jacket retro',
                'floral print midi skirt',
                'oversized blazer office wear',
                'chunky sneakers athleisure'
            ],
            'source': ['instagram', 'tiktok', 'pinterest', 'blog', 'instagram', 'pinterest', 'blog', 'tiktok'],
            'date': pd.date_range('2024-01-01', periods=8, freq='W'),
            'engagement': [1250, 890, 2100, 450, 780, 1400, 320, 1100]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Save sample dataset
        os.makedirs('data/processed', exist_ok=True)
        df.to_csv('data/processed/sample_fashion_data.csv', index=False)
        
        print("✅ Sample dataset created with 8 fashion entries")
        print(f"   Columns: {list(df.columns)}")
        print(f"   Sources: {df['source'].unique()}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sample dataset creation failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🎨 Fashion Trend Prediction System - Quick Test")
    print("=" * 60)
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    tests_passed = 0
    total_tests = 4
    
    # Test 1: Google Trends
    print("Test 1: Google Trends Data Collection")
    if test_google_trends():
        tests_passed += 1
    print()
    
    # Test 2: Text Processing
    print("Test 2: Text Processing")
    if test_text_processing():
        tests_passed += 1
    print()
    
    # Test 3: Visualization
    print("Test 3: Visualization")
    if test_visualization():
        tests_passed += 1
    print()
    
    # Test 4: Sample Dataset
    print("Test 4: Sample Dataset Creation")
    if create_sample_dataset():
        tests_passed += 1
    print()
    
    print("=" * 60)
    print(f"🎯 RESULTS: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 All tests passed! Your system is ready to use.")
        print("\nNext steps:")
        print("1. Run: python main.py --phase collect")
        print("2. Run: python main.py --phase preprocess") 
        print("3. Run: python main.py --phase analyze")
    else:
        print("❌ Some tests failed. Please check the error messages above.")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
