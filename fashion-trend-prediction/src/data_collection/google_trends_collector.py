"""
Google Trends Data Collection Script
Collects fashion trend data from Google Trends
"""

from pytrends.request import TrendReq
import pandas as pd
import time
from datetime import datetime, timedelta
import json
import os
from typing import List, Dict

class GoogleTrendsCollector:
    def __init__(self, hl='en-US', tz=360):
        """
        Initialize Google Trends collector
        
        Args:
            hl: Language (default: 'en-US')
            tz: Timezone offset (default: 360 for US Central)
        """
        self.pytrends = TrendReq(hl=hl, tz=tz)
        
    def get_fashion_trends(self, keywords: List[str], timeframe: str = 'today 12-m') -> pd.DataFrame:
        """
        Get trend data for fashion keywords
        
        Args:
            keywords: List of fashion keywords to analyze
            timeframe: Time period for analysis
            
        Returns:
            DataFrame with trend data
        """
        try:
            self.pytrends.build_payload(keywords, timeframe=timeframe)
            trend_data = self.pytrends.interest_over_time()
            
            if not trend_data.empty:
                trend_data = trend_data.drop('isPartial', axis=1, errors='ignore')
                
            return trend_data
            
        except Exception as e:
            print(f"Error getting trends for {keywords}: {e}")
            return pd.DataFrame()
    
    def get_related_queries(self, keyword: str) -> Dict:
        """
        Get related queries for a keyword
        
        Args:
            keyword: Fashion keyword to analyze
            
        Returns:
            Dictionary with related queries
        """
        try:
            self.pytrends.build_payload([keyword])
            related_queries = self.pytrends.related_queries()
            return related_queries.get(keyword, {})
            
        except Exception as e:
            print(f"Error getting related queries for {keyword}: {e}")
            return {}
    
    def get_trending_searches(self, geo: str = 'US') -> pd.DataFrame:
        """
        Get trending searches for a specific geography
        
        Args:
            geo: Geographic location (default: 'US')
            
        Returns:
            DataFrame with trending searches
        """
        try:
            trending_searches = self.pytrends.trending_searches(pn=geo)
            return trending_searches
            
        except Exception as e:
            print(f"Error getting trending searches for {geo}: {e}")
            return pd.DataFrame()
    
    def collect_comprehensive_fashion_trends(self, save_path: str = "data/raw/google_trends.json"):
        """
        Collect comprehensive fashion trend data
        
        Args:
            save_path: Path to save collected data
        """
        fashion_keywords = [
            # Trending tops
            ['crop top', 'mesh top', 'halter top', 'tube top', 'off shoulder top'],
            ['oversized blazer', 'cropped cardigan', 'puff sleeve top', 'corset top', 'wrap top'],
            
            # Trending bottoms
            ['wide leg jeans', 'flare jeans', 'mom jeans', 'cargo pants', 'bike shorts'],
            ['midi skirt', 'mini skirt', 'pleated skirt', 'slip skirt', 'tennis skirt'],
            ['palazzo pants', 'straight leg pants', 'high waisted jeans', 'baggy jeans', 'leather pants'],
            
            # Trending dresses
            ['maxi dress', 'midi dress', 'slip dress', 'wrap dress', 'shirt dress'],
            ['bodycon dress', 'flowy dress', 'cut out dress', 'backless dress', 'asymmetric dress'],
            
            # Outerwear trends
            ['oversized jacket', 'leather jacket', 'denim jacket', 'bomber jacket', 'puffer jacket'],
            ['trench coat', 'shacket', 'cape coat', 'teddy coat', 'long coat'],
            
            # Trending styles
            ['y2k fashion', 'cottagecore', 'dark academia', 'streetwear', 'coquette style'],
            ['minimalist fashion', 'maximalist fashion', 'grunge style', 'preppy style', 'boho chic'],
            ['clean girl aesthetic', 'old money style', 'coastal grandmother', 'dopamine dressing', 'quiet luxury'],
            
            # Color trends 2024-2025
            ['sage green', 'lavender purple', 'butter yellow', 'terracotta orange', 'sky blue'],
            ['cream white', 'chocolate brown', 'cherry red', 'forest green', 'dusty pink'],
            
            # Current pattern trends
            ['leopard print', 'zebra print', 'cow print', 'tie dye', 'gingham'],
            ['polka dots', 'stripes', 'plaid', 'floral print', 'abstract print'],
            
            # Trending accessories
            ['platform shoes', 'chunky sneakers', 'mary jane shoes', 'combat boots', 'knee high boots'],
            ['bucket hat', 'beanie', 'silk scarf', 'statement earrings', 'layered necklaces'],
            ['mini bag', 'oversized bag', 'tote bag', 'crossbody bag', 'belt bag'],
            
            # Fabric trends
            ['satin fabric', 'corduroy', 'velvet', 'linen', 'mesh fabric'],
            ['faux leather', 'knit fabric', 'denim', 'silk', 'cotton blend'],
            
            # Seasonal 2025 trends
            ['summer 2025 trends', 'fall 2025 fashion', 'spring fashion trends', 'winter style 2025'],
            ['resort wear', 'vacation outfits', 'work from home style', 'date night outfit', 'brunch outfit']
        ]
        
        all_data = {
            'trend_data': {},
            'related_queries': {},
            'collection_date': datetime.now().isoformat()
        }
        
        print("Starting Google Trends data collection...")
        
        # Collect trend data for each keyword group
        for i, keyword_group in enumerate(fashion_keywords):
            print(f"Collecting trends for group {i+1}: {keyword_group}")
            
            # Get trend data
            trend_data = self.get_fashion_trends(keyword_group)
            if not trend_data.empty:
                group_name = f"group_{i+1}"
                all_data['trend_data'][group_name] = {
                    'keywords': keyword_group,
                    'data': trend_data.to_dict('index')
                }
            
            # Get related queries for each keyword
            for keyword in keyword_group:
                related = self.get_related_queries(keyword)
                if related:
                    all_data['related_queries'][keyword] = related
                
                time.sleep(5)  # Increased rate limiting to avoid 429 errors
            
            time.sleep(10)  # Increased rate limiting between groups
        
        # Save data
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, indent=2, default=str)
        
        print(f"Google Trends data saved to: {save_path}")
        return all_data

def main():
    """
    Main function to run Google Trends data collection
    """
    collector = GoogleTrendsCollector()
    collector.collect_comprehensive_fashion_trends()

if __name__ == "__main__":
    main()
