"""
Simple Google Trends Collector - Avoids Rate Limits
Collects fewer keywords with longer delays
"""

from pytrends.request import TrendReq
import pandas as pd
import time
import json
import os
from datetime import datetime

def collect_limited_trends():
    """Collect trends for a limited set of keywords to avoid rate limits"""
    
    # Reduced keyword set - only most important trends
    priority_keywords = [
        ['crop top', 'wide leg jeans', 'oversized blazer'],
        ['tennis skirt', 'cargo pants', 'slip dress'],
        ['chunky sneakers', 'mini bag', 'silk scarf'],
        ['y2k fashion', 'clean girl aesthetic']
    ]
    
    pytrends = TrendReq(hl='en-US', tz=360)
    all_data = {'trend_data': {}, 'collection_date': datetime.now().isoformat()}
    
    print("🔄 Collecting limited trends with rate limiting...")
    
    for i, keyword_group in enumerate(priority_keywords):
        print(f"📊 Group {i+1}: {keyword_group}")
        
        try:
            # Build payload with longer timeframe
            pytrends.build_payload(keyword_group, timeframe='today 6-m')
            
            # Get trend data
            trend_data = pytrends.interest_over_time()
            
            if not trend_data.empty:
                trend_data = trend_data.drop('isPartial', axis=1, errors='ignore')
                all_data['trend_data'][f'group_{i+1}'] = {
                    'keywords': keyword_group,
                    'data': trend_data.to_dict('index')
                }
                print(f"✅ Successfully collected {len(keyword_group)} keywords")
            
        except Exception as e:
            print(f"❌ Error with group {i+1}: {e}")
        
        # Long delay to avoid rate limits
        print("⏳ Waiting 15 seconds...")
        time.sleep(15)
    
    # Save data
    os.makedirs('data/raw', exist_ok=True)
    with open('data/raw/google_trends_limited.json', 'w') as f:
        json.dump(all_data, f, indent=2, default=str)
    
    print("✅ Limited trends collection complete!")

if __name__ == "__main__":
    collect_limited_trends()
