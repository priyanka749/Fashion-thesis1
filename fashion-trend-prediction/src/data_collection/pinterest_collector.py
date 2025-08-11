"""
Pinterest Data Collection Script
Collects fashion-related pins and boards for trend analysis
"""

import requests
import pandas as pd
import json
import time
from datetime import datetime
import os
from typing import List, Dict, Any

class PinterestCollector:
    def __init__(self, access_token: str):
        """
        Initialize Pinterest API collector
        
        Args:
            access_token: Pinterest API access token
        """
        self.access_token = access_token
        self.base_url = "https://api.pinterest.com/v5"
        self.headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json"
        }
        
    def search_pins(self, query: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Search for pins based on query
        
        Args:
            query: Search term (e.g., "women fashion 2025", "trending outfits")
            limit: Number of pins to collect
            
        Returns:
            List of pin data dictionaries
        """
        pins = []
        url = f"{self.base_url}/search/pins"
        
        params = {
            "query": query,
            "limit": min(limit, 250)  # API limit per request
        }
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            pins.extend(data.get('items', []))
            
            print(f"Collected {len(pins)} pins for query: '{query}'")
            
        except requests.exceptions.RequestException as e:
            print(f"Error collecting pins for '{query}': {e}")
            
        return pins
    
    def get_board_pins(self, board_id: str, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get pins from a specific board
        
        Args:
            board_id: Pinterest board ID
            limit: Number of pins to collect
            
        Returns:
            List of pin data dictionaries
        """
        pins = []
        url = f"{self.base_url}/boards/{board_id}/pins"
        
        params = {"limit": min(limit, 100)}
        
        try:
            response = requests.get(url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            pins.extend(data.get('items', []))
            
            print(f"Collected {len(pins)} pins from board: {board_id}")
            
        except requests.exceptions.RequestException as e:
            print(f"Error collecting pins from board {board_id}: {e}")
            
        return pins
    
    def collect_fashion_data(self, save_path: str = "data/raw/pinterest_data.json"):
        """
        Collect comprehensive fashion data from Pinterest
        
        Args:
            save_path: Path to save collected data
        """
        all_pins = []
        
        # Fashion-related search queries
        fashion_queries = [
            "women fashion 2025",
            "trending outfits",
            "street style",
            "fashion inspiration",
            "outfit ideas",
            "style trends",
            "casual outfits",
            "formal wear",
            "summer fashion",
            "winter fashion",
            "accessories trends",
            "color trends fashion"
        ]
        
        print("Starting Pinterest data collection...")
        
        for query in fashion_queries:
            print(f"Collecting data for: {query}")
            pins = self.search_pins(query, limit=100)
            
            # Add metadata
            for pin in pins:
                pin['search_query'] = query
                pin['collection_date'] = datetime.now().isoformat()
                
            all_pins.extend(pins)
            
            # Rate limiting
            time.sleep(2)
        
        # Save data
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_pins, f, indent=2, ensure_ascii=False)
            
        print(f"Collected {len(all_pins)} total pins")
        print(f"Data saved to: {save_path}")
        
        return all_pins

def main():
    """
    Main function to run Pinterest data collection
    """
    # Note: You need to get your Pinterest API access token
    # Visit: https://developers.pinterest.com/docs/getting-started/authentication/
    
    ACCESS_TOKEN = "YOUR_PINTEREST_ACCESS_TOKEN"  # Replace with actual token
    
    if ACCESS_TOKEN == "YOUR_PINTEREST_ACCESS_TOKEN":
        print("Please set your Pinterest API access token in the script")
        print("Visit: https://developers.pinterest.com/docs/getting-started/authentication/")
        return
    
    collector = PinterestCollector(ACCESS_TOKEN)
    collector.collect_fashion_data()

if __name__ == "__main__":
    main()
