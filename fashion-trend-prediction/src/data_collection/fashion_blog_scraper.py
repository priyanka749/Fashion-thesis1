"""
Fashion Blog Scraper
Collects fashion content from blogs and magazines
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import json
import time
from datetime import datetime
import os
from typing import List, Dict, Any
import re

class FashionBlogScraper:
    def __init__(self):
        """Initialize the fashion blog scraper"""
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
    def scrape_fashion_articles(self, url: str) -> List[Dict[str, Any]]:
        """
        Scrape fashion articles from a given URL
        
        Args:
            url: URL of the fashion website
            
        Returns:
            List of article data dictionaries
        """
        articles = []
        
        try:
            response = requests.get(url, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Generic article selectors (adjust based on specific websites)
            article_selectors = [
                'article',
                '.post',
                '.article',
                '.entry',
                '.blog-post'
            ]
            
            for selector in article_selectors:
                found_articles = soup.select(selector)
                if found_articles:
                    break
            
            for article in found_articles[:20]:  # Limit to 20 articles per page
                article_data = self.extract_article_data(article, url)
                if article_data:
                    articles.append(article_data)
                    
        except Exception as e:
            print(f"Error scraping {url}: {e}")
            
        return articles
    
    def extract_article_data(self, article_element, source_url: str) -> Dict[str, Any]:
        """
        Extract data from an article element
        
        Args:
            article_element: BeautifulSoup element containing article
            source_url: Source URL of the website
            
        Returns:
            Dictionary with article data
        """
        try:
            # Extract title
            title_selectors = ['h1', 'h2', 'h3', '.title', '.headline']
            title = None
            for selector in title_selectors:
                title_elem = article_element.select_one(selector)
                if title_elem:
                    title = title_elem.get_text(strip=True)
                    break
            
            # Extract content
            content_selectors = ['.content', '.post-content', '.article-content', 'p']
            content = ""
            for selector in content_selectors:
                content_elems = article_element.select(selector)
                if content_elems:
                    content = ' '.join([elem.get_text(strip=True) for elem in content_elems])
                    break
            
            # Extract date
            date_selectors = ['.date', '.published', '.post-date', 'time']
            date = None
            for selector in date_selectors:
                date_elem = article_element.select_one(selector)
                if date_elem:
                    date = date_elem.get_text(strip=True)
                    break
            
            # Extract tags/categories
            tag_selectors = ['.tags', '.categories', '.labels']
            tags = []
            for selector in tag_selectors:
                tag_elems = article_element.select(selector + ' a')
                if tag_elems:
                    tags = [tag.get_text(strip=True) for tag in tag_elems]
                    break
            
            if title and content:
                return {
                    'title': title,
                    'content': content,
                    'date': date,
                    'tags': tags,
                    'source_url': source_url,
                    'scraped_at': datetime.now().isoformat(),
                    'word_count': len(content.split())
                }
                
        except Exception as e:
            print(f"Error extracting article data: {e}")
            
        return None
    
    def scrape_multiple_sources(self, save_path: str = "data/raw/fashion_blogs.json"):
        """
        Scrape multiple fashion blog sources
        
        Args:
            save_path: Path to save collected data
        """
        # Fashion blog and magazine URLs (add more as needed)
        fashion_urls = [
            # Note: Always check robots.txt and terms of service
            "https://www.whowhatwear.com/",
            "https://www.refinery29.com/en-us/style",
            "https://www.fashionista.com/",
            "https://www.popsugar.com/fashion",
            "https://stylecaster.com/fashion/",
            # Add more URLs here
        ]
        
        all_articles = []
        
        print("Starting fashion blog scraping...")
        
        for url in fashion_urls:
            print(f"Scraping: {url}")
            
            articles = self.scrape_fashion_articles(url)
            all_articles.extend(articles)
            
            print(f"Found {len(articles)} articles from {url}")
            
        
            time.sleep(3)
        
    
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(all_articles, f, indent=2, ensure_ascii=False)
        
        print(f"Scraped {len(all_articles)} total articles")
        print(f"Data saved to: {save_path}")
        
        return all_articles

def main():
    """
    Main function to run fashion blog scraping
    """
    scraper = FashionBlogScraper()
    
  
    print("Note: This scraper should only be used for educational/research purposes")
    print("Always check robots.txt and terms of service before scraping")
    
    scraper.scrape_multiple_sources()

if __name__ == "__main__":
    main()
