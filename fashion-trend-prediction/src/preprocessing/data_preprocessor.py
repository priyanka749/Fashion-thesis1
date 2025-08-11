"""
Data Preprocessing Pipeline
Cleans and prepares collected fashion data for analysis
"""

import pandas as pd
import numpy as np
import json
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from datetime import datetime
import os
from typing import List, Dict, Any

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class FashionDataPreprocessor:
    def __init__(self):
        """Initialize the data preprocessor"""
        self.stop_words = set(stopwords.words('english'))
        self.lemmatizer = WordNetLemmatizer()
        
        # Fashion-specific terms to keep (don't remove as stopwords)
        self.fashion_terms = {
            'style', 'fashion', 'trend', 'outfit', 'wear', 'clothing', 
            'dress', 'shirt', 'pants', 'skirt', 'shoes', 'accessories',
            'color', 'pattern', 'fabric', 'design', 'look', 'aesthetic'
        }
        
        # Remove fashion terms from stopwords
        self.stop_words = self.stop_words - self.fashion_terms
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text data
        
        Args:
            text: Raw text to clean
            
        Returns:
            Cleaned text
        """
        if not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove mentions and hashtags (but keep the word)
        text = re.sub(r'[@#]\w+', lambda m: m.group()[1:], text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove punctuation except for important fashion terms
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        return text.strip()
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text
        
        Args:
            text: Cleaned text
            
        Returns:
            List of processed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        processed_tokens = []
        for token in tokens:
            if token not in self.stop_words and len(token) > 2:
                lemmatized = self.lemmatizer.lemmatize(token)
                processed_tokens.append(lemmatized)
        
        return processed_tokens
    
    def extract_fashion_keywords(self, text: str) -> List[str]:
        """
        Extract fashion-specific keywords from text
        
        Args:
            text: Text to analyze
            
        Returns:
            List of fashion keywords
        """
        fashion_keywords = {
            # Clothing items
            'top', 'shirt', 'blouse', 'sweater', 'cardigan', 'jacket', 'coat',
            'dress', 'skirt', 'pants', 'jeans', 'shorts', 'leggings', 'jumpsuit',
            'shoes', 'boots', 'sneakers', 'heels', 'sandals', 'flats',
            
            # Styles
            'casual', 'formal', 'bohemian', 'minimalist', 'vintage', 'modern',
            'classic', 'trendy', 'chic', 'elegant', 'edgy', 'feminine',
            
            # Colors
            'black', 'white', 'red', 'blue', 'green', 'yellow', 'pink', 'purple',
            'orange', 'brown', 'gray', 'navy', 'beige', 'nude', 'neon', 'pastel',
            
            # Patterns
            'stripes', 'floral', 'geometric', 'animal', 'print', 'solid', 'polka',
            
            # Fabrics
            'cotton', 'silk', 'wool', 'denim', 'leather', 'lace', 'chiffon',
            'velvet', 'polyester', 'linen', 'cashmere'
        }
        
        tokens = self.tokenize_and_lemmatize(text)
        found_keywords = [token for token in tokens if token in fashion_keywords]
        
        return list(set(found_keywords))  # Remove duplicates
    
    def process_pinterest_data(self, file_path: str) -> pd.DataFrame:
        """
        Process Pinterest data
        
        Args:
            file_path: Path to Pinterest JSON data
            
        Returns:
            Processed DataFrame
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.json_normalize(data)
            
            # Clean and process text fields
            if 'description' in df.columns:
                df['description_clean'] = df['description'].apply(self.clean_text)
                df['description_tokens'] = df['description_clean'].apply(self.tokenize_and_lemmatize)
                df['fashion_keywords'] = df['description'].apply(self.extract_fashion_keywords)
            
            # Process dates
            if 'created_at' in df.columns:
                df['created_at'] = pd.to_datetime(df['created_at'], errors='coerce')
            
            # Add metadata
            df['data_source'] = 'pinterest'
            df['processed_at'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"Error processing Pinterest data: {e}")
            return pd.DataFrame()
    
    def process_blog_data(self, file_path: str) -> pd.DataFrame:
        """
        Process fashion blog data
        
        Args:
            file_path: Path to blog JSON data
            
        Returns:
            Processed DataFrame
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            df = pd.DataFrame(data)
            
            # Clean and process text fields
            df['title_clean'] = df['title'].apply(self.clean_text)
            df['content_clean'] = df['content'].apply(self.clean_text)
            
            # Combine title and content for analysis
            df['full_text'] = df['title_clean'] + ' ' + df['content_clean']
            df['full_text_tokens'] = df['full_text'].apply(self.tokenize_and_lemmatize)
            df['fashion_keywords'] = df['full_text'].apply(self.extract_fashion_keywords)
            
            # Process dates
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'], errors='coerce')
            
            # Add metadata
            df['data_source'] = 'fashion_blogs'
            df['processed_at'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"Error processing blog data: {e}")
            return pd.DataFrame()
    
    def process_trends_data(self, file_path: str) -> pd.DataFrame:
        """
        Process Google Trends data
        
        Args:
            file_path: Path to trends JSON data
            
        Returns:
            Processed DataFrame
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            trends_list = []
            
            # Process trend data
            for group_name, group_data in data.get('trend_data', {}).items():
                keywords = group_data.get('keywords', [])
                trend_data = group_data.get('data', {})
                
                for date_str, values in trend_data.items():
                    for keyword in keywords:
                        if keyword in values:
                            trends_list.append({
                                'date': date_str,
                                'keyword': keyword,
                                'trend_value': values[keyword],
                                'group': group_name
                            })
            
            df = pd.DataFrame(trends_list)
            
            if not df.empty:
                df['date'] = pd.to_datetime(df['date'])
                df['data_source'] = 'google_trends'
                df['processed_at'] = datetime.now()
            
            return df
            
        except Exception as e:
            print(f"Error processing trends data: {e}")
            return pd.DataFrame()
    
    def combine_all_data(self, output_path: str = "data/processed/combined_fashion_data.csv"):
        """
        Combine all processed data sources
        
        Args:
            output_path: Path to save combined data
        """
        combined_data = []
        
        # Process Pinterest data
        pinterest_path = "data/raw/pinterest_data.json"
        if os.path.exists(pinterest_path):
            pinterest_df = self.process_pinterest_data(pinterest_path)
            if not pinterest_df.empty:
                combined_data.append(pinterest_df)
                print(f"Processed {len(pinterest_df)} Pinterest records")
        
        # Process blog data
        blog_path = "data/raw/fashion_blogs.json"
        if os.path.exists(blog_path):
            blog_df = self.process_blog_data(blog_path)
            if not blog_df.empty:
                combined_data.append(blog_df)
                print(f"Processed {len(blog_df)} blog records")
        
        # Process trends data
        trends_path = "data/raw/google_trends.json"
        if os.path.exists(trends_path):
            trends_df = self.process_trends_data(trends_path)
            if not trends_df.empty:
                combined_data.append(trends_df)
                print(f"Processed {len(trends_df)} trend records")
        
        if combined_data:
            # Save individual processed datasets
            os.makedirs("data/processed", exist_ok=True)
            
            for i, df in enumerate(combined_data):
                source = df['data_source'].iloc[0] if 'data_source' in df.columns else f'source_{i}'
                df.to_csv(f"data/processed/{source}_processed.csv", index=False)
            
            print(f"Processed data saved to data/processed/")
        else:
            print("No data files found to process")

def main():
    """
    Main function to run data preprocessing
    """
    preprocessor = FashionDataPreprocessor()
    preprocessor.combine_all_data()

if __name__ == "__main__":
    main()
