"""
Fashion Trend Prediction Models
Machine learning models for predicting fashion trends
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import os
from datetime import datetime
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

class FashionTrendPredictor:
    def __init__(self):
        """Initialize the fashion trend predictor"""
        self.models = {}
        self.vectorizers = {}
        self.label_encoders = {}
        self.scalers = {}
        
    def prepare_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare features from fashion data
        
        Args:
            df: DataFrame with processed fashion data
            
        Returns:
            Tuple of features and labels
        """
      
        text_features = []
        
      
        if 'fashion_keywords' in df.columns:
            df['keywords_text'] = df['fashion_keywords'].apply(
                lambda x: ' '.join(x) if isinstance(x, list) else str(x)
            )
            text_features.append('keywords_text')
        
        if 'description_clean' in df.columns:
            text_features.append('description_clean')
        
        if 'full_text' in df.columns:
            text_features.append('full_text')
       
        text_data = ""
        for col in text_features:
            if col in df.columns:
                text_data = df[col].fillna('').astype(str)
                break
   
        if 'tfidf' not in self.vectorizers:
            self.vectorizers['tfidf'] = TfidfVectorizer(
                max_features=1000,
                stop_words='english',
                ngram_range=(1, 2)
            )
            text_vectors = self.vectorizers['tfidf'].fit_transform(text_data)
        else:
            text_vectors = self.vectorizers['tfidf'].transform(text_data)
      
        numerical_features = []
        
        if 'trend_value' in df.columns:
            numerical_features.append('trend_value')
        
        if 'word_count' in df.columns:
            numerical_features.append('word_count')
    
        if numerical_features:
            num_data = df[numerical_features].fillna(0)
            
            if 'scaler' not in self.scalers:
                self.scalers['scaler'] = StandardScaler()
                num_vectors = self.scalers['scaler'].fit_transform(num_data)
            else:
                num_vectors = self.scalers['scaler'].transform(num_data)
            
            # Combine text and numerical features
            features = np.hstack([text_vectors.toarray(), num_vectors])
        else:
            features = text_vectors.toarray()
        
        return features
    
    def create_trend_labels(self, df: pd.DataFrame) -> np.ndarray:
        """
        Create trend labels for classification
        
        Args:
            df: DataFrame with fashion data
            
        Returns:
            Array of trend labels
        """
        # Create labels based on different criteria
        labels = []
        
        for _, row in df.iterrows():

            label = "neutral"
            
            # Check for trending keywords
            if isinstance(row.get('fashion_keywords'), list):
                keywords = row['fashion_keywords']
                
                # High trend indicators
                high_trend_words = ['trending', 'viral', 'popular', 'hot', 'new']
                if any(word in ' '.join(keywords) for word in high_trend_words):
                    label = "trending"
                
                # Seasonal trends
                seasonal_words = ['summer', 'winter', 'spring', 'fall']
                if any(word in ' '.join(keywords) for word in seasonal_words):
                    label = "seasonal"
           
            if 'trend_value' in row and pd.notna(row['trend_value']):
                trend_val = float(row['trend_value'])
                if trend_val > 70:
                    label = "trending"
                elif trend_val > 40:
                    label = "moderate"
                else:
                    label = "low"
            
            labels.append(label)
        

        if 'trend_labels' not in self.label_encoders:
            self.label_encoders['trend_labels'] = LabelEncoder()
            encoded_labels = self.label_encoders['trend_labels'].fit_transform(labels)
        else:
            encoded_labels = self.label_encoders['trend_labels'].transform(labels)
        
        return encoded_labels
    
    def train_models(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train multiple models for trend prediction
        
        Args:
            df: Training DataFrame
            
        Returns:
            Dictionary with model performance metrics
        """
        print("Preparing features...")
        X = self.prepare_features(df)
        y = self.create_trend_labels(df)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Define models
        model_configs = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(random_state=42),
            'naive_bayes': MultinomialNB()
        }
        
        results = {}
        
        print("Training models...")
        for name, model in model_configs.items():
            print(f"Training {name}...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predictions
            y_pred = model.predict(X_test)
            
            # Metrics
            accuracy = accuracy_score(y_test, y_pred)
            cv_scores = cross_val_score(model, X_train, y_train, cv=5)
            
            results[name] = {
                'model': model,
                'accuracy': accuracy,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred),
                'confusion_matrix': confusion_matrix(y_test, y_pred)
            }
            
            # Save model
            self.models[name] = model
            
            print(f"{name} - Accuracy: {accuracy:.3f}, CV Score: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return results
    
    def predict_trends(self, new_data: pd.DataFrame, model_name: str = 'random_forest') -> np.ndarray:
        """
        Predict trends for new data
        
        Args:
            new_data: DataFrame with new fashion data
            model_name: Name of the model to use
            
        Returns:
            Array of predictions
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained yet")
  
        X_new = self.prepare_features(new_data)
 
        predictions = self.models[model_name].predict(X_new)
        
     
        decoded_predictions = self.label_encoders['trend_labels'].inverse_transform(predictions)
        
        return decoded_predictions
    
    def save_models(self, save_dir: str = "models/"):
        """
        Save trained models and preprocessors
        
        Args:
            save_dir: Directory to save models
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save models
        for name, model in self.models.items():
            joblib.dump(model, f"{save_dir}/{name}_model.pkl")
        
        # Save vectorizers
        for name, vectorizer in self.vectorizers.items():
            joblib.dump(vectorizer, f"{save_dir}/{name}_vectorizer.pkl")
        
        # Save label encoders
        for name, encoder in self.label_encoders.items():
            joblib.dump(encoder, f"{save_dir}/{name}_encoder.pkl")
        
        # Save scalers
        for name, scaler in self.scalers.items():
            joblib.dump(scaler, f"{save_dir}/{name}_scaler.pkl")
        
        print(f"Models saved to {save_dir}")
    
    def load_models(self, save_dir: str = "models/"):
        """
        Load trained models and preprocessors
        
        Args:
            save_dir: Directory containing saved models
        """
        # Load models
        model_files = [f for f in os.listdir(save_dir) if f.endswith('_model.pkl')]
        for model_file in model_files:
            name = model_file.replace('_model.pkl', '')
            self.models[name] = joblib.load(f"{save_dir}/{model_file}")
        
        # Load vectorizers
        vectorizer_files = [f for f in os.listdir(save_dir) if f.endswith('_vectorizer.pkl')]
        for vec_file in vectorizer_files:
            name = vec_file.replace('_vectorizer.pkl', '')
            self.vectorizers[name] = joblib.load(f"{save_dir}/{vec_file}")
        
        # Load encoders
        encoder_files = [f for f in os.listdir(save_dir) if f.endswith('_encoder.pkl')]
        for enc_file in encoder_files:
            name = enc_file.replace('_encoder.pkl', '')
            self.label_encoders[name] = joblib.load(f"{save_dir}/{enc_file}")
        
        # Load scalers
        scaler_files = [f for f in os.listdir(save_dir) if f.endswith('_scaler.pkl')]
        for scaler_file in scaler_files:
            name = scaler_file.replace('_scaler.pkl', '')
            self.scalers[name] = joblib.load(f"{save_dir}/{scaler_file}")
        
        print(f"Models loaded from {save_dir}")
    
    def visualize_results(self, results: Dict[str, Any], save_path: str = "outputs/model_comparison.png"):
        """
        Visualize model performance
        
        Args:
            results: Results from model training
            save_path: Path to save visualization
        """
        # Extract accuracy scores
        model_names = list(results.keys())
        accuracies = [results[name]['accuracy'] for name in model_names]
        cv_means = [results[name]['cv_mean'] for name in model_names]
        

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        ax1.bar(model_names, accuracies, alpha=0.7, color='skyblue')
        ax1.set_title('Model Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
       
        ax2.bar(model_names, cv_means, alpha=0.7, color='lightcoral')
        ax2.set_title('Cross-Validation Score Comparison')
        ax2.set_ylabel('CV Mean Score')
        ax2.set_ylim(0, 1)
        for i, v in enumerate(cv_means):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
      
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Model comparison saved to {save_path}")

def main():
    """
    Main function to run model training
    """
    # Load processed data
    data_files = [
        "data/processed/pinterest_processed.csv",
        "data/processed/fashion_blogs_processed.csv",
        "data/processed/google_trends_processed.csv"
    ]
    
    all_data = []
    for file_path in data_files:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            all_data.append(df)
            print(f"Loaded {len(df)} records from {file_path}")
    
    if not all_data:
        print("No processed data found. Please run data collection and preprocessing first.")
        return

    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total records: {len(combined_df)}")
    
 
    predictor = FashionTrendPredictor()
    

    results = predictor.train_models(combined_df)
    
 
    predictor.save_models()
    
 
    predictor.visualize_results(results)

if __name__ == "__main__":
    main()
