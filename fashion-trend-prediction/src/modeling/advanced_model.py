
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

try:
    from xgboost import XGBClassifier
    from lightgbm import LGBMClassifier
    from imblearn.over_sampling import SMOTE
    ADVANCED_LIBS_AVAILABLE = True
except ImportError:
    print('Please install xgboost, lightgbm, and imbalanced-learn.')
    ADVANCED_LIBS_AVAILABLE = False
    
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

class AdvancedFashionTrendModel:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.encoder = None
        self.feature_columns = None
        self.performance_metrics = {}
        
    def prepare_features(self, df):
        """Prepare features exactly as in the notebook"""
    
        date_col = [col for col in df.columns if 'date' in col.lower()][0]
        cat_col = [col for col in df.columns if 'category' in col.lower()][0]
        score_col = [col for col in df.columns if 'score' in col.lower() or 'trend' in col.lower()][0]
        
 
        df_work = df.copy()
        df_work[date_col] = pd.to_datetime(df_work[date_col], errors='coerce')
        df_work['year'] = df_work[date_col].dt.year
        df_work['month'] = df_work[date_col].dt.month
        df_work['day'] = df_work[date_col].dt.day
      
        if 'pop7_trendmom' not in df_work.columns and 'popularity_7day_avg' in df_work.columns and 'trend_momentum' in df_work.columns:
            df_work['pop7_trendmom'] = df_work['popularity_7day_avg'] * df_work['trend_momentum']
        
        # Select features
        features = ['year', 'month', 'day']
        for col in ['region', 'location', 'country', 'item', 'popularity_7day_avg', 'popularity_30day_avg', 'popularity_change', 'trend_momentum', 'search_volume', 'pop7_trendmom']:
            if col in df_work.columns:
                features.append(col)
        

        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        cat_encoded = encoder.fit_transform(df_work[[cat_col]])
        cat_encoded_df = pd.DataFrame(cat_encoded, columns=encoder.get_feature_names_out([cat_col]), index=df_work.index)
        
        X = pd.concat([df_work[features], cat_encoded_df], axis=1).fillna(0)
        median_score = df_work[score_col].median()
        y = (df_work[score_col] > median_score).astype(int)
        
        return X, y, encoder, features
    
    def train(self, df):
        """Train the advanced model using the exact notebook approach"""
        if not ADVANCED_LIBS_AVAILABLE:
            print("Advanced libraries not available, using basic model")
            return False
 
        X, y, encoder, features = self.prepare_features(df)
  
        if abs(y.mean() - 0.5) > 0.1 and X.shape[0] > 100:
            smote = SMOTE(random_state=42)
            X, y = smote.fit_resample(X, y)

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
     
        X_train = pd.DataFrame(X_train).select_dtypes(include=[np.number])
        X_test = pd.DataFrame(X_test).select_dtypes(include=[np.number])
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
      
        models = {
            'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'LightGBM': LGBMClassifier(random_state=42),
            'RandomForest': RandomForestClassifier(random_state=42),
            'LogisticRegression': LogisticRegression(max_iter=1000, random_state=42)
        }
        
        results = []
        best_model = None
        best_score = 0
        
        for name, model in models.items():
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            acc = accuracy_score(y_test, y_pred) * 100
            prec = precision_score(y_test, y_pred) * 100
            rec = recall_score(y_test, y_pred) * 100
            f1 = f1_score(y_test, y_pred) * 100
            
            results.append({'Model': name, 'Accuracy (%)': acc, 'Precision (%)': prec, 'Recall (%)': rec, 'F1 (%)': f1})
            
            if acc > best_score:
                best_score = acc
                best_model = model
                self.best_model_name = name
        
        # Store the best model and preprocessing objects
        self.model = best_model
        self.scaler = scaler
        self.encoder = encoder
        self.feature_columns = features
        
        # Store performance metrics
        results_df = pd.DataFrame(results)
        best_result = results_df.loc[results_df['Accuracy (%)'].idxmax()]
        
        self.performance_metrics = {
            'accuracy': float(best_result['Accuracy (%)']),
            'precision': float(best_result['Precision (%)']),
            'recall': float(best_result['Recall (%)']),
            'f1_score': float(best_result['F1 (%)']),
            'model_name': self.best_model_name
        }
        
      
        return True
    
    def predict(self, date_str, category):
        """Make prediction for a given date and category"""
        if self.model is None:
            raise ValueError("Model not trained yet")
            
        # Parse date
        from datetime import datetime
        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
        
        # Create feature vector (simplified for API use)
        features_dict = {
            'year': date_obj.year,
            'month': date_obj.month,
            'day': date_obj.day
        }
        
        for col in self.feature_columns:
            if col not in features_dict:
                features_dict[col] = 0
        
   
        feature_df = pd.DataFrame([features_dict])
        
     
        cat_features = [col for col in feature_df.columns if col in self.feature_columns]
        X = feature_df[cat_features].fillna(0)
     
        X = X.select_dtypes(include=[np.number])
        

        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        prediction = self.model.predict_proba(X_scaled)[0]
        predicted_class = self.model.predict(X_scaled)[0]
        
        return {
            'predicted_class': int(predicted_class),
            'probability_not_trending': float(prediction[0]),
            'probability_trending': float(prediction[1]),
            'confidence': float(max(prediction))
        }
    
    def get_performance_metrics(self):
        """Return the performance metrics"""
        return self.performance_metrics
