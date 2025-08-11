# Fashion Trend Prediction System - User Guide

## 🎯 Overview

This system uses AI and machine learning to predict fashion trends for women aged 20-25 by analyzing data from social media, fashion blogs, and search trends.

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download the project
cd fashion-trend-prediction

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (required for text processing)
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')"
```

### 2. Basic Usage

```bash
# Run the complete pipeline
python main.py --phase all

# Or run individual phases
python main.py --phase setup      # Setup environment
python main.py --phase collect    # Data collection
python main.py --phase preprocess # Data preprocessing  
python main.py --phase model     # Train ML models
python main.py --phase analyze   # Generate analysis
```

## 📊 Data Collection

### Google Trends (Recommended - Free)
- Automatically collects fashion keyword trends
- No API key required
- Provides trend scores over time

### Pinterest (Requires API Token)
1. Visit [Pinterest Developers](https://developers.pinterest.com/)
2. Create an app and get access token
3. Add token to `src/data_collection/pinterest_collector.py`

### Fashion Blogs (Use Carefully)
- Respects robots.txt and rate limits
- Only use for educational/research purposes
- Check terms of service before scraping

## 🔧 Configuration

Edit `config/config.yaml` to customize:
- Keywords to track
- Data collection settings
- Model parameters
- Analysis options

## 📈 Understanding Results

### Generated Files
- `outputs/fashion_wordcloud.png` - Visual representation of trending keywords
- `outputs/top_keywords.png` - Bar chart of most frequent fashion terms
- `outputs/color_trends.png` - Popular color analysis
- `outputs/fashion_dashboard.html` - Interactive dashboard
- `outputs/fashion_trend_report.md` - Comprehensive text report

### Model Files
- `models/random_forest_model.pkl` - Trained Random Forest model
- `models/tfidf_vectorizer.pkl` - Text vectorization model
- `models/trend_labels_encoder.pkl` - Label encoding model

## 🎨 Customization

### Adding New Data Sources
1. Create collector in `src/data_collection/`
2. Add preprocessing logic in `src/preprocessing/`
3. Update main pipeline in `main.py`

### Custom Keywords
Edit `config/config.yaml` to add industry-specific terms:
```yaml
FASHION_KEYWORDS:
  custom_category:
    - "new keyword 1"
    - "new keyword 2"
```

### New Analysis Types
Add functions to `src/analysis/trend_analyzer.py`:
```python
def analyze_custom_metric(self, datasets):
    # Your custom analysis logic
    pass
```

## 🤖 Machine Learning

### Available Models
- **Random Forest** - Best for interpretability
- **Gradient Boosting** - Often highest accuracy
- **SVM** - Good for complex patterns
- **Naive Bayes** - Fast and simple baseline

### Model Performance
Check `outputs/model_comparison.png` for accuracy comparison.

### Making Predictions
```python
from src.modeling.trend_predictor import FashionTrendPredictor

predictor = FashionTrendPredictor()
predictor.load_models("models/")

# Predict on new data
predictions = predictor.predict_trends(new_data, model_name='random_forest')
```

## 📋 Project Structure

```
fashion-trend-prediction/
├── data/
│   ├── raw/           # Original collected data
│   ├── processed/     # Cleaned and prepared data
│   └── external/      # External datasets
├── src/
│   ├── data_collection/   # Data gathering scripts
│   ├── preprocessing/     # Data cleaning and preparation
│   ├── modeling/         # Machine learning models
│   ├── analysis/         # Analysis and visualization
│   └── utils/           # Utility functions
├── models/           # Trained model files
├── outputs/          # Results and visualizations
├── config/           # Configuration files
├── notebooks/        # Jupyter notebooks (optional)
├── main.py          # Main execution script
└── requirements.txt # Python dependencies
```

## ⚠️ Important Notes

### Data Privacy
- Only uses publicly available data
- Follows GDPR and privacy guidelines
- No personal information collected

### Rate Limiting
- Respects API rate limits
- Includes delays between requests
- Follows robots.txt guidelines

### Ethical Usage
- Designed for educational/research purposes
- Promotes inclusive and sustainable fashion
- Monitors for algorithmic bias

## 🐛 Troubleshooting

### Common Issues

**Import errors:**
```bash
# Make sure you're in the project directory
cd fashion-trend-prediction
python main.py
```

**Missing dependencies:**
```bash
pip install -r requirements.txt
```

**NLTK data missing:**
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords') 
nltk.download('wordnet')
```

**No data collected:**
- Check internet connection
- Verify API tokens (for Pinterest)
- Check rate limiting delays

### Getting Help
1. Check the error message in terminal
2. Verify all dependencies are installed
3. Ensure you have write permissions in project directory
4. Check configuration in `config/config.yaml`

## 📚 Further Development

### Advanced Features
- Real-time trend monitoring
- Mobile app integration
- Advanced deep learning models
- Multi-language support

### Research Extensions
- Different age demographics
- Male fashion trends
- Seasonal pattern analysis
- Regional trend variations

## 📞 Support

For issues or questions:
1. Check this user guide
2. Review error messages carefully
3. Verify configuration settings
4. Ensure proper data permissions

Remember: This system is designed for educational and research purposes. Always respect data privacy, platform terms of service, and ethical guidelines when collecting and analyzing fashion data.
