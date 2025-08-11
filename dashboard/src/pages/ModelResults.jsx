import { Analytics, Psychology, TrendingUp } from '@mui/icons-material';
import {
  Alert,
  Box,
  Button,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Grid,
  LinearProgress,
  MenuItem,
  Paper,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  TextField,
  Typography
} from '@mui/material';
import { DatePicker } from '@mui/x-date-pickers/DatePicker';
import axios from 'axios';
import dayjs from 'dayjs';
import { useEffect, useState } from 'react';

const ModelResults = () => {
  const [predictions, setPredictions] = useState(null);
  const [categories, setCategories] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  
  // Prediction form state
  const [selectedDate, setSelectedDate] = useState(dayjs());
  const [selectedCategory, setSelectedCategory] = useState('');
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionLoading, setPredictionLoading] = useState(false);
  const [predictionError, setPredictionError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('🔄 Fetching model predictions and categories...');
        const [predictionsRes, categoriesRes] = await Promise.all([
          axios.get('http://localhost:5000/api/predictions'),
          axios.get('http://localhost:5000/api/categories')
        ]);
        
        console.log('✅ Predictions response:', predictionsRes.data);
        console.log('✅ Categories response:', categoriesRes.data);
        
        setPredictions(predictionsRes.data);
        setCategories(categoriesRes.data.categories || []);
        
        // Set default category
        if (categoriesRes.data.categories && categoriesRes.data.categories.length > 0) {
          setSelectedCategory(categoriesRes.data.categories[0]);
        }
        
        setLoading(false);
      } catch (err) {
        console.error('❌ Error fetching data:', err);
        setError('Failed to load model predictions. Make sure your API server is running.');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  // Handle category change
  const handleCategoryChange = (category) => {
    setSelectedCategory(category);
  };

  const handlePredict = async () => {
    if (!selectedDate || !selectedCategory) {
      setPredictionError('Please select both date and category');
      return;
    }

    setPredictionLoading(true);
    setPredictionError(null);

    try {
      const response = await axios.post('http://localhost:5000/api/predict', {
        date: selectedDate.format('YYYY-MM-DD'),
        category: selectedCategory
      });
      
      setPredictionResult(response.data);
    } catch (err) {
      setPredictionError(err.response?.data?.error || 'Failed to make prediction');
    } finally {
      setPredictionLoading(false);
    }
  };

  if (loading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;

  const getScoreColor = (score) => {
    if (score >= 0.8) return 'success';
    if (score >= 0.6) return 'warning';
    return 'error';
  };

  const getTrendIcon = (direction) => {
    return direction === 'up' ? '📈' : '📉';
  };

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        <Psychology sx={{ mr: 2, verticalAlign: 'middle' }} />
        Fashion Trend Prediction Model
      </Typography>

        <Grid container spacing={3}>
          {/* Interactive Prediction Tool */}
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3, background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
              <Typography variant="h5" gutterBottom>
                <Analytics sx={{ mr: 1, verticalAlign: 'middle' }} />
                Make a Prediction
              </Typography>
              <Typography variant="body1" sx={{ mb: 3, opacity: 0.9 }}>
                Select a date and category to predict trend scores using our advanced model
              </Typography>
              
              <Grid container spacing={2} alignItems="center">
                <Grid item xs={12} md={4}>
                  <DatePicker
                    label="Select Date"
                    value={selectedDate}
                    onChange={(newValue) => setSelectedDate(newValue)}
                    slotProps={{
                      textField: {
                        fullWidth: true,
                        sx: { 
                          '& .MuiOutlinedInput-root': { 
                            backgroundColor: 'rgba(255,255,255,0.1)',
                            '& fieldset': { borderColor: 'rgba(255,255,255,0.3)' },
                            '&:hover fieldset': { borderColor: 'rgba(255,255,255,0.5)' },
                            '&.Mui-focused fieldset': { borderColor: 'white' }
                          },
                          '& .MuiInputLabel-root': { color: 'rgba(255,255,255,0.7)' },
                          '& .MuiInputBase-input': { color: 'white' }
                        }
                      }
                    }}
                  />
                </Grid>
                <Grid item xs={12} md={4}>
                  <TextField
                    select
                    label="Category"
                    value={selectedCategory}
                    onChange={(e) => handleCategoryChange(e.target.value)}
                    fullWidth
                    sx={{ 
                      '& .MuiOutlinedInput-root': { 
                        backgroundColor: 'rgba(255,255,255,0.1)',
                        '& fieldset': { borderColor: 'rgba(255,255,255,0.3)' },
                        '&:hover fieldset': { borderColor: 'rgba(255,255,255,0.5)' },
                        '&.Mui-focused fieldset': { borderColor: 'white' }
                      },
                      '& .MuiInputLabel-root': { color: 'rgba(255,255,255,0.7)' },
                      '& .MuiInputBase-input': { color: 'white' }
                    }}
                  >
                    {categories.map((category) => (
                      <MenuItem key={category} value={category}>
                        {category}
                      </MenuItem>
                    ))}
                  </TextField>
                </Grid>
                <Grid item xs={12} md={4}>
                  <Button
                    variant="contained"
                    onClick={handlePredict}
                    disabled={predictionLoading || !selectedDate || !selectedCategory}
                    fullWidth
                    size="large"
                    startIcon={predictionLoading ? <CircularProgress size={20} color="inherit" /> : <TrendingUp />}
                    sx={{ 
                      background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)',
                      boxShadow: '0 3px 5px 2px rgba(255, 105, 135, .3)',
                      color: 'white',
                      fontWeight: 'bold',
                      height: 56,
                      '&:hover': {
                        background: 'linear-gradient(45deg, #FE6B8B 60%, #FF8E53 100%)',
                        boxShadow: '0 6px 10px 4px rgba(255, 105, 135, .3)',
                      },
                      '&:disabled': {
                        background: 'rgba(255,255,255,0.2)',
                        color: 'rgba(255,255,255,0.5)'
                      }
                    }}
                  >
                    {predictionLoading ? 'Predicting...' : 'PREDICT TREND'}
                  </Button>
                </Grid>
              </Grid>

              {predictionError && (
                <Alert severity="error" sx={{ mt: 2 }}>
                  {predictionError}
                </Alert>
              )}

              {predictionResult && (
                <Box sx={{ mt: 3, p: 2, bgcolor: 'rgba(255,255,255,0.1)', borderRadius: 2 }}>
                  <Typography variant="h6" gutterBottom>
                    Prediction Result {getTrendIcon(predictionResult.prediction.trend_direction)}
                  </Typography>
                  <Grid container spacing={2}>
                    <Grid item xs={12} md={3}>
                      <Typography variant="body2" sx={{ opacity: 0.8 }}>Predicted Score</Typography>
                      <Typography variant="h4" color="white">
                        {predictionResult.prediction.predicted_score}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Typography variant="body2" sx={{ opacity: 0.8 }}>Confidence</Typography>
                      <Typography variant="h4" color="white">
                        {(predictionResult.prediction.confidence * 100).toFixed(1)}%
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Typography variant="body2" sx={{ opacity: 0.8 }}>Trend Direction</Typography>
                      <Typography variant="h4" color="white">
                        {predictionResult.prediction.trend_direction.toUpperCase()}
                      </Typography>
                    </Grid>
                    <Grid item xs={12} md={3}>
                      <Typography variant="body2" sx={{ opacity: 0.8 }}>Trend Strength</Typography>
                      <Typography variant="h4" color="white">
                        {predictionResult.interpretation.trend_strength}
                      </Typography>
                    </Grid>
                  </Grid>
                </Box>
              )}
            </Paper>
          </Grid>

          {/* Top Trending Items Table */}
          {predictionResult?.ai_insights?.top_trending_items && (
            <Grid item xs={12}>
              <Paper elevation={3} sx={{ p: 3, bgcolor: 'white', color: 'black', border: '2px solid #e0e0e0' }}>
                <Typography variant="h5" gutterBottom sx={{ color: 'black', fontWeight: 'bold' }}>
                  📈 Top 10 Trending Items
                </Typography>
                <Typography variant="body1" sx={{ mb: 3, color: '#666' }}>
                  {selectedCategory} trends for {selectedDate?.format('MMM DD, YYYY')}
                </Typography>
                
                <TableContainer sx={{ bgcolor: 'white', borderRadius: 2, border: '1px solid #e0e0e0' }}>
                  <Table>
                    <TableHead>
                      <TableRow sx={{ bgcolor: '#f5f5f5' }}>
                        <TableCell sx={{ color: 'black', fontWeight: 'bold', borderBottom: '2px solid #e0e0e0' }}>#</TableCell>
                        <TableCell sx={{ color: 'black', fontWeight: 'bold', borderBottom: '2px solid #e0e0e0' }}>Trending Item</TableCell>
                        <TableCell sx={{ color: 'black', fontWeight: 'bold', borderBottom: '2px solid #e0e0e0' }}>Trend Score</TableCell>
                        <TableCell sx={{ color: 'black', fontWeight: 'bold', borderBottom: '2px solid #e0e0e0' }}>Direction</TableCell>
                        <TableCell sx={{ color: 'black', fontWeight: 'bold', borderBottom: '2px solid #e0e0e0' }}>Why Trending</TableCell>
                        <TableCell sx={{ color: 'black', fontWeight: 'bold', borderBottom: '2px solid #e0e0e0' }}>Target Audience</TableCell>
                      </TableRow>
                    </TableHead>
                    <TableBody>
                      {predictionResult.ai_insights.top_trending_items.map((item, index) => (
                        <TableRow key={index} hover sx={{ 
                          '&:hover': { bgcolor: '#f9f9f9' },
                          '&:nth-of-type(even)': { bgcolor: '#fafafa' },
                          borderBottom: '1px solid #e0e0e0'
                        }}>
                          <TableCell sx={{ color: 'black', borderBottom: '1px solid #e0e0e0' }}>
                            <Typography variant="h6" sx={{ color: '#d4af37', fontWeight: 'bold' }}>
                              #{index + 1}
                            </Typography>
                          </TableCell>
                          <TableCell sx={{ color: 'black', borderBottom: '1px solid #e0e0e0' }}>
                            <Typography variant="subtitle1" fontWeight="medium" sx={{ color: 'black' }}>
                              {item.item_name}
                            </Typography>
                          </TableCell>
                          <TableCell sx={{ color: 'black', borderBottom: '1px solid #e0e0e0' }}>
                            <Box display="flex" alignItems="center">
                              <Typography variant="h6" sx={{ mr: 1, color: '#2e7d32', fontWeight: 'bold' }}>
                                {(item.trend_score * 100).toFixed(0)}%
                              </Typography>
                              <LinearProgress 
                                variant="determinate" 
                                value={item.trend_score * 100} 
                                sx={{ 
                                  width: 60, 
                                  height: 6, 
                                  borderRadius: 3,
                                  bgcolor: '#e0e0e0',
                                  '& .MuiLinearProgress-bar': { bgcolor: '#2e7d32' }
                                }}
                              />
                            </Box>
                          </TableCell>
                          <TableCell sx={{ color: 'black', borderBottom: '1px solid #e0e0e0' }}>
                            <Chip 
                              label={item.trend_direction}
                              color={item.trend_direction === 'up' ? 'success' : 'default'}
                              variant="filled"
                              size="small"
                              icon={<span>{getTrendIcon(item.trend_direction)}</span>}
                              sx={{ 
                                color: 'white',
                                bgcolor: item.trend_direction === 'up' ? '#2e7d32' : '#757575',
                                '& .MuiChip-icon': { color: 'white' }
                              }}
                            />
                          </TableCell>
                          <TableCell sx={{ color: 'black', borderBottom: '1px solid #e0e0e0' }}>
                            <Typography variant="body2" sx={{ maxWidth: 200, color: '#333' }}>
                              {item.popularity_reason}
                            </Typography>
                          </TableCell>
                          <TableCell sx={{ color: 'black', borderBottom: '1px solid #e0e0e0' }}>
                            <Typography variant="body2" sx={{ color: '#333' }}>
                              {item.target_demographic}
                            </Typography>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </TableContainer>

                {/* Category Insights Summary */}
                {predictionResult?.ai_insights?.category_insights && (
                  <Box sx={{ mt: 3, p: 2, bgcolor: '#f8f9fa', borderRadius: 2, border: '1px solid #e0e0e0' }}>
                    <Typography variant="h6" gutterBottom sx={{ color: 'black', fontWeight: 'bold' }}>
                      📊 Category Insights
                    </Typography>
                    <Grid container spacing={2}>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" sx={{ color: '#666', fontWeight: 'medium' }}>Overall Trend</Typography>
                        <Typography variant="body1" sx={{ color: 'black' }}>
                          {predictionResult.ai_insights.category_insights.overall_trend}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" sx={{ color: '#666', fontWeight: 'medium' }}>Market Opportunities</Typography>
                        <Typography variant="body1" sx={{ color: 'black' }}>
                          {predictionResult.ai_insights.category_insights.market_opportunities}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" sx={{ color: '#666', fontWeight: 'medium' }}>Seasonal Factors</Typography>
                        <Typography variant="body1" sx={{ color: 'black' }}>
                          {predictionResult.ai_insights.category_insights.seasonal_factors}
                        </Typography>
                      </Grid>
                      <Grid item xs={12} md={6}>
                        <Typography variant="body2" sx={{ color: '#666', fontWeight: 'medium' }}>Price Trends</Typography>
                        <Typography variant="body1" sx={{ color: 'black' }}>
                          {predictionResult.ai_insights.category_insights.price_trends}
                        </Typography>
                      </Grid>
                    </Grid>
                  </Box>
                )}
              </Paper>
            </Grid>
          )}

          {/* Model Performance Metrics */}
          <Grid item xs={12} md={3}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">
                  Accuracy
                </Typography>
                <Typography variant="h3" color="success.main">
                  {(predictions?.model_performance?.accuracy_percentage || 0).toFixed(1)}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={predictions?.model_performance?.accuracy_percentage || 0} 
                  sx={{ mt: 2, height: 8, borderRadius: 4 }}
                />
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                  XGBoost Classification
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">
                  Precision
                </Typography>
                <Typography variant="h3" color="info.main">
                  {(predictions?.model_performance?.precision_percentage || 0).toFixed(1)}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={predictions?.model_performance?.precision_percentage || 0} 
                  sx={{ mt: 2, height: 8, borderRadius: 4 }}
                  color="info"
                />
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                  Precision Score
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">
                  Recall
                </Typography>
                <Typography variant="h3" color="warning.main">
                  {(predictions?.model_performance?.recall_percentage || 0).toFixed(1)}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={predictions?.model_performance?.recall_percentage || 0} 
                  sx={{ mt: 2, height: 8, borderRadius: 4 }}
                  color="warning"
                />
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                  Recall Score
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          <Grid item xs={12} md={3}>
            <Card elevation={3}>
              <CardContent>
                <Typography variant="h6" gutterBottom color="primary">
                  F1 Score
                </Typography>
                <Typography variant="h3" color="secondary.main">
                  {(predictions?.model_performance?.f1_percentage || 0).toFixed(1)}%
                </Typography>
                <LinearProgress 
                  variant="determinate" 
                  value={predictions?.model_performance?.f1_percentage || 0} 
                  sx={{ mt: 2, height: 8, borderRadius: 4 }}
                  color="secondary"
                />
                <Typography variant="body2" color="textSecondary" sx={{ mt: 1 }}>
                  F1 Score
                </Typography>
              </CardContent>
            </Card>
          </Grid>

          {/* Recent Predictions */}
          <Grid item xs={12}>
            <Paper elevation={2} sx={{ p: 3 }}>
              <Typography variant="h5" gutterBottom>
                <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
                Recent Category Predictions
              </Typography>
              <Typography variant="body2" color="textSecondary" sx={{ mb: 2 }}>
                AI predictions for top fashion categories today
              </Typography>
              
              <TableContainer>
                <Table>
                  <TableHead>
                    <TableRow>
                      <TableCell><strong>Category</strong></TableCell>
                      <TableCell><strong>Predicted Score</strong></TableCell>
                      <TableCell><strong>Trend</strong></TableCell>
                      <TableCell><strong>Strength</strong></TableCell>
                    </TableRow>
                  </TableHead>
                  <TableBody>
                    {predictions?.recent_predictions?.map((prediction, index) => (
                      <TableRow key={index} hover>
                        <TableCell>
                          <Typography variant="subtitle1" fontWeight="medium">
                            {prediction.category}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Typography variant="h6" color="primary">
                            {prediction.predicted_score}
                          </Typography>
                        </TableCell>
                        <TableCell>
                          <Chip 
                            label={prediction.trend}
                            color={prediction.trend === 'up' ? 'success' : prediction.trend === 'stable' ? 'warning' : 'error'}
                            variant="filled"
                            icon={prediction.trend === 'up' ? <span>📈</span> : prediction.trend === 'stable' ? <span>➡️</span> : <span>📉</span>}
                          />
                        </TableCell>
                        <TableCell>
                          <Typography variant="body2">
                            {prediction.strength.toFixed(2)}
                          </Typography>
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              </TableContainer>
            </Paper>
          </Grid>

          {/* Model Details */}
          <Grid item xs={12}>
            <Paper elevation={2} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Model Information
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="textSecondary">Algorithm</Typography>
                  <Typography variant="body1">{predictions?.model_info?.type || 'Random Forest'}</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="textSecondary">Features</Typography>
                  <Typography variant="body1">{predictions?.model_info?.features?.join(', ') || 'Date, Category'}</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="textSecondary">Training Samples</Typography>
                  <Typography variant="body1">{predictions?.model_info?.training_samples || 'N/A'}</Typography>
                </Grid>
                <Grid item xs={12} sm={6} md={3}>
                  <Typography variant="body2" color="textSecondary">Last Updated</Typography>
                  <Typography variant="body1">{predictions?.prediction_date || 'Today'}</Typography>
                </Grid>
              </Grid>
            </Paper>
          </Grid>
        </Grid>
      </Box>
    );
  };

export default ModelResults;
