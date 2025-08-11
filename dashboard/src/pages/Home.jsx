import { Category, DataUsage, Timeline, TrendingUp } from '@mui/icons-material';
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  CircularProgress,
  Grid,
  Paper,
  Typography
} from '@mui/material';
import axios from 'axios';
import { useEffect, useState } from 'react';

const Home = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        console.log('🔍 Attempting to fetch data from API...');
        const response = await axios.get('http://localhost:5000/api/data/summary');
        console.log('✅ API Response received:', response.data);
        setData(response.data);
        setLoading(false);
      } catch (err) {
        console.error('❌ API Error:', err);
        console.error('❌ Error details:', {
          message: err.message,
          code: err.code,
          response: err.response
        });
        
        let errorMessage = 'Failed to connect to backend. ';
        if (err.code === 'ECONNREFUSED') {
          errorMessage += 'Connection refused - API server not running on port 5000.';
        } else if (err.response) {
          errorMessage += `Server responded with ${err.response.status}: ${err.response.statusText}`;
        } else if (err.request) {
          errorMessage += 'No response received from server.';
        } else {
          errorMessage += err.message;
        }
        
        setError(errorMessage);
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Typography variant="h3" gutterBottom color="primary">
        Fashion Trend Analysis Dashboard
      </Typography>
      
      <Typography variant="h6" gutterBottom color="textSecondary">
        AI-Based Fashion Trend Prediction System - Real Data Analytics
      </Typography>

      <Grid container spacing={3} sx={{ mt: 2 }}>
        <Grid item xs={12} md={3}>
          <Card elevation={3} sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
            <CardContent>
              <Box display="flex" alignItems="center">
                <DataUsage sx={{ fontSize: 40, mr: 2 }} />
                <Box>
                  <Typography variant="h4">
                    {(data?.total_records || 0).toLocaleString()}
                  </Typography>
                  <Typography variant="body2">
                    Total Fashion Records
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <Card elevation={3} sx={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', color: 'white' }}>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Category sx={{ fontSize: 40, mr: 2 }} />
                <Box>
                  <Typography variant="h4">
                    {data?.categories || 0}
                  </Typography>
                  <Typography variant="body2">
                    Fashion Categories
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
        
        <Grid item xs={12} md={3}>
          <Card elevation={3} sx={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', color: 'white' }}>
            <CardContent>
              <Box display="flex" alignItems="center">
                <TrendingUp sx={{ fontSize: 40, mr: 2 }} />
                <Box>
                  <Typography variant="h4">
                    {data?.avg_popularity ? data.avg_popularity.toFixed(2) : '0.00'}
                  </Typography>
                  <Typography variant="body2">
                    Avg Popularity Score
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>

        <Grid item xs={12} md={3}>
          <Card elevation={3} sx={{ background: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)', color: 'white' }}>
            <CardContent>
              <Box display="flex" alignItems="center">
                <Timeline sx={{ fontSize: 40, mr: 2 }} />
                <Box>
                  <Typography variant="h4">
                    {data?.date_range ? `${new Date(data.date_range.start).getFullYear()}-${new Date(data.date_range.end).getFullYear()}` : 'N/A'}
                  </Typography>
                  <Typography variant="body2">
                    Data Time Range
                  </Typography>
                </Box>
              </Box>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      {/* Top Categories */}
      <Paper elevation={2} sx={{ p: 3, mt: 3 }}>
        <Typography variant="h5" gutterBottom>
          <Category sx={{ mr: 1, verticalAlign: 'middle' }} />
          Fashion Categories Overview
        </Typography>
        <Grid container spacing={2}>
          {data?.category_list?.slice(0, 8).map((category, index) => (
            <Grid item xs={12} sm={6} md={3} key={index}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="h6" gutterBottom>
                    {category}
                  </Typography>
                  <Chip 
                    label="Active Category" 
                    color="primary" 
                    variant="outlined"
                    sx={{ mt: 1 }}
                  />
                </CardContent>
              </Card>
            </Grid>
          ))}
        </Grid>
      </Paper>

      {/* Model Performance Metrics */}
      {data?.model_performance && (
        <Paper elevation={2} sx={{ p: 3, mt: 3 }}>
          <Typography variant="h5" gutterBottom>
            <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
            🏆 XGBoost Enhanced Model Performance (from notebook)
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined" sx={{ background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)', color: 'white' }}>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ opacity: 0.9 }}>
                    📊 Accuracy
                  </Typography>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {data.model_performance.accuracy.toFixed(2)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined" sx={{ background: 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)', color: 'white' }}>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ opacity: 0.9 }}>
                    🎯 Precision
                  </Typography>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {data.model_performance.precision.toFixed(2)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined" sx={{ background: 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)', color: 'white' }}>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ opacity: 0.9 }}>
                    📈 Recall
                  </Typography>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {data.model_performance.recall.toFixed(2)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={3}>
              <Card variant="outlined" sx={{ background: 'linear-gradient(135deg, #fa709a 0%, #fee140 100%)', color: 'white' }}>
                <CardContent>
                  <Typography variant="subtitle2" sx={{ opacity: 0.9 }}>
                    ⚖️ F1 Score
                  </Typography>
                  <Typography variant="h4" sx={{ fontWeight: 'bold' }}>
                    {data.model_performance.f1_score.toFixed(2)}%
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
          <Typography variant="body2" color="textSecondary" sx={{ mt: 2, textAlign: 'center' }}>
            🤖 Advanced XGBoost model with comprehensive feature engineering + Gemini AI
          </Typography>
        </Paper>
      )}

      {/* Data Statistics */}
      {data?.popularity_stats && (
        <Paper elevation={2} sx={{ p: 3, mt: 3 }}>
          <Typography variant="h5" gutterBottom>
            <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
            Popularity Score Statistics
          </Typography>
          <Grid container spacing={3}>
            <Grid item xs={12} sm={6} md={2.4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" color="textSecondary">
                    Mean Score
                  </Typography>
                  <Typography variant="h5" color="primary">
                    {data.popularity_stats.mean.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" color="textSecondary">
                    Median Score
                  </Typography>
                  <Typography variant="h5" color="primary">
                    {data.popularity_stats.median.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" color="textSecondary">
                    Standard Deviation
                  </Typography>
                  <Typography variant="h5" color="primary">
                    {data.popularity_stats.std.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" color="textSecondary">
                    Min Score
                  </Typography>
                  <Typography variant="h5" color="primary">
                    {data.popularity_stats.min.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
            <Grid item xs={12} sm={6} md={2.4}>
              <Card variant="outlined">
                <CardContent>
                  <Typography variant="subtitle2" color="textSecondary">
                    Max Score
                  </Typography>
                  <Typography variant="h5" color="primary">
                    {data.popularity_stats.max.toFixed(2)}
                  </Typography>
                </CardContent>
              </Card>
            </Grid>
          </Grid>
        </Paper>
      )}
    </Box>
  );
};

export default Home;
