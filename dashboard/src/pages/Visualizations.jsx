import { Analytics, Category, Timeline, TrendingUp } from '@mui/icons-material';
import {
  Alert,
  Box,
  Card,
  CardContent,
  CardMedia,
  CircularProgress,
  Grid,
  Paper,
  Typography
} from '@mui/material';
import axios from 'axios';
import { useEffect, useState } from 'react';
import { Bar, BarChart, CartesianGrid, Cell, Legend, Line, LineChart, Pie, PieChart, Tooltip, XAxis, YAxis } from 'recharts';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884D8', '#82CA9D'];

const Visualizations = () => {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await axios.get('http://localhost:5000/api/visualizations');
        setData(response.data);
        setLoading(false);
      } catch (err) {
        setError('Failed to load visualization data. Make sure your API server is running.');
        setLoading(false);
      }
    };

    fetchData();
  }, []);

  if (loading) return <CircularProgress />;
  if (error) return <Alert severity="error">{error}</Alert>;

  return (
    <Box>
      <Typography variant="h4" gutterBottom>
        <Analytics sx={{ mr: 2, verticalAlign: 'middle' }} />
        Fashion Trend Visualizations
      </Typography>

      <Grid container spacing={3}>
   
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              <Category sx={{ mr: 1, verticalAlign: 'middle' }} />
              Category Distribution
            </Typography>
            <BarChart width={500} height={300} data={data?.category_distribution || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="count" fill="#8884d8" />
            </BarChart>
          </Paper>
        </Grid>


        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              <TrendingUp sx={{ mr: 1, verticalAlign: 'middle' }} />
              Top Categories by Popularity
            </Typography>
            <BarChart width={500} height={300} data={data?.top_categories || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="category" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="avg_popularity" fill="#82ca9d" />
            </BarChart>
          </Paper>
        </Grid>

       
        <Grid item xs={12}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              <Timeline sx={{ mr: 1, verticalAlign: 'middle' }} />
              Monthly Popularity Trends
            </Typography>
            <LineChart width={1000} height={300} data={data?.monthly_trends || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="month" />
              <YAxis />
              <Tooltip />
              <Legend />
              <Line type="monotone" dataKey="avg_popularity" stroke="#8884d8" strokeWidth={2} />
            </LineChart>
          </Paper>
        </Grid>

        {/* Seasonal Trends */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Seasonal Trends
            </Typography>
            <PieChart width={400} height={300}>
              <Pie
                data={data?.seasonal_trends || []}
                cx={200}
                cy={150}
                labelLine={false}
                label={({ season, avg_popularity }) => `${season}: ${(avg_popularity * 100).toFixed(1)}%`}
                outerRadius={80}
                fill="#8884d8"
                dataKey="avg_popularity"
              >
                {(data?.seasonal_trends || []).map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </Paper>
        </Grid>

        {/* Weekly Trends */}
        <Grid item xs={12} md={6}>
          <Paper elevation={3} sx={{ p: 3 }}>
            <Typography variant="h6" gutterBottom>
              Weekly Trends (Recent 12 Weeks)
            </Typography>
            <LineChart width={400} height={300} data={data?.weekly_trends || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="week" angle={-45} textAnchor="end" height={60} />
              <YAxis />
              <Tooltip />
              <Line type="monotone" dataKey="avg_popularity" stroke="#ff7300" strokeWidth={2} />
            </LineChart>
          </Paper>
        </Grid>

        {/* Generated Images */}
        {data?.available_images && data.available_images.length > 0 && (
          <Grid item xs={12}>
            <Paper elevation={3} sx={{ p: 3 }}>
              <Typography variant="h6" gutterBottom>
                Generated Analysis Charts
              </Typography>
              <Grid container spacing={2}>
                {data.available_images.map((image, index) => (
                  <Grid item xs={12} md={4} key={index}>
                    <Card>
                      <CardMedia
                        component="img"
                        height="200"
                        image={`http://localhost:5000/api/images/${image}`}
                        alt={image}
                      />
                      <CardContent>
                        <Typography variant="body2" color="text.secondary">
                          {image.replace('.png', '').replace('_', ' ').toUpperCase()}
                        </Typography>
                      </CardContent>
                    </Card>
                  </Grid>
                ))}
              </Grid>
            </Paper>
          </Grid>
        )}
      </Grid>
    </Box>
  );
};

export default Visualizations;