import {
    BarChart as BarChartIcon,
    Dashboard as DashboardIcon,
    Psychology as InsightsIcon
} from '@mui/icons-material';
import {
    Box,
    Drawer,
    List,
    ListItem,
    ListItemIcon,
    ListItemText,
    Toolbar,
    Typography
} from '@mui/material';
import { Link, useLocation } from 'react-router-dom';

const drawerWidth = 240;

const Sidebar = () => {
  const location = useLocation();

  const menuItems = [
    { text: 'Dashboard Home', icon: <DashboardIcon />, path: '/' },
    { text: 'Visualizations', icon: <BarChartIcon />, path: '/visualizations' },
    { text: 'Model Results', icon: <InsightsIcon />, path: '/model-results' }
  ];

  return (
    <Drawer
      variant="permanent"
      sx={{
        width: drawerWidth,
        flexShrink: 0,
        [`& .MuiDrawer-paper`]: {
          width: drawerWidth,
          boxSizing: 'border-box',
          backgroundColor: '#f5f5f5'
        },
      }}
    >
      <Toolbar>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <DashboardIcon color="primary" />
          <Typography variant="h6" noWrap component="div" color="primary">
            Fashion AI
          </Typography>
        </Box>
      </Toolbar>
      
      <List>
        {menuItems.map((item) => (
          <ListItem
            key={item.text}
            component={Link}
            to={item.path}
            sx={{
              color: 'inherit',
              textDecoration: 'none',
              backgroundColor: location.pathname === item.path ? 'primary.light' : 'inherit',
              '&:hover': {
                backgroundColor: 'primary.light',
                color: 'white'
              }
            }}
          >
            <ListItemIcon sx={{ color: 'inherit' }}>
              {item.icon}
            </ListItemIcon>
            <ListItemText primary={item.text} />
          </ListItem>
        ))}
      </List>
    </Drawer>
  );
};

export default Sidebar;
