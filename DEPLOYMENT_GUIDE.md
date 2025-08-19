# Lassa Fever Decision Support System - Deployment Guide

## Render.com Deployment

### Prerequisites
- GitHub repository with your code
- Render.com account

### Step-by-Step Deployment

1. **Push Code to GitHub**
   ```bash
   git add .
   git commit -m "feat: production-ready deployment with enhanced UI and improved model accuracy"
   git push origin main
   ```

2. **Create New Web Service on Render**
   - Go to [Render.com](https://render.com)
   - Click "New" → "Web Service"
   - Connect your GitHub repository

3. **Configure Service Settings**
   - **Name**: `lassa-fever-diagnosis`
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT project.app_enhanced:app`
   - **Instance Type**: `Free` (or `Starter` for better performance)

4. **Environment Variables**
   Add these environment variables in Render dashboard:
   ```
   PYTHON_VERSION=3.10.12
   FLASK_ENV=production
   FLASK_APP=project.app_enhanced.py
   SECRET_KEY=your-secure-secret-key-here
   ```

5. **Deploy**
   - Click "Create Web Service"
   - Wait for build and deployment (5-10 minutes)

### File Structure for Deployment
```
project/
├── requirements.txt          # Python dependencies
├── runtime.txt              # Python version
├── render.yaml              # Render configuration
├── project/
│   ├── app_enhanced.py      # Main Flask application
│   ├── models.py            # ML models
│   ├── database.py          # Database management
│   ├── train_balanced.py    # Improved training script
│   ├── templates/           # HTML templates
│   └── uploads/             # File uploads
└── README.md
```

### Performance Optimizations

1. **Model Loading**
   - Models are loaded once at startup
   - Cached in memory for fast predictions

2. **Database**
   - SQLite for development
   - Consider PostgreSQL for production

3. **Static Files**
   - CSS/JS served via CDN
   - Images optimized for web

### Monitoring and Maintenance

1. **Health Checks**
   - Render automatically monitors service health
   - Check logs in Render dashboard

2. **Updates**
   - Push to GitHub triggers automatic redeployment
   - Zero-downtime deployments

3. **Scaling**
   - Upgrade to paid plan for auto-scaling
   - Monitor resource usage

### Troubleshooting

**Common Issues:**

1. **Build Fails**
   - Check `requirements.txt` for correct versions
   - Ensure Python version compatibility

2. **App Won't Start**
   - Verify start command: `gunicorn --bind 0.0.0.0:$PORT project.app_enhanced:app`
   - Check environment variables

3. **Model Loading Errors**
   - Ensure model files are included in repository
   - Check file paths are relative

4. **Memory Issues**
   - Upgrade to Starter plan ($7/month)
   - Optimize model size

### Security Considerations

1. **Environment Variables**
   - Never commit secrets to repository
   - Use Render's environment variables

2. **File Uploads**
   - Validate file types and sizes
   - Scan uploaded files

3. **HTTPS**
   - Render provides free SSL certificates
   - All traffic encrypted by default

### Cost Estimation

- **Free Tier**: $0/month (limited resources, sleeps after 15min inactivity)
- **Starter**: $7/month (always on, better performance)
- **Standard**: $25/month (high performance, auto-scaling)

### Alternative Deployment Options

1. **Heroku**
   - Similar process to Render
   - Use `Procfile` instead of render.yaml

2. **Railway**
   - Automatic deployment from GitHub
   - Built-in database options

3. **DigitalOcean App Platform**
   - Competitive pricing
   - Integrated with DO ecosystem

### Post-Deployment Testing

1. **Functionality Tests**
   - Test single patient prediction
   - Test batch processing
   - Verify all navigation links

2. **Performance Tests**
   - Check response times
   - Test with multiple concurrent users

3. **Mobile Testing**
   - Verify responsive design
   - Test on different devices

### Maintenance Schedule

- **Weekly**: Check logs and performance metrics
- **Monthly**: Update dependencies if needed
- **Quarterly**: Review and optimize model performance

For support, check Render's documentation or contact their support team.
