# 🚀 Lassa Fever Diagnosis System - Live Deployment Guide

## Overview
This guide provides step-by-step instructions for deploying your Lassa fever diagnosis system to production environments.

## ⚠️ Important Note About Model Performance
**Current Model Status**: Your model has 19.26% accuracy with high recall (93.63%) but low precision (14.52%). While functional, consider improving the model before clinical deployment.

## 🎯 Deployment Options

### Option 1: Heroku Deployment (Recommended for Testing)

#### Prerequisites
- Heroku account (free tier available)
- Git installed
- Heroku CLI installed

#### Steps

1. **Prepare your application**
   ```bash
   cd "c:\Users\akint\OneDrive\Documents\final year project code\project"
   
   # Initialize git repository if not already done
   git init
   git add .
   git commit -m "Initial commit for deployment"
   ```

2. **Install Heroku CLI**
   - Download from: https://devcenter.heroku.com/articles/heroku-cli
   - Login: `heroku login`

3. **Create Heroku app**
   ```bash
   heroku create your-lassa-fever-app-name
   ```

4. **Set environment variables**
   ```bash
   heroku config:set FLASK_ENV=production
   heroku config:set SECRET_KEY=your-super-secret-key-here
   ```

5. **Deploy to Heroku**
   ```bash
   git push heroku main
   ```

6. **Open your app**
   ```bash
   heroku open
   ```

#### Heroku Considerations
- **Free tier limitations**: App sleeps after 30 minutes of inactivity
- **Memory limit**: 512MB RAM (sufficient for your model)
- **Build time**: May take 5-10 minutes due to PyTorch installation

### Option 2: Railway Deployment (Alternative Cloud)

1. **Sign up at Railway.app**
2. **Connect your GitHub repository**
3. **Railway will auto-detect and deploy your Flask app**
4. **Set environment variables in Railway dashboard**

### Option 3: Docker Deployment (Self-hosted)

#### Prerequisites
- Docker installed
- Server with at least 2GB RAM

#### Steps

1. **Build Docker image**
   ```bash
   cd "c:\Users\akint\OneDrive\Documents\final year project code\project"
   docker build -t lassa-fever-diagnosis .
   ```

2. **Run container**
   ```bash
   docker run -p 5000:5000 -e FLASK_ENV=production lassa-fever-diagnosis
   ```

3. **Access application**
   - Open browser to `http://localhost:5000`

### Option 4: AWS EC2 Deployment (Production)

#### Prerequisites
- AWS account
- EC2 instance (t2.medium recommended)
- Domain name (optional)

#### Steps

1. **Launch EC2 instance**
   - Choose Ubuntu 20.04 LTS
   - Instance type: t2.medium (2GB RAM)
   - Configure security group (port 80, 443, 22)

2. **Connect to instance and setup**
   ```bash
   # Update system
   sudo apt update && sudo apt upgrade -y
   
   # Install Python and dependencies
   sudo apt install python3-pip nginx git -y
   
   # Clone your repository
   git clone https://github.com/yourusername/lassa-fever-diagnosis.git
   cd lassa-fever-diagnosis
   
   # Install dependencies
   pip3 install -r requirements_production.txt
   
   # Install Gunicorn
   pip3 install gunicorn
   ```

3. **Configure Nginx**
   ```bash
   sudo nano /etc/nginx/sites-available/lassa-fever
   ```
   
   Add configuration:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://127.0.0.1:5000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

4. **Enable site and start services**
   ```bash
   sudo ln -s /etc/nginx/sites-available/lassa-fever /etc/nginx/sites-enabled
   sudo systemctl restart nginx
   
   # Start your app with Gunicorn
   gunicorn app_production:app --bind 127.0.0.1:5000 --workers 2
   ```

## 🔧 Production Considerations

### Security
- **Change SECRET_KEY**: Use a strong, unique secret key
- **HTTPS**: Enable SSL/TLS certificates (Let's Encrypt recommended)
- **Input validation**: All user inputs are validated
- **Rate limiting**: Consider adding rate limiting for API endpoints

### Performance
- **Model optimization**: Current model may be slow for large-scale use
- **Caching**: Consider Redis for caching predictions
- **Load balancing**: Use multiple workers for high traffic

### Monitoring
- **Health checks**: `/health` endpoint available for monitoring
- **Logging**: Application logs to stdout (captured by most platforms)
- **Error tracking**: Consider Sentry for error monitoring

### Model Updates
- **Continuous deployment**: Set up CI/CD pipeline for model updates
- **A/B testing**: Test new models alongside current version
- **Rollback strategy**: Keep previous model versions for quick rollback

## 🏥 Clinical Deployment Considerations

### Regulatory Compliance
- **FDA/CE marking**: May be required for clinical use
- **HIPAA compliance**: Ensure patient data protection
- **Audit trails**: Log all predictions for regulatory review

### Integration
- **EMR integration**: Connect with hospital information systems
- **HL7 FHIR**: Use healthcare data standards
- **API documentation**: Provide comprehensive API docs for integration

### Quality Assurance
- **Model validation**: Regular performance monitoring
- **Clinical validation**: Validate against clinical outcomes
- **Bias testing**: Test for demographic biases

## 📊 Current Model Performance Warning

**⚠️ IMPORTANT**: Your current model has the following performance characteristics:
- **Accuracy**: 19.26% (very low)
- **Sensitivity**: 93.63% (good - catches most Lassa cases)
- **Specificity**: ~14.52% (poor - many false positives)

**Recommendations before clinical deployment**:
1. **Improve model accuracy** using the enhanced training scripts provided
2. **Validate on independent test set**
3. **Conduct clinical validation study**
4. **Implement proper threshold tuning for clinical decision-making**

## 🚀 Quick Start Commands

### Local Testing
```bash
# Test production app locally
python app_production.py
```

### Heroku Deployment
```bash
# One-command deployment
git add . && git commit -m "Deploy" && git push heroku main
```

### Docker Deployment
```bash
# Build and run
docker build -t lassa-fever . && docker run -p 5000:5000 lassa-fever
```

## 📞 Support and Troubleshooting

### Common Issues
1. **Model not loading**: Ensure `model.pth` and `preproc.pkl` are in the deployment
2. **Memory errors**: Use CPU-only deployment, increase server RAM
3. **Slow predictions**: Consider model optimization or GPU acceleration

### Logs and Debugging
- Check application logs for errors
- Use `/health` endpoint to verify system status
- Monitor memory usage during predictions

## 🎯 Next Steps After Deployment

1. **Monitor performance**: Track prediction accuracy and response times
2. **Collect feedback**: Gather user feedback for improvements
3. **Model improvements**: Implement the enhanced training pipeline
4. **Scale gradually**: Start with limited users, scale based on performance
5. **Clinical validation**: Conduct proper clinical studies before medical use

---

**Remember**: This system is currently a research prototype. Ensure proper validation and regulatory approval before clinical deployment.
