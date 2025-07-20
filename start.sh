#!/bin/bash

echo "🚀 Starting CLV Analysis Dashboard..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create necessary directories
mkdir -p api_results clv_results_kaggle logs dashboard_results

# Build and start the services
echo "🔨 Building Docker images..."
docker-compose build

echo "🎯 Starting Streamlit Dashboard..."
docker-compose up -d clv-dashboard

# Wait for the dashboard to be ready
echo "⏳ Waiting for Dashboard to be ready..."
for i in {1..30}; do
    if curl -s http://localhost:8501/_stcore/health > /dev/null 2>&1; then
        echo "✅ CLV Analysis Dashboard is ready!"
        echo ""
        echo "📊 Dashboard URL: http://localhost:8501"
        echo ""
        echo "🎯 Features:"
        echo "  • Upload CSV files or use sample dataset"
        echo "  • Interactive RFM analysis"
        echo "  • Customer lifetime value calculations"
        echo "  • Customer segmentation insights"
        echo "  • Executive summary and recommendations"
        echo "  • Download analysis results"
        echo ""
        echo "🛑 To stop: docker-compose down"
        echo "📋 View logs: docker-compose logs -f clv-dashboard"
        echo ""
        echo "🔧 To start API instead: docker-compose up -d --profile api"
        exit 0
    fi
    sleep 2
done

echo "❌ Dashboard failed to start within 60 seconds"
echo "📋 Check logs with: docker-compose logs clv-dashboard"
exit 1