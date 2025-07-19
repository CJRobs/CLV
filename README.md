# Customer Lifetime Value (CLV) Analysis System

A production-ready Python system for comprehensive Customer Lifetime Value analysis, including RFM segmentation, predictive modeling, and advanced lifetime value calculations.

## Features

- **RFM Analysis**: Recency, Frequency, Monetary segmentation
- **CLV Calculation**: Multiple methods including simple and advanced models
- **Predictive Modeling**: Random Forest for CLV prediction
- **Lifetime Models**: BG/NBD and Gamma-Gamma models
- **Customer Segmentation**: 8 distinct segments with actionable insights
- **Comprehensive Reporting**: Executive summaries, Excel reports, and KPI dashboards
- **Interactive Visualizations**: Static and interactive plots
- **Production Ready**: Error handling, logging, and modular design

## Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd clv-analysis

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

```python
from main import CLVAnalysisPipeline

# Initialize pipeline
pipeline = CLVAnalysisPipeline()

# Run complete analysis
results = pipeline.run_complete_analysis(
    data_path='your_data.csv',  # or df=your_dataframe
    sample_size=10000,          # Optional: use sample for faster processing
    save_outputs=True           # Save reports and visualizations
)
```

## Project Structure

```
clv-analysis/
├── config.py          # Configuration settings
├── utils.py           # Utility functions
├── models.py          # CLV models (RFM, BG/NBD, etc.)
├── visualization.py   # Visualization functions
├── reporting.py       # Report generation
├── main.py           # Main execution pipeline
├── requirements.txt   # Dependencies
└── README.md         # Documentation
```

## Usage Examples

### 1. Basic CLV Analysis

```python
from main import CLVAnalysisPipeline

pipeline = CLVAnalysisPipeline()
pipeline.load_data(df=your_dataframe)
rfm_df = pipeline.run_basic_analysis()
```

### 2. Advanced Lifetime Analysis

```python
# After basic analysis
lifetime_results = pipeline.run_lifetime_analysis(sample_size=5000)
```

### 3. Generate Reports Only

```python
pipeline.generate_reports(output_dir='reports/')
```

### 4. Custom Configuration

```python
from config import CLV_CONFIG

# Modify configuration
custom_config = CLV_CONFIG.copy()
custom_config['profit_margin'] = 0.25  # 25% profit margin
custom_config['discount_rate'] = 0.15  # 15% discount rate

pipeline = CLVAnalysisPipeline(config=custom_config)
```

## Data Requirements

Your transaction data should include these columns:
- `User_ID`: Customer identifier
- `Product_ID`: Product identifier
- `Purchase`: Transaction amount
- `Gender`: Customer gender (optional)
- `Age`: Age group (optional)
- `Occupation`: Customer occupation (optional)
- `City_Category`: City category (optional)
- `Product_Category`: Product category (optional)

## Customer Segments

The system identifies 8 customer segments:

1. **Champions**: Best customers (highest CLV)
2. **Loyal Customers**: Regular, high-value customers
3. **Potential Loyalists**: Recent customers with potential
4. **New Customers**: Recently acquired
5. **Promising**: Showing good potential
6. **Need Attention**: Declining engagement
7. **About to Sleep**: At risk of churning
8. **At Risk**: Likely to churn

## Output Files

When `save_outputs=True`, the system generates:

1. **executive_summary.txt**: High-level summary
2. **clv_detailed_report.xlsx**: Comprehensive Excel report with multiple sheets
3. **action_plan.csv**: Prioritized recommendations by segment
4. **kpi_dashboard.json**: KPI data for dashboards
5. **feature_importance.csv**: Model feature importance
6. **clv_dashboard.html**: Interactive Plotly dashboard

## Advanced Features

### Lifetime Models

- **BG/NBD Model**: Predicts future purchase behavior
- **Gamma-Gamma Model**: Predicts transaction values
- **Probability Alive**: Likelihood customer remains active

### Predictive Modeling

- Random Forest model for CLV prediction
- Feature importance analysis
- Model performance metrics (MSE, R²)

### Customization

All components can be used independently:

```python
from models import RFMAnalyzer, LifetimeValueModels
from visualization import CLVVisualizer
from reporting import CLVReporter

# Use individual components
rfm_analyzer = RFMAnalyzer()
rfm_df = rfm_analyzer.calculate_rfm_metrics(df)
```

## Troubleshooting

### Convergence Issues

If BG/NBD model fails to converge:
- The system automatically tries different penalizer values
- Falls back to simple CLV calculation if needed
- Check logs for details

### Memory Issues

For large datasets:
- Use `sample_size` parameter
- Process in batches
- Increase system memory

### Missing Dependencies

```bash
# Install specific package
pip install lifetimes

# Or reinstall all
pip install -r requirements.txt --upgrade
```

## Performance Considerations

- Basic analysis: ~1-2 minutes for 100k customers
- Lifetime models: ~5-10 minutes for 10k customers
- Use sampling for initial exploration
- Full analysis recommended for final results

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License.

## Contact

For questions or support, please open an issue on GitHub.