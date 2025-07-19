# main.py
"""
Main execution file for CLV analysis
"""

import pandas as pd
import numpy as np
import logging
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple

# Import custom modules
from config import CLV_CONFIG
from utils import validate_data, create_synthetic_dates, export_results
from models import RFMAnalyzer, CLVPredictor, LifetimeValueModels
from visualization import create_executive_report_plots, CLVVisualizer
from reporting import CLVReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'clv_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class CLVAnalysisPipeline:
    """
    Complete CLV analysis pipeline
    """
    
    def __init__(self, config: Dict = CLV_CONFIG):
        self.config = config
        self.rfm_analyzer = RFMAnalyzer(config)
        self.clv_predictor = CLVPredictor()
        self.lifetime_models = LifetimeValueModels()
        self.visualizer = CLVVisualizer()
        self.reporter = CLVReporter()
        
        # Results storage
        self.results = {}
        
    def load_data(self, data_path: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Load data from file or use provided dataframe
        """
        if df is not None:
            logger.info("Using provided dataframe")
            self.df = df
        elif data_path:
            logger.info(f"Loading data from {data_path}")
            self.df = pd.read_csv(data_path)
        else:
            # Example: Load from kagglehub
            try:
                import kagglehub
                dataset_path = kagglehub.dataset_download("devarajv88/walmart-sales-dataset")
                data_dir = Path(dataset_path)
                csv_files = list(data_dir.glob("*.csv"))
                if csv_files:
                    self.df = pd.read_csv(csv_files[0])
                    logger.info(f"Loaded data from {csv_files[0].name}")
                    # Adjust Purchase values if needed
                    if self.df['Purchase'].max() > 10000:
                        self.df['Purchase'] = self.df['Purchase'] / 100
                        logger.info("Scaled Purchase values by 1/100")
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                raise
        
        # Validate data
        validate_data(self.df)
        logger.info(f"Data loaded successfully: {self.df.shape[0]} transactions, {self.df['User_ID'].nunique()} customers")
        
        return self.df
    
    def run_basic_analysis(self) -> pd.DataFrame:
        """
        Run basic RFM and CLV analysis
        """
        logger.info("="*60)
        logger.info("PHASE 1: BASIC CLV ANALYSIS")
        logger.info("="*60)
        
        # Step 1: Calculate RFM metrics
        logger.info("Step 1: Calculating RFM metrics...")
        rfm_df = self.rfm_analyzer.calculate_rfm_metrics(self.df)
        
        # Step 2: Create features
        logger.info("Step 2: Creating features...")
        rfm_df = self.rfm_analyzer.create_features(rfm_df, self.df)
        
        # Step 3: Segment customers
        logger.info("Step 3: Segmenting customers...")
        rfm_df = self.rfm_analyzer.segment_customers(rfm_df)
        
        # Step 4: Calculate CLV
        logger.info("Step 4: Calculating CLV...")
        rfm_df = self.rfm_analyzer.calculate_clv(rfm_df)
        
        # Step 5: Train predictive model
        logger.info("Step 5: Training predictive model...")
        model_results = self.clv_predictor.train(rfm_df)
        
        # Store results
        self.results['rfm_df'] = rfm_df
        self.results['model_metrics'] = model_results
        
        # Display results
        logger.info("\nBASIC ANALYSIS RESULTS:")
        logger.info(f"Total CLV: ${rfm_df['CLV'].sum():,.0f}")
        logger.info(f"Average CLV: ${rfm_df['CLV'].mean():,.2f}")
        logger.info(f"Model R²: {model_results['r2']:.3f}")
        
        return rfm_df
    
    def run_lifetime_analysis(self, sample_size: Optional[int] = None) -> Dict:
        """
        Run advanced lifetime value analysis
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: LIFETIME VALUE ANALYSIS")
        logger.info("="*60)
        
        if 'rfm_df' not in self.results:
            raise ValueError("Run basic analysis first")
        
        rfm_df = self.results['rfm_df']
        
        # Use sample if specified
        if sample_size and sample_size < len(self.df):
            logger.info(f"Using sample of {sample_size} customers")
            sampled_users = np.random.choice(self.df['User_ID'].unique(), 
                                           size=sample_size, replace=False)
            sample_df = self.df[self.df['User_ID'].isin(sampled_users)]
        else:
            sample_df = self.df
        
        # Step 1: Create synthetic dates
        logger.info("Step 1: Creating synthetic transaction dates...")
        transactions_df = create_synthetic_dates(sample_df)
        
        # Step 2: Create summary data for lifetime models
        logger.info("Step 2: Creating summary data...")
        try:
            from lifetimes.utils import summary_data_from_transaction_data
            summary_df = summary_data_from_transaction_data(
                transactions_df,
                'User_ID',
                'Transaction_Date',
                monetary_value_col='Purchase',
                observation_period_end=pd.Timestamp('2024-12-31')
            )
            logger.info(f"Summary created for {len(summary_df)} customers")
        except ImportError:
            logger.error("lifetimes package not installed. Run: pip install lifetimes")
            return None
        
        # Step 3: Fit BG/NBD model
        logger.info("Step 3: Fitting BG/NBD model...")
        bgf, summary_df = self.lifetime_models.fit_bgnbd(summary_df)
        
        # Step 4: Fit Gamma-Gamma model
        logger.info("Step 4: Fitting Gamma-Gamma model...")
        ggf, summary_df = self.lifetime_models.fit_gamma_gamma(summary_df)
        
        # Step 5: Calculate lifetime CLV
        logger.info("Step 5: Calculating lifetime CLV...")
        summary_df = self.lifetime_models.calculate_clv(summary_df)
        
        # Store results
        self.results['summary_df'] = summary_df
        self.results['bgf_model'] = bgf
        self.results['ggf_model'] = ggf
        self.results['transactions_df'] = transactions_df
        
        # Display results
        if 'CLV_12M' in summary_df.columns:
            logger.info("\nLIFETIME ANALYSIS RESULTS:")
            logger.info(f"Average 3-Month CLV: ${summary_df['CLV_3M'].mean():,.2f}")
            logger.info(f"Average 6-Month CLV: ${summary_df['CLV_6M'].mean():,.2f}")
            logger.info(f"Average 12-Month CLV: ${summary_df['CLV_12M'].mean():,.2f}")
            
            if 'probability_alive' in summary_df.columns:
                logger.info(f"Average Probability Alive: {summary_df['probability_alive'].mean():.2%}")
        
        return self.results
    
    def generate_visualizations(self, save_plots: bool = False) -> None:
        """
        Generate all visualizations
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: GENERATING VISUALIZATIONS")
        logger.info("="*60)
        
        if 'rfm_df' not in self.results:
            raise ValueError("Run analysis first")
        
        rfm_df = self.results['rfm_df']
        summary_df = self.results.get('summary_df', None)
        
        # Create visualizations
        create_executive_report_plots(rfm_df, summary_df)
        
        # Create interactive dashboard
        fig = self.visualizer.create_interactive_dashboard(rfm_df, summary_df)
        
        if save_plots:
            fig.write_html('clv_dashboard.html')
            logger.info("Interactive dashboard saved to clv_dashboard.html")
        
        return fig
    
    def generate_reports(self, output_dir: str = '.') -> None:
        """
        Generate all reports
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: GENERATING REPORTS")
        logger.info("="*60)
        
        if 'rfm_df' not in self.results:
            raise ValueError("Run analysis first")
        
        rfm_df = self.results['rfm_df']
        summary_df = self.results.get('summary_df', None)
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # 1. Executive Summary (text)
        exec_summary = self.reporter.generate_executive_summary(rfm_df, summary_df)
        with open(output_path / 'executive_summary.txt', 'w') as f:
            f.write(exec_summary)
        logger.info("Executive summary saved")
        
        # 2. Detailed Excel Report
        self.reporter.export_detailed_report(
            rfm_df, 
            summary_df,
            str(output_path / 'clv_detailed_report.xlsx')
        )
        
        # 3. Action Plan
        action_plan = self.reporter.generate_action_plan(rfm_df)
        action_plan.to_csv(output_path / 'action_plan.csv', index=False)
        logger.info("Action plan saved")
        
        # 4. KPI Dashboard Data
        kpis = self.reporter.generate_kpi_dashboard(rfm_df, summary_df)
        import json
        with open(output_path / 'kpi_dashboard.json', 'w') as f:
            json.dump(kpis, f, indent=2, default=str)
        logger.info("KPI dashboard data saved")
        
        # 5. Model results (if available)
        if 'model_metrics' in self.results:
            model_metrics = self.results['model_metrics']
            model_metrics['feature_importance'].to_csv(
                output_path / 'feature_importance.csv', 
                index=False
            )
            logger.info("Feature importance saved")
        
        logger.info(f"\nAll reports saved to: {output_path.absolute()}")
    
    def run_complete_analysis(self, 
                            data_path: str = None, 
                            df: pd.DataFrame = None,
                            sample_size: Optional[int] = None,
                            save_outputs: bool = True) -> Dict:
        """
        Run complete CLV analysis pipeline
        """
        logger.info("\n" + "="*80)
        logger.info("CUSTOMER LIFETIME VALUE ANALYSIS")
        logger.info("="*80)
        
        try:
            # Load data
            self.load_data(data_path, df)
            
            # Run basic analysis
            self.run_basic_analysis()
            
            # Run lifetime analysis
            self.run_lifetime_analysis(sample_size)
            
            # Generate visualizations
            self.generate_visualizations(save_plots=save_outputs)
            
            # Generate reports
            if save_outputs:
                self.generate_reports()
            
            logger.info("\n" + "="*80)
            logger.info("ANALYSIS COMPLETE!")
            logger.info("="*80)
            
            # Print summary
            print(self.reporter.generate_executive_summary(
                self.results['rfm_df'], 
                self.results.get('summary_df', None)
            ))
            
            return self.results
            
        except Exception as e:
            logger.error(f"Error in analysis pipeline: {e}")
            raise

def main():
    """
    Main execution function
    """
    # Initialize pipeline
    pipeline = CLVAnalysisPipeline()
    
    # Option 1: Load from file
    # results = pipeline.run_complete_analysis(data_path='your_data.csv')
    
    # Option 2: Use existing dataframe
    # results = pipeline.run_complete_analysis(df=your_dataframe)
    
    # Option 3: Load from kagglehub (default)
    results = pipeline.run_complete_analysis(
        sample_size=10000,  # Use sample for faster processing
        save_outputs=True
    )
    
    return results

if __name__ == "__main__":
    results = main()