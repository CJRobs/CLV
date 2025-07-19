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

# Ensure these modules are in the same directory or accessible via PYTHONPATH
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
        Load data from file or use provided dataframe.
        Prioritizes provided df, then data_path, then tries KaggleHub.
        """
        if df is not None:
            logger.info("Using provided dataframe.")
            self.df = df
        elif data_path:
            logger.info(f"Loading data from {data_path}.")
            try:
                self.df = pd.read_csv(data_path)
            except FileNotFoundError:
                logger.error(f"Error: Data file not found at {data_path}")
                raise
            except Exception as e:
                logger.error(f"Error reading CSV from {data_path}: {e}")
                raise
        else:
            # Attempt to load from kagglehub if no path or df is provided
            logger.info("No data path or dataframe provided. Attempting to load from KaggleHub 'devarajv88/walmart-sales-dataset'.")
            try:
                import kagglehub
                dataset_path = kagglehub.dataset_download("devarajv88/walmart-sales-dataset")
                data_dir = Path(dataset_path)
                csv_files = list(data_dir.glob("*.csv"))
                if csv_files:
                    self.df = pd.read_csv(csv_files[0])
                    logger.info(f"Loaded data from KaggleHub: {csv_files[0].name}")
                    # Adjust Purchase values if they appear unusually high
                    if 'Purchase' in self.df.columns:
                        if self.df['Purchase'].max() > 10000 and self.df['Purchase'].min() >= 0: # Heuristic for large values
                            self.df['Purchase'] = self.df['Purchase'] / 100
                            logger.info("Scaled 'Purchase' values by 1/100 (assuming original values were in cents or similar).")
                    else:
                        logger.warning("No 'Purchase' column found in loaded data. CLV calculation might be affected.")
                else:
                    logger.error("No CSV files found in the downloaded Kaggle dataset. Please check the dataset content.")
                    raise FileNotFoundError("No CSV files found in the Kaggle dataset.")
            except ImportError:
                logger.error("kagglehub is not installed. Please install it with 'pip install kagglehub' to use Kaggle datasets as default.")
                raise
            except Exception as e:
                logger.error(f"Error loading data from KaggleHub: {e}")
                raise
        
        # Validate data
        validate_data(self.df)
        logger.info(f"Data loaded successfully: {self.df.shape[0]} transactions, {self.df['User_ID'].nunique()} customers.")
        
        return self.df
    
    def run_basic_analysis(self) -> pd.DataFrame:
        """
        Run basic RFM and CLV analysis.
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
        
        # Step 4: Calculate CLV (basic method)
        logger.info("Step 4: Calculating CLV (basic method)...")
        rfm_df = self.rfm_analyzer.calculate_clv(rfm_df)
        
        # Step 5: Train predictive model
        logger.info("Step 5: Training predictive model for basic CLV...")
        model_results = self.clv_predictor.train(rfm_df)
        
        # Store results
        self.results['rfm_df'] = rfm_df
        self.results['model_metrics'] = model_results
        
        # Display results
        logger.info("\nBASIC ANALYSIS RESULTS:")
        if 'CLV' in rfm_df.columns and not rfm_df['CLV'].isnull().all():
            logger.info(f"Total Basic CLV: ${rfm_df['CLV'].sum():,.0f}")
            logger.info(f"Average Basic CLV: ${rfm_df['CLV'].mean():,.2f}")
        else:
            logger.warning("Basic CLV not calculated or contains only NaN values.")

        if 'r2' in model_results and pd.notna(model_results['r2']):
            logger.info(f"Basic CLV Model R²: {model_results['r2']:.3f}")
        else:
            logger.warning("Basic CLV Model R² not available (training might have failed).")
        
        return rfm_df
    
    def run_lifetime_analysis(self, sample_size: Optional[int] = None) -> Dict:
        """
        Run advanced lifetime value analysis using `lifetimes` library.
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 2: LIFETIME VALUE ANALYSIS")
        logger.info("="*60)
        
        if 'rfm_df' not in self.results:
            logger.error("Basic analysis (Phase 1) must be run first to initialize RFM data.")
            raise ValueError("Run basic analysis first to get RFM data.")
        
        # Determine the population for sampling unique User_IDs
        all_unique_users = self.df['User_ID'].unique()
        num_unique_users = len(all_unique_users)

        # Apply sampling if sample_size is provided and valid
        if sample_size is not None and sample_size > 0:
            if sample_size < num_unique_users:
                logger.info(f"Using a sample of {sample_size} customers from {num_unique_users} total unique customers for lifetime analysis.")
                sampled_users = np.random.choice(all_unique_users, size=sample_size, replace=False)
                sample_df = self.df[self.df['User_ID'].isin(sampled_users)].copy() # Use .copy() to avoid SettingWithCopyWarning
            else:
                logger.warning(f"Requested sample size ({sample_size}) is greater than or equal to the total unique customers ({num_unique_users}). Using all unique customers for lifetime analysis.")
                sample_df = self.df.copy()
        else:
            logger.info(f"No valid sample size specified. Using all {num_unique_users} unique customers for lifetime analysis.")
            sample_df = self.df.copy()

        if sample_df.empty:
            logger.warning("Sampled DataFrame is empty. Skipping lifetime analysis completely.")
            return {}

        # Step 1: Create synthetic dates (if original data lacks them)
        logger.info("Step 1: Creating synthetic transaction dates (if needed)...")
        transactions_df = create_synthetic_dates(sample_df)

        if transactions_df.empty:
            logger.warning("Transactions DataFrame is empty after creating synthetic dates. Skipping lifetime analysis.")
            return {}
        
        # Step 2: Create summary data for lifetime models (RFM summary for lifetimes)
        logger.info("Step 2: Creating summary data for lifetimes models...")
        summary_df = pd.DataFrame() # Initialize to ensure it exists
        try:
            from lifetimes.utils import summary_data_from_transaction_data
            
            # Ensure 'Transaction_Date' is datetime and 'Purchase' is numeric
            transactions_df['Transaction_Date'] = pd.to_datetime(transactions_df['Transaction_Date'], errors='coerce')
            transactions_df['Purchase'] = pd.to_numeric(transactions_df['Purchase'], errors='coerce')
            transactions_df.dropna(subset=['Transaction_Date', 'Purchase', 'User_ID'], inplace=True)

            if transactions_df.empty:
                logger.warning("Transactions DataFrame is empty after type conversion and dropping NaNs. Cannot create summary data.")
                return {}

            # Set observation_period_end to the latest date in the transactions data
            # Plus one day to ensure the last transaction is included in the T value
            observation_end = transactions_df['Transaction_Date'].max() + pd.Timedelta(days=1)
            
            summary_df = summary_data_from_transaction_data(
                transactions_df,
                'User_ID',
                'Transaction_Date',
                monetary_value_col='Purchase',
                observation_period_end=observation_end
            )
            logger.info(f"Summary data created for {len(summary_df)} customers for lifetime models.")
        except ImportError:
            logger.error("The 'lifetimes' package is not installed. Please install it with 'pip install lifetimes' to run lifetime analysis.")
            return {}
        except Exception as e:
            logger.error(f"Error creating summary data from transaction data for lifetime models: {e}")
            return {} 
        
        # Ensure summary_df is not empty before proceeding to model fitting
        if summary_df.empty:
            logger.warning("Summary DataFrame is empty after creation. Skipping lifetime model fitting.")
            return {}

        # Step 3: Fit BG/NBD model
        logger.info("Step 3: Fitting BG/NBD model...")
        # Pass a copy to methods to avoid SettingWithCopyWarning
        bgf, summary_df = self.lifetime_models.fit_bgnbd(summary_df.copy())
        
        # Step 4: Fit Gamma-Gamma model
        logger.info("Step 4: Fitting Gamma-Gamma model...")
        ggf, summary_df = self.lifetime_models.fit_gamma_gamma(summary_df.copy())
        
        # Step 5: Calculate lifetime CLV
        logger.info("Step 5: Calculating lifetime CLV using BG/NBD and Gamma-Gamma models...")
        # Ensure bgf and ggf are passed, as the method now expects them
        summary_df = self.lifetime_models.calculate_clv(summary_df.copy(), bgf=bgf, ggf=ggf)
        
        # Store results
        self.results['summary_df'] = summary_df
        self.results['bgf_model'] = bgf
        self.results['ggf_model'] = ggf
        self.results['transactions_df'] = transactions_df # Store for potential debugging/visualization
        
        # Display results
        logger.info("\nLIFETIME ANALYSIS RESULTS:")
        if 'CLV_12M' in summary_df.columns and not summary_df['CLV_12M'].isnull().all():
            if 'CLV_3M' in summary_df.columns and not summary_df['CLV_3M'].isnull().all():
                logger.info(f"Average 3-Month CLV (Lifetime): ${summary_df['CLV_3M'].mean():,.2f}")
            if 'CLV_6M' in summary_df.columns and not summary_df['CLV_6M'].isnull().all():
                logger.info(f"Average 6-Month CLV (Lifetime): ${summary_df['CLV_6M'].mean():,.2f}")
            logger.info(f"Average 12-Month CLV (Lifetime): ${summary_df['CLV_12M'].mean():,.2f}")
            
            if 'probability_alive' in summary_df.columns and not summary_df['probability_alive'].isnull().all():
                logger.info(f"Average Probability Alive: {summary_df['probability_alive'].mean():.2%}")
            else:
                logger.warning("Probability alive data not available for display (BG/NBD model might not have fitted).")
        else:
            logger.warning("Lifetime CLV (12-Month) data not available or all values are NaN. Check model fitting steps.")
        
        return self.results
    
    def generate_visualizations(self, save_plots: bool = False) -> None:
        """
        Generate all visualizations for executive report and interactive dashboard.
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 3: GENERATING VISUALIZATIONS")
        logger.info("="*60)
        
        # Check if basic analysis results (rfm_df) are available
        if 'rfm_df' not in self.results or self.results['rfm_df'].empty:
            logger.warning("RFM data not available or is empty. Skipping visualization generation.")
            return
        
        rfm_df = self.results['rfm_df']
        summary_df = self.results.get('summary_df', None) # lifetime analysis results are optional

        # Create static plots for executive report
        logger.info("Creating static plots for executive report...")
        create_executive_report_plots(rfm_df, summary_df)
        
        # Create interactive dashboard
        logger.info("Creating interactive CLV dashboard...")
        fig = self.visualizer.create_interactive_dashboard(rfm_df, summary_df)
        
        if save_plots and fig: # Ensure fig is not None before trying to save
            plot_output_dir = Path('.') # Default to current directory
            # Consider adding a specific output directory for plots to config.py
            # e.g., output_dir = Path(self.config.get('plot_output_dir', '.'))
            plot_output_dir.mkdir(parents=True, exist_ok=True) # Ensure directory exists
            
            html_path = plot_output_dir / 'clv_dashboard.html'
            fig.write_html(str(html_path))
            logger.info(f"Interactive dashboard saved to {html_path.resolve()}.")
        elif not save_plots:
            logger.info("Interactive dashboard not saved (save_plots is False).")
        else: # fig is None or invalid
            logger.warning("Interactive dashboard figure not created, cannot save.")
        
        # Optional: Display interactive plot if running in an environment that supports it
        # try:
        #     if fig:
        #         fig.show()
        # except Exception as e:
        #     logger.warning(f"Could not display interactive plot (requires suitable environment, e.g., Jupyter): {e}")

    def generate_reports(self, output_dir: str = '.') -> None:
        """
        Generate all textual and tabular reports.
        """
        logger.info("\n" + "="*60)
        logger.info("PHASE 4: GENERATING REPORTS")
        logger.info("="*60)
        
        if 'rfm_df' not in self.results or self.results['rfm_df'].empty:
            logger.warning("RFM data not available or is empty. Skipping report generation.")
            return
        
        rfm_df = self.results['rfm_df']
        summary_df = self.results.get('summary_df', None) # lifetime analysis results are optional
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Reports will be saved to: {output_path.absolute()}")
        
        # 1. Executive Summary (text file)
        exec_summary = self.reporter.generate_executive_summary(rfm_df, summary_df)
        with open(output_path / 'executive_summary.txt', 'w') as f:
            f.write(exec_summary)
        logger.info("Executive summary saved to executive_summary.txt.")
        
        # 2. Detailed Excel Report
        excel_report_path = output_path / 'clv_detailed_report.xlsx'
        self.reporter.export_detailed_report(
            rfm_df, 
            summary_df,
            str(excel_report_path)
        )
        logger.info(f"Detailed Excel report saved to {excel_report_path.name}.")
        
        # 3. Action Plan (CSV)
        action_plan = self.reporter.generate_action_plan(rfm_df)
        if not action_plan.empty:
            action_plan_path = output_path / 'action_plan.csv'
            action_plan.to_csv(action_plan_path, index=False)
            logger.info(f"Action plan saved to {action_plan_path.name}.")
        else:
            logger.warning("Action plan is empty; not saving action_plan.csv.")
        
        # 4. KPI Dashboard Data (JSON)
        kpis = self.reporter.generate_kpi_dashboard(rfm_df, summary_df)
        import json
        kpi_json_path = output_path / 'kpi_dashboard.json'
        with open(kpi_json_path, 'w') as f:
            # Use default=str for any non-serializable types like datetime objects
            json.dump(kpis, f, indent=2, default=str)
        logger.info(f"KPI dashboard data saved to {kpi_json_path.name}.")
        
        # 5. Model results (Feature Importance CSV, if available)
        if 'model_metrics' in self.results and 'feature_importance' in self.results['model_metrics']:
            model_feature_importance = self.results['model_metrics']['feature_importance']
            if not model_feature_importance.empty:
                feature_importance_path = output_path / 'feature_importance.csv'
                model_feature_importance.to_csv(feature_importance_path, index=False)
                logger.info(f"Feature importance saved to {feature_importance_path.name}.")
            else:
                logger.warning("Feature importance data is empty; not saving feature_importance.csv.")
        else:
            logger.info("Model metrics or feature importance not available for saving.")
        
        logger.info(f"\nAll reports saved to: {output_path.absolute()}")
    
    def run_complete_analysis(self, 
                            data_path: Optional[str] = None, 
                            df: Optional[pd.DataFrame] = None,
                            sample_size: Optional[int] = None,
                            save_outputs: bool = True,
                            output_directory: str = '.') -> Dict:
        """
        Runs the complete CLV analysis pipeline from data loading to reporting.
        """
        logger.info("\n" + "="*80)
        logger.info("STARTING CUSTOMER LIFETIME VALUE ANALYSIS PIPELINE")
        logger.info("="*80)
        
        try:
            # Create the main output directory upfront
            Path(output_directory).mkdir(parents=True, exist_ok=True)

            # Load data
            self.load_data(data_path, df)
            
            # Run basic analysis (RFM, simple CLV, predictive model)
            self.run_basic_analysis()
            
            # Run lifetime analysis (BG/NBD, Gamma-Gamma, probabilistic CLV)
            # This is where the sample_size will be applied
            self.run_lifetime_analysis(sample_size)
            
            # Generate visualizations (static plots and interactive dashboard)
            self.generate_visualizations(save_plots=save_outputs)
            
            # Generate reports (text summary, excel, action plan, KPIs, feature importance)
            if save_outputs:
                self.generate_reports(output_dir=output_directory)
            else:
                logger.info("Output reports not saved as 'save_outputs' is False.")
            
            logger.info("\n" + "="*80)
            logger.info("CUSTOMER LIFETIME VALUE ANALYSIS COMPLETE!")
            logger.info("="*80)
            
            # Print a summary to console for quick review
            if 'rfm_df' in self.results and not self.results['rfm_df'].empty:
                print("\n--- Executive Summary (Console) ---")
                print(self.reporter.generate_executive_summary(
                    self.results['rfm_df'], 
                    self.results.get('summary_df', None)
                ))
                print("-----------------------------------")
            else:
                logger.error("RFM data not available after analysis. Cannot print executive summary to console.")
            
            return self.results
            
        except Exception as e:
            logger.critical(f"A critical error occurred in the CLV analysis pipeline: {e}", exc_info=True)
            raise # Re-raise the exception after logging for debugging

def main():
    """
    Main execution function to run the CLV analysis pipeline.
    """
    # Initialize pipeline
    pipeline = CLVAnalysisPipeline()
    
    results = pipeline.run_complete_analysis(
        sample_size=2000,  # Process 1000 unique customers for faster execution on large datasets.
                           # Adjust this number based on your system's memory and desired speed.
                           # Set to None to process all unique customers.
        save_outputs=True, # Set to False if you only want console output and no files
        output_directory='./clv_results_kaggle' # Specify an output folder for all generated files
    )
    
    return results

if __name__ == "__main__":
    results = main()