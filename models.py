# models.py
"""
CLV modeling functions including RFM, BG/NBD, and Gamma-Gamma models
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import logging
from typing import Dict, Tuple, Optional, List

# Ensure these imports are correct for your project structure
from config import CLV_CONFIG, MAPPINGS, SEGMENT_RULES, LIFESPAN_MULTIPLIERS, BGNBD_PENALIZERS
from utils import safe_division, calculate_rfm_scores # Assuming these are defined in your utils.py

# Import lifetimes models explicitly
try:
    from lifetimes import BetaGeoFitter, GammaGammaFitter
    # from lifetimes import ParetoNBD as PNBDFitter # If you intend to use ParetoNBD
except ImportError:
    logging.warning("The 'lifetimes' package is not installed. Advanced CLV models (BG/NBD, Gamma-Gamma) will not be available.")
    BetaGeoFitter = None
    GammaGammaFitter = None
    # PNBDFitter = None


logger = logging.getLogger(__name__)

class RFMAnalyzer:
    """
    RFM (Recency, Frequency, Monetary) Analysis
    """
    
    def __init__(self, config: Dict = CLV_CONFIG):
        self.config = config
        
    def calculate_rfm_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate RFM metrics for each customer
        """
        logger.info("Calculating RFM metrics...")
        
        # Ensure 'Purchase' is numeric and handle potential missing 'Product_ID' for frequency count
        df['Purchase'] = pd.to_numeric(df['Purchase'], errors='coerce').fillna(0)
        
        # Aggregate by customer
        # For frequency, if 'Product_ID' is not robust, consider 'Purchase' count or even just `size` of group
        agg_dict = {
            'Purchase': ['sum', 'mean', 'std']
        }
        if 'Product_ID' in df.columns:
            agg_dict['Product_ID'] = 'count' # Using Product_ID count as Frequency proxy
        else:
            logger.warning("No 'Product_ID' column found. Using general transaction count for Frequency.")
            agg_dict['Purchase'] = ['count', 'sum', 'mean', 'std'] # 'count' will be the first aggregated result

        rfm_df = df.groupby('User_ID').agg(agg_dict).reset_index()
        
        # Flatten column names dynamically based on what was aggregated
        if 'Product_ID' in df.columns:
            rfm_df.columns = ['User_ID', 'Frequency', 'Total_Purchase', 'Avg_Purchase', 'Std_Purchase']
        else:
            rfm_df.columns = ['User_ID', 'Frequency', 'Total_Purchase', 'Avg_Purchase', 'Std_Purchase'] # If using purchase count
        
        # Fill NaN values in Std_Purchase with 0 (for customers with only one purchase)
        rfm_df['Std_Purchase'] = rfm_df['Std_Purchase'].fillna(0)
        
        # Add customer demographics (taking the first occurrence)
        # Check which demographic columns actually exist in df
        demographic_cols = ['Gender', 'Age', 'Occupation', 'City_Category', 
                            'Stay_In_Current_City_Years', 'Marital_Status']
        
        # Filter for existing columns in the original DataFrame
        existing_demog_cols = [col for col in demographic_cols if col in df.columns]
        
        if existing_demog_cols:
            customer_info = df.groupby('User_ID').first()[existing_demog_cols].reset_index()
            rfm_df = rfm_df.merge(customer_info, on='User_ID', how='left')
        else:
            logger.warning("No standard demographic columns found in the input DataFrame. Skipping demographic merge.")
            
        # Calculate coefficient of variation
        rfm_df['Purchase_CV'] = safe_division(
            rfm_df['Std_Purchase'], 
            rfm_df['Avg_Purchase']
        )
        
        logger.info(f"RFM metrics calculated for {len(rfm_df)} customers")
        
        return rfm_df
    
    def create_features(self, rfm_df: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create additional features for CLV prediction
        """
        logger.info("Creating CLV features...")
        
        # Product diversity
        if 'Product_ID' in df.columns:
            product_diversity = df.groupby('User_ID')['Product_ID'].nunique().reset_index()
            product_diversity.columns = ['User_ID', 'Product_Diversity']
            rfm_df = rfm_df.merge(product_diversity, on='User_ID', how='left')
        else:
            logger.warning("No 'Product_ID' column found for Product_Diversity feature.")
            rfm_df['Product_Diversity'] = 1 # Default to 1 if not available
            
        # Category diversity
        if 'Product_Category' in df.columns:
            category_diversity = df.groupby('User_ID')['Product_Category'].nunique().reset_index()
            category_diversity.columns = ['User_ID', 'Category_Diversity']
            rfm_df = rfm_df.merge(category_diversity, on='User_ID', how='left')
        else:
            logger.warning("No 'Product_Category' column found for Category_Diversity feature.")
            rfm_df['Category_Diversity'] = 1 # Default to 1 if not available

        # Encode categorical variables - check if columns exist before encoding
        if 'Gender' in rfm_df.columns:
            rfm_df['Gender_Encoded'] = (rfm_df['Gender'] == 'M').astype(int)
        else:
            rfm_df['Gender_Encoded'] = 0.5 # Neutral default if missing
            
        if 'Marital_Status' in rfm_df.columns:
            # Ensure Marital_Status is numeric before direct conversion
            rfm_df['Marital_Status_Encoded'] = pd.to_numeric(rfm_df['Marital_Status'], errors='coerce').fillna(0).astype(int)
        else:
            rfm_df['Marital_Status_Encoded'] = 0.5 # Neutral default if missing
        
        # Encode using mappings from config - safely apply mappings
        if 'Age' in rfm_df.columns and 'age' in MAPPINGS:
            rfm_df['Age_Encoded'] = rfm_df['Age'].map(MAPPINGS['age']).fillna(rfm_df['Age'].map(MAPPINGS['age']).mean()) # Fill with mean if mapping fails
        else:
            rfm_df['Age_Encoded'] = 0.5 # Neutral default if missing
            
        if 'City_Category' in rfm_df.columns and 'city' in MAPPINGS:
            rfm_df['City_Category_Encoded'] = rfm_df['City_Category'].map(MAPPINGS['city']).fillna(rfm_df['City_Category'].map(MAPPINGS['city']).mean())
        else:
            rfm_df['City_Category_Encoded'] = 0.5 # Neutral default if missing
            
        if 'Stay_In_Current_City_Years' in rfm_df.columns and 'stay_years' in MAPPINGS:
            rfm_df['Stay_Years_Encoded'] = rfm_df['Stay_In_Current_City_Years'].map(MAPPINGS['stay_years']).fillna(rfm_df['Stay_In_Current_City_Years'].map(MAPPINGS['stay_years']).mean())
        else:
            rfm_df['Stay_Years_Encoded'] = 0.5 # Neutral default if missing
        
        return rfm_df
    
    def segment_customers(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Segment customers based on RFM scores
        """
        logger.info("Segmenting customers...")
        
        # Calculate RFM scores
        # Ensure calculate_rfm_scores in utils.py can handle missing Recency/Monetary if they're not explicitly added yet
        rfm_df = calculate_rfm_scores(rfm_df)
        
        # Define segments based on rules
        def assign_segment(rfm_score):
            for segment, scores in SEGMENT_RULES.items():
                if rfm_score in scores:
                    return segment
            return 'Unsegmented/At Risk'  # Default segment if no match
        
        rfm_df['Segment'] = rfm_df['RFM_Score'].apply(assign_segment)
        
        # Log segment distribution
        segment_counts = rfm_df['Segment'].value_counts()
        logger.info(f"Customer segments:\n{segment_counts}")
        
        return rfm_df
    
    def calculate_clv(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CLV using simple formula
        """
        logger.info("Calculating CLV (simple method)...")
        
        # Ensure 'Avg_Purchase' and 'Frequency' are available
        if 'Avg_Purchase' not in rfm_df.columns or 'Frequency' not in rfm_df.columns:
            logger.error("Missing 'Avg_Purchase' or 'Frequency' for simple CLV calculation.")
            rfm_df['CLV'] = np.nan
            return rfm_df

        # Map segments to lifespan multipliers
        # Ensure 'Segment' column exists and LIFESPAN_MULTIPLIERS is correctly defined in config
        if 'Segment' in rfm_df.columns and LIFESPAN_MULTIPLIERS:
            rfm_df['Lifespan_Multiplier'] = rfm_df['Segment'].map(LIFESPAN_MULTIPLIERS).fillna(1.0) # Default to 1 if no map
        else:
            logger.warning("Segment or Lifespan Multipliers not available. Using default Lifespan_Multiplier=1.")
            rfm_df['Lifespan_Multiplier'] = 1.0
            
        # Calculate projected annual value
        # Ensure config values are numeric and accessible
        time_period_factor = safe_division(12, self.config.get('time_period', 1)) # Default to 1 month period
        
        rfm_df['Projected_Annual_Value'] = (
            rfm_df['Avg_Purchase'] * rfm_df['Frequency'] * time_period_factor
        ).fillna(0)
        
        # Calculate CLV
        profit_margin = self.config.get('profit_margin', 0.1) # Default 10%
        discount_rate = self.config.get('discount_rate', 0.05) # Default 5% annual discount
        
        rfm_df['CLV'] = (
            rfm_df['Projected_Annual_Value'] * rfm_df['Lifespan_Multiplier'] * profit_margin / 
            (1 + discount_rate)
        ).fillna(0) # Fill any NaNs from calculation

        logger.info(f"Average CLV (simple): ${rfm_df['CLV'].mean():,.2f}")
        
        return rfm_df

class CLVPredictor:
    """
    Machine Learning model for CLV prediction
    """
    
    def __init__(self):
        self.model = None
        self.scaler = None
        self.feature_cols = None
        self.feature_importance = None
        
    def prepare_features(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for modeling
        """
        # Define feature columns dynamically based on what was created
        # Ensure 'Occupation' is handled for one-hot encoding if it's categorical string
        base_features = [
            'Frequency', 'Total_Purchase', 'Avg_Purchase', 'Std_Purchase',
            'Product_Diversity', 'Category_Diversity', 'Gender_Encoded',
            'Age_Encoded', 'City_Category_Encoded',
            'Stay_Years_Encoded', 'Marital_Status_Encoded'
        ]
        
        # Handle 'Occupation' if it exists and is not already encoded
        if 'Occupation' in rfm_df.columns and rfm_df['Occupation'].dtype == 'object':
            # One-hot encode Occupation as it's typically categorical
            rfm_df = pd.get_dummies(rfm_df, columns=['Occupation'], prefix='Occupation', dummy_na=False)
            # Add new dummy columns to feature_cols
            occupation_dummies = [col for col in rfm_df.columns if col.startswith('Occupation_')]
            self.feature_cols = base_features + occupation_dummies
        else:
            self.feature_cols = base_features
        
        # Filter feature_cols to only include those actually present in the DataFrame
        self.feature_cols = [col for col in self.feature_cols if col in rfm_df.columns]
        
        # Remove rows with NaN values in selected features or CLV
        model_df = rfm_df.dropna(subset=[col for col in self.feature_cols if col in rfm_df.columns] + ['CLV'])
        
        if 'CLV' not in model_df.columns or model_df['CLV'].isnull().all():
            logger.warning("CLV column is missing or all NaN after preparing features. Model training may fail.")
        
        return model_df
    
    def train(self, rfm_df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train Random Forest model to predict CLV
        """
        logger.info("Training CLV prediction model...")
        
        # Prepare data
        model_df = self.prepare_features(rfm_df)
        
        if model_df.empty or 'CLV' not in model_df.columns or model_df['CLV'].isnull().all():
            logger.warning("Not enough valid data or 'CLV' column missing/all NaN after preparation. Skipping model training.")
            return {
                'mse': np.nan,
                'r2': np.nan,
                'feature_importance': pd.DataFrame()
            }

        X = model_df[self.feature_cols]
        y = model_df['CLV']
        
        if X.empty or y.empty or y.nunique() <= 1:
            logger.warning("Insufficient data or constant target variable for model training. Skipping model training.")
            return {
                'mse': np.nan,
                'r2': np.nan,
                'feature_importance': pd.DataFrame()
            }

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale features
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = RandomForestRegressor(
            n_estimators=100, 
            random_state=42, 
            n_jobs=-1
        )
        try:
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Feature importance
            self.feature_importance = pd.DataFrame({
                'feature': self.feature_cols,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info(f"Model trained - MSE: {mse:.2f}, R²: {r2:.3f}")
            
            return {
                'mse': mse,
                'r2': r2,
                'feature_importance': self.feature_importance
            }
        except Exception as e:
            logger.error(f"Error during Random Forest model training: {e}")
            self.model = None # Reset model if training fails
            return {
                'mse': np.nan,
                'r2': np.nan,
                'feature_importance': pd.DataFrame()
            }
    
    def predict(self, rfm_df: pd.DataFrame) -> np.ndarray:
        """
        Predict CLV for new customers
        """
        if self.model is None or self.scaler is None or self.feature_cols is None:
            raise ValueError("Model not trained or scaler/features not initialized. Call .train() first.")
        
        # To predict on new data, it needs to go through the same feature preparation,
        # including one-hot encoding for 'Occupation' if it was encoded during training.
        
        # Create a copy to avoid modifying the original rfm_df
        predict_df = rfm_df.copy()

        # Handle 'Occupation' encoding for prediction
        if 'Occupation' in predict_df.columns and predict_df['Occupation'].dtype == 'object':
            predict_df = pd.get_dummies(predict_df, columns=['Occupation'], prefix='Occupation', dummy_na=False)
        
        # Ensure all columns from self.feature_cols are present. Add missing ones with 0.
        for col in self.feature_cols:
            if col not in predict_df.columns:
                predict_df[col] = 0 # Assume 0 for missing feature during prediction (e.g., a dummy variable)
                logger.warning(f"Missing feature '{col}' in prediction data. Filling with 0.")
            # Ensure numeric type
            predict_df[col] = pd.to_numeric(predict_df[col], errors='coerce').fillna(0) # Also fill any NaNs introduced by coerce

        X = predict_df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)

class LifetimeValueModels:
    """
    Advanced lifetime value models (BG/NBD, Gamma-Gamma)
    """
    
    def __init__(self):
        self.bgf = None
        self.ggf = None
        self.pnbd = None # For ParetoNBD if used in the future
        
    def fit_bgnbd(self, summary_df: pd.DataFrame) -> Tuple[Optional[BetaGeoFitter], pd.DataFrame]:
        """
        Fit BG/NBD model with automatic penalizer adjustment
        """
        logger.info("Fitting BG/NBD model...")
        
        if BetaGeoFitter is None:
            logger.error("lifetimes package or BetaGeoFitter not available. Cannot fit BG/NBD model.")
            # Ensure predictions columns are initialized as NaN even if model fails to fit
            summary_df['predicted_purchases_14_days'] = np.nan
            summary_df['predicted_purchases_30_days'] = np.nan
            summary_df['predicted_purchases_90_days'] = np.nan
            summary_df['probability_alive'] = np.nan
            return None, summary_df
        
        # Ensure required columns are numeric and not all NaNs
        required_cols = ['frequency', 'recency', 'T']
        for col in required_cols:
            if col not in summary_df.columns or summary_df[col].isnull().all():
                logger.error(f"Missing or all NaN column '{col}' for BG/NBD fitting.")
                summary_df['predicted_purchases_14_days'] = np.nan
                summary_df['predicted_purchases_30_days'] = np.nan
                summary_df['predicted_purchases_90_days'] = np.nan
                summary_df['probability_alive'] = np.nan
                return None, summary_df
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').fillna(0) # Fill NaNs for model input
            
        # Filter for valid data points for BG/NBD
        # recency and T must be >= 0, frequency must be >= 0
        fit_df = summary_df[(summary_df['frequency'] >= 0) & (summary_df['recency'] >= 0) & (summary_df['T'] >= 0)].copy()

        if fit_df.empty:
            logger.warning("No valid data points for BG/NBD fitting after cleaning. Skipping model fitting.")
            summary_df['predicted_purchases_14_days'] = np.nan
            summary_df['predicted_purchases_30_days'] = np.nan
            summary_df['predicted_purchases_90_days'] = np.nan
            summary_df['probability_alive'] = np.nan
            return None, summary_df
            
        # Try different penalizer values
        for penalizer in BGNBD_PENALIZERS:
            try:
                logger.info(f"Trying BG/NBD with penalizer_coef={penalizer}...")
                bgf_temp = BetaGeoFitter(penalizer_coef=penalizer)
                bgf_temp.fit(
                    fit_df['frequency'], 
                    fit_df['recency'], 
                    fit_df['T']
                )
                self.bgf = bgf_temp # Assign if successful
                logger.info(f"BG/NBD model converged with penalizer={penalizer}")
                
                # Add predictions for ALL customers in the original summary_df
                # Use the original summary_df (or a copy) for predictions after fitting on fit_df subset
                summary_df = self._add_bgnbd_predictions(summary_df)
                
                return self.bgf, summary_df
                
            except Exception as e:
                logger.warning(f"Failed to fit BG/NBD with penalizer={penalizer}: {str(e)}")
                continue
        
        logger.error("BG/NBD model failed to converge with all tested penalizers.")
        self.bgf = None # Ensure model is None if fitting failed
        # Add default NaN predictions if all attempts fail
        summary_df['predicted_purchases_14_days'] = np.nan
        summary_df['predicted_purchases_30_days'] = np.nan
        summary_df['predicted_purchases_90_days'] = np.nan
        summary_df['probability_alive'] = np.nan
        return None, summary_df
    
    def _add_bgnbd_predictions(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add BG/NBD predictions to summary dataframe.
        Assumes self.bgf has been successfully fitted.
        """
        if self.bgf is None:
            logger.warning("BG/NBD model not fitted. Cannot add predictions.")
            summary_df['predicted_purchases_14_days'] = np.nan
            summary_df['predicted_purchases_30_days'] = np.nan
            summary_df['predicted_purchases_90_days'] = np.nan
            summary_df['probability_alive'] = np.nan
            return summary_df

        # Ensure prediction inputs are numeric and filled for safety
        freq_pred = summary_df['frequency'].fillna(0)
        rec_pred = summary_df['recency'].fillna(0)
        T_pred = summary_df['T'].fillna(0)

        try:
            summary_df['predicted_purchases_14_days'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(
                14, freq_pred, rec_pred, T_pred
            )
            
            summary_df['predicted_purchases_30_days'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(
                30, freq_pred, rec_pred, T_pred
            )
            
            summary_df['predicted_purchases_90_days'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(
                90, freq_pred, rec_pred, T_pred
            )
            
            summary_df['probability_alive'] = self.bgf.conditional_probability_alive(
                freq_pred, rec_pred, T_pred
            )
            
        except Exception as e:
            logger.error(f"Error adding BG/NBD predictions: {e}. Setting predictions to NaN.")
            summary_df['predicted_purchases_14_days'] = np.nan
            summary_df['predicted_purchases_30_days'] = np.nan
            summary_df['predicted_purchases_90_days'] = np.nan
            summary_df['probability_alive'] = np.nan
            
        return summary_df
    
    def fit_gamma_gamma(self, summary_df: pd.DataFrame) -> Tuple[Optional[GammaGammaFitter], pd.DataFrame]:
        """
        Fit Gamma-Gamma model for monetary value
        """
        logger.info("Fitting Gamma-Gamma model...")
        
        if GammaGammaFitter is None:
            logger.error("lifetimes package or GammaGammaFitter not available. Cannot fit Gamma-Gamma model.")
            summary_df['predicted_avg_value'] = np.nan
            return None, summary_df
        
        # Filter to returning customers only (frequency > 0 and monetary_value > 0)
        # Ensure monetary_value is numeric and positive
        summary_df['frequency'] = pd.to_numeric(summary_df['frequency'], errors='coerce').fillna(0)
        summary_df['monetary_value'] = pd.to_numeric(summary_df['monetary_value'], errors='coerce').fillna(0)

        returning_customers = summary_df[
            (summary_df['frequency'] > 0) & 
            (summary_df['monetary_value'] > 0)
        ].copy() # .copy() to prevent SettingWithCopyWarning
        
        if returning_customers.empty:
            logger.warning("No returning customers with positive monetary value found for Gamma-Gamma fitting. Skipping model fitting.")
            summary_df['predicted_avg_value'] = np.nan
            return None, summary_df
        
        try:
            self.ggf = GammaGammaFitter(penalizer_coef=0.01) # Use a small penalizer
            self.ggf.fit(
                returning_customers['frequency'], 
                returning_customers['monetary_value']
            )
            
            # Predict average transaction value for ALL customers in original summary_df
            # For customers with frequency=0, predicted_avg_value should be NaN or 0, as Gamma-Gamma is for repeat buyers.
            summary_df['predicted_avg_value'] = np.nan # Initialize as NaN
            
            # Apply prediction only where frequency > 0
            mask_repeat_buyers = (summary_df['frequency'] > 0)
            if mask_repeat_buyers.any():
                freq_for_pred = summary_df.loc[mask_repeat_buyers, 'frequency'].fillna(0)
                mon_for_pred = summary_df.loc[mask_repeat_buyers, 'monetary_value'].fillna(0)
                
                # Further filter to ensure positive monetary value for prediction input
                valid_gg_input_mask = (freq_for_pred > 0) & (mon_for_pred > 0)
                
                if valid_gg_input_mask.any():
                    summary_df.loc[mask_repeat_buyers & valid_gg_input_mask, 'predicted_avg_value'] = \
                        self.ggf.conditional_expected_average_profit(
                            freq_for_pred[valid_gg_input_mask],
                            mon_for_pred[valid_gg_input_mask]
                        )
                else:
                    logger.warning("No valid data for Gamma-Gamma prediction after filtering for positive frequency/monetary values.")
            
            # Fill NaN predicted_avg_value with 0 for customers who are not repeat buyers or had issues
            summary_df['predicted_avg_value'] = summary_df['predicted_avg_value'].fillna(0)
            
            logger.info("Gamma-Gamma model fitted successfully")
            return self.ggf, summary_df
            
        except Exception as e:
            logger.error(f"Error fitting Gamma-Gamma model: {e}. Setting predicted_avg_value to monetary_value fallback.")
            self.ggf = None # Ensure model is None if fitting failed
            summary_df['predicted_avg_value'] = summary_df['monetary_value'].fillna(0) # Fallback
            return None, summary_df
    
    def calculate_clv(self, summary_df: pd.DataFrame, 
                     bgf: Optional[BetaGeoFitter] = None, # <--- CORRECTED: Added bgf and ggf as parameters
                     ggf: Optional[GammaGammaFitter] = None, # <--- CORRECTED: Added bgf and ggf as parameters
                     months: List[int] = [3, 6, 12],
                     discount_rate: float = 0.01) -> pd.DataFrame:
        """
        Calculate CLV using lifetime models (BG/NBD and Gamma-Gamma).
        Uses provided fitted models (bgf, ggf) or falls back to self.bgf/self.ggf.
        """
        logger.info("Calculating lifetime CLV...")
        
        # Use provided models if available, else use instance models
        actual_bgf = bgf if bgf is not None else self.bgf
        actual_ggf = ggf if ggf is not None else self.ggf

        if actual_bgf is None or actual_ggf is None:
            logger.warning("BG/NBD or Gamma-Gamma model not fitted/provided. Using fallback CLV calculation.")
            # Fallback calculation if models are not available
            for m in months:
                col_name = f'CLV_{m}M'
                # Ensure frequency and monetary_value are numeric for fallback
                freq = summary_df['frequency'].fillna(0)
                mon_val = summary_df['monetary_value'].fillna(0)
                summary_df[col_name] = (
                    mon_val *
                    freq * (m / 12.0) # Assume frequency is annual or adjust based on data period
                ).fillna(0)
            return summary_df
        
        # Ensure required columns for `customer_lifetime_value` are numeric and not all NaNs
        required_cols = ['frequency', 'recency', 'T', 'monetary_value']
        for col in required_cols:
            if col not in summary_df.columns or summary_df[col].isnull().all():
                logger.error(f"Missing or all NaN column '{col}' for `customer_lifetime_value`. Using fallback calculation.")
                # Fallback calculation if crucial columns are missing/bad
                for m in months:
                    col_name = f'CLV_{m}M'
                    freq = summary_df['frequency'].fillna(0) if 'frequency' in summary_df.columns else 0
                    mon_val = summary_df['monetary_value'].fillna(0) if 'monetary_value' in summary_df.columns else 0
                    summary_df[col_name] = (mon_val * freq * (m / 12.0)).fillna(0)
                return summary_df
            summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce').fillna(0) # Fill NaNs for calculation

        # Use lifetime models
        for m in months:
            col_name = f'CLV_{m}M'
            try:
                # The `discount_rate` for `customer_lifetime_value` should be the per-period discount rate.
                # If discount_rate is annual (e.g., 0.01 for 1%), and `time` is in months,
                # you'd typically convert it: (1 + annual_rate)^(1/12) - 1 for monthly discount.
                # For simplicity or if discount rate is already low, 0.01/12 can be a proxy for monthly.
                # If `freq` in lifetimes.customer_lifetime_value corresponds to `T` (e.g., 'D' for days),
                # the discount rate needs to match that frequency.
                # For simplicity, let's use the given discount_rate (annual) and the `time` in months
                # and assume `lifetimes` handles the `freq='D'` internally for the period conversion.
                # A more precise financial discount rate per day would be (1 + annual_rate)^(1/365) - 1.
                # For basic CLV, often discount rate is set to 0.0 for initial exploration.

                clv = actual_ggf.customer_lifetime_value(
                    actual_bgf, 
                    summary_df['frequency'], 
                    summary_df['recency'],
                    summary_df['T'], 
                    summary_df['monetary_value'],
                    time=m,              # Predict CLV for `m` periods
                    freq='D',            # Units of 'T' (days in this case, from summary_data_from_transaction_data)
                    discount_rate=discount_rate # Use the annual rate, lifetimes handles conversion with 'freq'
                )
                summary_df[col_name] = clv.fillna(0) # Fill any NaNs from the CLV calculation
            except Exception as e:
                logger.error(f"Error calculating {col_name} using lifetimes models: {e}. Falling back to simple calculation.")
                # Fallback calculation for this specific month if `lifetimes` call fails
                summary_df[col_name] = (
                    summary_df['monetary_value'].fillna(0) *
                    summary_df['frequency'].fillna(0) *
                    (m / 12.0)
                ).fillna(0)
        
        return summary_df