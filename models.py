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

from config import *
from utils import safe_division, calculate_rfm_scores

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
        
        # Aggregate by customer
        rfm_df = df.groupby('User_ID').agg({
            'Purchase': ['count', 'sum', 'mean', 'std']
        }).reset_index()
        
        # Flatten column names
        rfm_df.columns = ['User_ID', 'Frequency', 'Total_Purchase', 
                          'Avg_Purchase', 'Std_Purchase']
        
        # Fill NaN values in Std_Purchase with 0
        rfm_df['Std_Purchase'] = rfm_df['Std_Purchase'].fillna(0)
        
        # Add customer demographics (taking the first occurrence)
        customer_info = df.groupby('User_ID').first()[
            ['Gender', 'Age', 'Occupation', 'City_Category', 
             'Stay_In_Current_City_Years', 'Marital_Status']
        ].reset_index()
        
        rfm_df = rfm_df.merge(customer_info, on='User_ID', how='left')
        
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
        product_diversity = df.groupby('User_ID')['Product_ID'].nunique().reset_index()
        product_diversity.columns = ['User_ID', 'Product_Diversity']
        
        # Category diversity
        category_diversity = df.groupby('User_ID')['Product_Category'].nunique().reset_index()
        category_diversity.columns = ['User_ID', 'Category_Diversity']
        
        # Merge diversity metrics
        rfm_df = rfm_df.merge(product_diversity, on='User_ID', how='left')
        rfm_df = rfm_df.merge(category_diversity, on='User_ID', how='left')
        
        # Encode categorical variables
        rfm_df['Gender_Encoded'] = (rfm_df['Gender'] == 'M').astype(int)
        rfm_df['Marital_Status_Encoded'] = rfm_df['Marital_Status'].astype(int)
        
        # Encode using mappings from config
        rfm_df['Age_Encoded'] = rfm_df['Age'].map(MAPPINGS['age'])
        rfm_df['City_Category_Encoded'] = rfm_df['City_Category'].map(MAPPINGS['city'])
        rfm_df['Stay_Years_Encoded'] = rfm_df['Stay_In_Current_City_Years'].map(MAPPINGS['stay_years'])
        
        return rfm_df
    
    def segment_customers(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Segment customers based on RFM scores
        """
        logger.info("Segmenting customers...")
        
        # Calculate RFM scores
        rfm_df = calculate_rfm_scores(rfm_df)
        
        # Define segments based on rules
        def assign_segment(rfm_score):
            for segment, scores in SEGMENT_RULES.items():
                if rfm_score in scores:
                    return segment
            return 'At Risk'  # Default segment
        
        rfm_df['Segment'] = rfm_df['RFM_Score'].apply(assign_segment)
        
        # Log segment distribution
        segment_counts = rfm_df['Segment'].value_counts()
        logger.info(f"Customer segments:\n{segment_counts}")
        
        return rfm_df
    
    def calculate_clv(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CLV using simple formula
        """
        logger.info("Calculating CLV...")
        
        # Map segments to lifespan multipliers
        rfm_df['Lifespan_Multiplier'] = rfm_df['Segment'].map(LIFESPAN_MULTIPLIERS)
        
        # Calculate projected annual value
        rfm_df['Projected_Annual_Value'] = (
            rfm_df['Avg_Purchase'] * 
            rfm_df['Frequency'] * 
            (12 / self.config['time_period'])
        )
        
        # Calculate CLV
        rfm_df['CLV'] = (
            rfm_df['Projected_Annual_Value'] * 
            rfm_df['Lifespan_Multiplier'] * 
            self.config['profit_margin'] / 
            (1 + self.config['discount_rate'])
        )
        
        logger.info(f"Average CLV: ${rfm_df['CLV'].mean():,.2f}")
        
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
        self.feature_cols = [
            'Frequency', 'Total_Purchase', 'Avg_Purchase', 'Std_Purchase',
            'Product_Diversity', 'Category_Diversity', 'Gender_Encoded',
            'Age_Encoded', 'Occupation', 'City_Category_Encoded',
            'Stay_Years_Encoded', 'Marital_Status_Encoded'
        ]
        
        # Remove rows with NaN values
        model_df = rfm_df.dropna(subset=self.feature_cols + ['CLV'])
        
        return model_df
    
    def train(self, rfm_df: pd.DataFrame, test_size: float = 0.2) -> Dict:
        """
        Train Random Forest model to predict CLV
        """
        logger.info("Training CLV prediction model...")
        
        # Prepare data
        model_df = self.prepare_features(rfm_df)
        
        X = model_df[self.feature_cols]
        y = model_df['CLV']
        
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
    
    def predict(self, rfm_df: pd.DataFrame) -> np.ndarray:
        """
        Predict CLV for new customers
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        model_df = self.prepare_features(rfm_df)
        X = model_df[self.feature_cols]
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)

class LifetimeValueModels:
    """
    Advanced lifetime value models (BG/NBD, Gamma-Gamma)
    """
    
    def __init__(self):
        self.bgf = None
        self.ggf = None
        self.pnbd = None
        
    def fit_bgnbd(self, summary_df: pd.DataFrame) -> Tuple[Optional[object], pd.DataFrame]:
        """
        Fit BG/NBD model with automatic penalizer adjustment
        """
        logger.info("Fitting BG/NBD model...")
        
        try:
            from lifetimes import BetaGeoFitter
        except ImportError:
            logger.error("lifetimes package not installed")
            return None, summary_df
        
        # Try different penalizer values
        for penalizer in BGNBD_PENALIZERS:
            try:
                logger.info(f"Trying penalizer_coef={penalizer}...")
                self.bgf = BetaGeoFitter(penalizer_coef=penalizer)
                self.bgf.fit(
                    summary_df['frequency'], 
                    summary_df['recency'], 
                    summary_df['T']
                )
                logger.info(f"BG/NBD model converged with penalizer={penalizer}")
                
                # Add predictions
                summary_df = self._add_bgnbd_predictions(summary_df)
                
                return self.bgf, summary_df
                
            except Exception as e:
                logger.warning(f"Failed with penalizer={penalizer}: {str(e)[:50]}")
                continue
        
        logger.error("BG/NBD model failed to converge")
        return None, summary_df
    
    def _add_bgnbd_predictions(self, summary_df: pd.DataFrame) -> pd.DataFrame:
        """
        Add BG/NBD predictions to summary dataframe
        """
        try:
            summary_df['predicted_purchases_14_days'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(
                14, summary_df['frequency'], summary_df['recency'], summary_df['T']
            )
            
            summary_df['predicted_purchases_30_days'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(
                30, summary_df['frequency'], summary_df['recency'], summary_df['T']
            )
            
            summary_df['predicted_purchases_90_days'] = self.bgf.conditional_expected_number_of_purchases_up_to_time(
                90, summary_df['frequency'], summary_df['recency'], summary_df['T']
            )
            
            summary_df['probability_alive'] = self.bgf.conditional_probability_alive(
                summary_df['frequency'], summary_df['recency'], summary_df['T']
            )
            
        except Exception as e:
            logger.error(f"Error adding BG/NBD predictions: {e}")
            # Add default values
            summary_df['probability_alive'] = 0.5
            
        return summary_df
    
    def fit_gamma_gamma(self, summary_df: pd.DataFrame) -> Tuple[Optional[object], pd.DataFrame]:
        """
        Fit Gamma-Gamma model for monetary value
        """
        logger.info("Fitting Gamma-Gamma model...")
        
        try:
            from lifetimes import GammaGammaFitter
        except ImportError:
            logger.error("lifetimes package not installed")
            return None, summary_df
        
        # Filter to returning customers only
        returning_customers = summary_df[summary_df['frequency'] > 0]
        
        if len(returning_customers) == 0:
            logger.error("No returning customers found")
            return None, summary_df
        
        try:
            self.ggf = GammaGammaFitter(penalizer_coef=0.01)
            self.ggf.fit(
                returning_customers['frequency'], 
                returning_customers['monetary_value']
            )
            
            # Predict average transaction value
            summary_df['predicted_avg_value'] = self.ggf.conditional_expected_average_profit(
                summary_df['frequency'],
                summary_df['monetary_value']
            )
            
            logger.info("Gamma-Gamma model fitted successfully")
            return self.ggf, summary_df
            
        except Exception as e:
            logger.error(f"Error fitting Gamma-Gamma model: {e}")
            summary_df['predicted_avg_value'] = summary_df['monetary_value']
            return None, summary_df
    
    def calculate_clv(self, summary_df: pd.DataFrame, 
                     months: List[int] = [3, 6, 12],
                     discount_rate: float = 0.01) -> pd.DataFrame:
        """
        Calculate CLV using lifetime models
        """
        logger.info("Calculating lifetime CLV...")
        
        if self.bgf is None or self.ggf is None:
            logger.warning("Using fallback CLV calculation")
            # Fallback calculation
            for m in months:
                col_name = f'CLV_{m}M'
                summary_df[col_name] = (
                    summary_df['monetary_value'] * 
                    summary_df['frequency'] * 
                    (m / 12)
                )
        else:
            # Use lifetime models
            for m in months:
                col_name = f'CLV_{m}M'
                try:
                    clv = self.ggf.customer_lifetime_value(
                        self.bgf, 
                        summary_df['frequency'], 
                        summary_df['recency'],
                        summary_df['T'], 
                        summary_df['monetary_value'],
                        time=m, 
                        discount_rate=discount_rate/12
                    )
                    summary_df[col_name] = clv
                except Exception as e:
                    logger.error(f"Error calculating {col_name}: {e}")
                    # Fallback
                    summary_df[col_name] = (
                        summary_df['monetary_value'] * 
                        summary_df['frequency'] * 
                        (m / 12)
                    )
        
        return summary_df