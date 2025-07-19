# visualization.py
"""
Visualization functions for CLV analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import logging
from typing import Dict, Optional, Tuple

from config import VIZ_CONFIG

logger = logging.getLogger(__name__)

class CLVVisualizer:
    """
    Create visualizations for CLV analysis
    """
    
    def __init__(self, figsize: Tuple = VIZ_CONFIG['figure_size']):
        self.figsize = figsize
        
    def plot_rfm_analysis(self, rfm_df: pd.DataFrame) -> None:
        """
        Create RFM analysis visualizations
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. CLV Distribution
        axes[0, 0].hist(rfm_df['CLV'], bins=50, edgecolor='black', alpha=0.7)
        axes[0, 0].axvline(rfm_df['CLV'].mean(), color='red', 
                           linestyle='--', label=f'Mean: ${rfm_df["CLV"].mean():,.2f}')
        axes[0, 0].set_title('CLV Distribution', fontsize=14, fontweight='bold')
        axes[0, 0].set_xlabel('CLV ($)')
        axes[0, 0].set_ylabel('Number of Customers')
        axes[0, 0].legend()
        
        # 2. CLV by Segment
        segment_clv = rfm_df.groupby('Segment')['CLV'].mean().sort_values(ascending=False)
        colors = plt.cm.viridis(np.linspace(0, 1, len(segment_clv)))
        segment_clv.plot(kind='bar', ax=axes[0, 1], color=colors)
        axes[0, 1].set_title('Average CLV by Customer Segment', fontsize=14, fontweight='bold')
        axes[0, 1].set_xlabel('Segment')
        axes[0, 1].set_ylabel('Average CLV ($)')
        plt.setp(axes[0, 1].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # 3. Frequency vs Monetary Value
        scatter = axes[1, 0].scatter(rfm_df['Frequency'], rfm_df['Total_Purchase'], 
                                    c=rfm_df['CLV'], cmap='viridis', alpha=0.6, s=50)
        axes[1, 0].set_xlabel('Purchase Frequency')
        axes[1, 0].set_ylabel('Total Purchase Value ($)')
        axes[1, 0].set_title('Frequency vs Monetary Value', fontsize=14, fontweight='bold')
        cbar = plt.colorbar(scatter, ax=axes[1, 0])
        cbar.set_label('CLV ($)')
        
        # 4. CLV by Demographics
        demographic_clv = rfm_df.groupby(['Gender', 'Age'])['CLV'].mean().unstack()
        demographic_clv.plot(kind='bar', ax=axes[1, 1], width=0.8)
        axes[1, 1].set_title('Average CLV by Demographics', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('Gender')
        axes[1, 1].set_ylabel('Average CLV ($)')
        axes[1, 1].legend(title='Age Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.show()
    
    def plot_lifetime_models(self, summary_df: pd.DataFrame, bgf=None) -> None:
        """
        Create visualizations for lifetime models
        """
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        
        # 1. CLV Distribution (12M)
        if 'CLV_12M' in summary_df.columns:
            axes[0, 0].hist(summary_df['CLV_12M'].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[0, 0].axvline(summary_df['CLV_12M'].mean(), color='red', 
                               linestyle='--', label=f'Mean: ${summary_df["CLV_12M"].mean():,.2f}')
            axes[0, 0].set_title('12-Month CLV Distribution', fontsize=14, fontweight='bold')
            axes[0, 0].set_xlabel('CLV ($)')
            axes[0, 0].set_ylabel('Number of Customers')
            axes[0, 0].legend()
        
        # 2. Probability Alive
        if 'probability_alive' in summary_df.columns:
            axes[0, 1].hist(summary_df['probability_alive'].dropna(), bins=50, edgecolor='black', alpha=0.7)
            axes[0, 1].set_title('Probability Customer is Alive', fontsize=14, fontweight='bold')
            axes[0, 1].set_xlabel('Probability')
            axes[0, 1].set_ylabel('Number of Customers')
            axes[0, 1].axvline(0.5, color='red', linestyle='--', label='50% threshold')
            axes[0, 1].legend()
        
        # 3. Frequency-Recency Matrix
        if bgf is not None:
            try:
                from lifetimes.plotting import plot_frequency_recency_matrix
                plot_frequency_recency_matrix(bgf, ax=axes[1, 0])
                axes[1, 0].set_title('Frequency-Recency Matrix', fontsize=14, fontweight='bold')
            except:
                axes[1, 0].text(0.5, 0.5, 'Frequency-Recency Matrix\nNot Available', 
                               ha='center', va='center', transform=axes[1, 0].transAxes)
        else:
            # Alternative visualization
            axes[1, 0].scatter(summary_df['recency'], summary_df['frequency'], 
                              c=summary_df.get('CLV_12M', summary_df['monetary_value']), 
                              cmap='viridis', alpha=0.6)
            axes[1, 0].set_xlabel('Recency (days)')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Recency vs Frequency', fontsize=14, fontweight='bold')
        
        # 4. Expected vs Actual Purchases
        if 'predicted_purchases_90_days' in summary_df.columns:
            axes[1, 1].scatter(summary_df['frequency'], 
                              summary_df['predicted_purchases_90_days'], alpha=0.5)
            max_val = max(summary_df['frequency'].max(), 
                         summary_df['predicted_purchases_90_days'].max())
            axes[1, 1].plot([0, max_val], [0, max_val], 'r--', label='Perfect Prediction')
            axes[1, 1].set_xlabel('Actual Frequency')
            axes[1, 1].set_ylabel('Predicted Purchases (90 days)')
            axes[1, 1].set_title('Model Predictions vs Actual', fontsize=14, fontweight='bold')
            axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
    
    def create_interactive_dashboard(self, rfm_df: pd.DataFrame, 
                                   summary_df: Optional[pd.DataFrame] = None) -> go.Figure:
        """
        Create interactive Plotly dashboard
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('CLV Distribution', 'CLV by Segment', 
                           'Customer Segments', 'CLV Trends'),
            specs=[[{'type': 'histogram'}, {'type': 'bar'}],
                   [{'type': 'pie'}, {'type': 'scatter'}]]
        )
        
        # 1. CLV Distribution
        fig.add_trace(
            go.Histogram(x=rfm_df['CLV'], name='CLV Distribution', nbinsx=50),
            row=1, col=1
        )
        
        # 2. CLV by Segment
        segment_data = rfm_df.groupby('Segment')['CLV'].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=segment_data.values, y=segment_data.index, 
                   orientation='h', name='Avg CLV'),
            row=1, col=2
        )
        
        # 3. Customer Segments Pie Chart
        segment_counts = rfm_df['Segment'].value_counts()
        fig.add_trace(
            go.Pie(labels=segment_counts.index, values=segment_counts.values,
                   name='Segments'),
            row=2, col=1
        )
        
        # 4. CLV vs Frequency Scatter
        fig.add_trace(
            go.Scatter(x=rfm_df['Frequency'], y=rfm_df['CLV'],
                      mode='markers', name='CLV vs Frequency',
                      marker=dict(color=rfm_df['Total_Purchase'], 
                                colorscale='viridis', showscale=True)),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            title_text="Customer Lifetime Value Dashboard",
            title_font_size=24,
            showlegend=False,
            height=800
        )
        
        # Update axes
        fig.update_xaxes(title_text="CLV ($)", row=1, col=1)
        fig.update_xaxes(title_text="Average CLV ($)", row=1, col=2)
        fig.update_xaxes(title_text="Purchase Frequency", row=2, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="CLV ($)", row=2, col=2)
        
        return fig
    
    def plot_pareto_analysis(self, rfm_df: pd.DataFrame) -> None:
        """
        Create Pareto analysis chart
        """
        # Sort by CLV descending
        sorted_df = rfm_df.sort_values('CLV', ascending=False).reset_index(drop=True)
        
        # Calculate cumulative values
        sorted_df['cumulative_clv'] = sorted_df['CLV'].cumsum()
        sorted_df['cumulative_pct'] = sorted_df['cumulative_clv'] / sorted_df['CLV'].sum() * 100
        sorted_df['customer_pct'] = (sorted_df.index + 1) / len(sorted_df) * 100
        
        # Create plot
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Bar chart for CLV
        ax1.bar(range(len(sorted_df)), sorted_df['CLV'], alpha=0.3, color='blue')
        ax1.set_xlabel('Customers (ranked by CLV)')
        ax1.set_ylabel('Individual CLV ($)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Line chart for cumulative percentage
        ax2 = ax1.twinx()
        ax2.plot(sorted_df['customer_pct'], sorted_df['cumulative_pct'], 
                color='red', linewidth=2, label='Cumulative CLV %')
        ax2.axhline(y=80, color='green', linestyle='--', label='80% threshold')
        ax2.set_ylabel('Cumulative CLV (%)', color='red')
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.set_ylim(0, 105)
        
        # Find 80/20 point
        idx_80 = sorted_df[sorted_df['cumulative_pct'] >= 80].index[0]
        pct_customers_80 = sorted_df.loc[idx_80, 'customer_pct']
        
        ax2.axvline(x=pct_customers_80, color='green', linestyle=':', alpha=0.7)
        ax2.text(pct_customers_80 + 2, 85, f'{pct_customers_80:.1f}% of customers\ngenerate 80% of CLV',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.5))
        
        plt.title('Pareto Analysis: Customer CLV Distribution', fontsize=16, fontweight='bold')
        ax2.legend(loc='center right')
        plt.tight_layout()
        plt.show()
    
    def create_cohort_analysis(self, rfm_df: pd.DataFrame) -> None:
        """
        Create cohort analysis visualization
        """
        # Create cohorts based on customer characteristics
        cohort_metrics = []
        
        # By Age
        for age in rfm_df['Age'].unique():
            age_df = rfm_df[rfm_df['Age'] == age]
            cohort_metrics.append({
                'Cohort_Type': 'Age',
                'Cohort': age,
                'Customers': len(age_df),
                'Avg_CLV': age_df['CLV'].mean(),
                'Total_CLV': age_df['CLV'].sum()
            })
        
        # By City Category
        for city in rfm_df['City_Category'].unique():
            city_df = rfm_df[rfm_df['City_Category'] == city]
            cohort_metrics.append({
                'Cohort_Type': 'City',
                'Cohort': f'City {city}',
                'Customers': len(city_df),
                'Avg_CLV': city_df['CLV'].mean(),
                'Total_CLV': city_df['CLV'].sum()
            })
        
        cohort_df = pd.DataFrame(cohort_metrics)
        
        # Create visualization
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Age cohorts
        age_cohorts = cohort_df[cohort_df['Cohort_Type'] == 'Age']
        age_cohorts = age_cohorts.sort_values('Avg_CLV', ascending=True)
        
        axes[0].barh(age_cohorts['Cohort'], age_cohorts['Avg_CLV'])
        axes[0].set_xlabel('Average CLV ($)')
        axes[0].set_title('Average CLV by Age Group', fontsize=14, fontweight='bold')
        
        # City cohorts
        city_cohorts = cohort_df[cohort_df['Cohort_Type'] == 'City']
        city_cohorts = city_cohorts.sort_values('Total_CLV', ascending=False)
        
        axes[1].bar(city_cohorts['Cohort'], city_cohorts['Total_CLV'])
        axes[1].set_ylabel('Total CLV ($)')
        axes[1].set_title('Total CLV by City Category', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.show()

def create_executive_report_plots(rfm_df: pd.DataFrame, 
                                 summary_df: Optional[pd.DataFrame] = None) -> None:
    """
    Create a set of plots for executive reporting
    """
    visualizer = CLVVisualizer()
    
    logger.info("Creating executive report visualizations...")
    
    # 1. Main RFM Analysis
    visualizer.plot_rfm_analysis(rfm_df)
    
    # 2. Pareto Analysis
    visualizer.plot_pareto_analysis(rfm_df)
    
    # 3. Cohort Analysis
    visualizer.create_cohort_analysis(rfm_df)
    
    # 4. Lifetime models (if available)
    if summary_df is not None:
        visualizer.plot_lifetime_models(summary_df)
    
    logger.info("Executive report visualizations completed")