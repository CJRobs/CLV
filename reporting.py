# reporting.py
"""
Reporting and summary functions for CLV analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Optional, List
from config import SEGMENT_RECOMMENDATIONS

logger = logging.getLogger(__name__)

class CLVReporter:
    """
    Generate reports and summaries for CLV analysis
    """
    
    def __init__(self):
        self.report_date = datetime.now().strftime('%Y-%m-%d')
        
    def generate_executive_summary(self, rfm_df: pd.DataFrame, 
                                 summary_df: Optional[pd.DataFrame] = None) -> str:
        """
        Generate executive summary report
        """
        report = []
        report.append("="*80)
        report.append("CUSTOMER LIFETIME VALUE (CLV) EXECUTIVE SUMMARY")
        report.append(f"Report Date: {self.report_date}")
        report.append("="*80)
        
        # Overall metrics
        report.append("\n📊 KEY PERFORMANCE INDICATORS:")
        report.append(f"• Total Customers: {len(rfm_df):,}")
        report.append(f"• Total CLV: ${rfm_df['CLV'].sum():,.0f}")
        report.append(f"• Average CLV: ${rfm_df['CLV'].mean():,.2f}")
        report.append(f"• Median CLV: ${rfm_df['CLV'].median():,.2f}")
        
        # Pareto analysis
        sorted_clv = rfm_df['CLV'].sort_values(ascending=False)
        cumsum = sorted_clv.cumsum()
        total = sorted_clv.sum()
        pct_80 = len(cumsum[cumsum <= total * 0.8]) / len(rfm_df) * 100
        
        report.append(f"\n📈 PARETO ANALYSIS:")
        report.append(f"• Top {pct_80:.1f}% of customers generate 80% of CLV")
        report.append(f"• Top 10% generate ${sorted_clv.head(int(len(rfm_df)*0.1)).sum():,.0f}")
        
        # Segment analysis
        report.append("\n🎯 CUSTOMER SEGMENTS:")
        segment_summary = self._create_segment_summary(rfm_df)
        for _, row in segment_summary.iterrows():
            report.append(f"\n{row.name}:")
            report.append(f"  • Customers: {row['Customer_Count']:,} ({row['Customer_Pct']:.1f}%)")
            report.append(f"  • Total CLV: ${row['Total_CLV']:,.0f} ({row['CLV_Pct']:.1f}%)")
            report.append(f"  • Avg CLV: ${row['Avg_CLV']:,.2f}")
            report.append(f"  • Action: {SEGMENT_RECOMMENDATIONS.get(row.name, 'Monitor')}")
        
        # Risk analysis
        at_risk_segments = ['At Risk', 'About to Sleep', 'Need Attention']
        at_risk = rfm_df[rfm_df['Segment'].isin(at_risk_segments)]
        
        report.append("\n⚠️ RISK ANALYSIS:")
        report.append(f"• Customers at risk: {len(at_risk):,} ({len(at_risk)/len(rfm_df)*100:.1f}%)")
        report.append(f"• CLV at risk: ${at_risk['CLV'].sum():,.0f}")
        
        # If lifetime models are available
        if summary_df is not None and 'probability_alive' in summary_df.columns:
            high_risk = summary_df[
                (summary_df['CLV_12M'] > summary_df['CLV_12M'].median()) & 
                (summary_df['probability_alive'] < 0.5)
            ]
            report.append(f"• High-value customers with <50% probability alive: {len(high_risk):,}")
            report.append(f"• Potential CLV loss: ${high_risk['CLV_12M'].sum():,.0f}")
        
        # Growth opportunities
        growth_segments = ['New Customers', 'Promising', 'Potential Loyalists']
        growth = rfm_df[rfm_df['Segment'].isin(growth_segments)]
        
        report.append("\n🚀 GROWTH OPPORTUNITIES:")
        report.append(f"• Growth segment customers: {len(growth):,}")
        report.append(f"• Current CLV: ${growth['CLV'].sum():,.0f}")
        report.append(f"• Average purchase frequency: {growth['Frequency'].mean():.1f}")
        
        # Top customers
        report.append("\n🏆 TOP 10 CUSTOMERS:")
        top_10 = rfm_df.nlargest(10, 'CLV')[['User_ID', 'Segment', 'CLV', 'Frequency']]
        for idx, (_, customer) in enumerate(top_10.iterrows(), 1):
            report.append(f"{idx}. ID {customer['User_ID']}: ${customer['CLV']:,.2f} "
                         f"({customer['Segment']}, {customer['Frequency']} purchases)")
        
        return "\n".join(report)
    
    def _create_segment_summary(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create detailed segment summary
        """
        summary = rfm_df.groupby('Segment').agg({
            'User_ID': 'count',
            'CLV': ['sum', 'mean', 'median'],
            'Frequency': 'mean',
            'Total_Purchase': 'mean'
        })
        
        summary.columns = ['Customer_Count', 'Total_CLV', 'Avg_CLV', 
                          'Median_CLV', 'Avg_Frequency', 'Avg_Purchase']
        
        # Add percentages
        summary['Customer_Pct'] = summary['Customer_Count'] / len(rfm_df) * 100
        summary['CLV_Pct'] = summary['Total_CLV'] / rfm_df['CLV'].sum() * 100
        
        return summary.sort_values('Total_CLV', ascending=False)
    
    def generate_action_plan(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate actionable recommendations by segment
        """
        actions = []
        
        for segment in rfm_df['Segment'].unique():
            segment_df = rfm_df[rfm_df['Segment'] == segment]
            
            action = {
                'Segment': segment,
                'Customer_Count': len(segment_df),
                'Total_CLV': segment_df['CLV'].sum(),
                'Avg_CLV': segment_df['CLV'].mean(),
                'Priority': self._get_priority(segment),
                'Recommendation': SEGMENT_RECOMMENDATIONS.get(segment, 'Monitor'),
                'Expected_Impact': self._estimate_impact(segment, segment_df)
            }
            
            actions.append(action)
        
        action_df = pd.DataFrame(actions)
        return action_df.sort_values(['Priority', 'Total_CLV'], ascending=[True, False])
    
    def _get_priority(self, segment: str) -> int:
        """
        Assign priority to segments
        """
        priority_map = {
            'At Risk': 1,
            'About to Sleep': 1,
            'Champions': 2,
            'Need Attention': 2,
            'Loyal Customers': 3,
            'Potential Loyalists': 3,
            'Promising': 4,
            'New Customers': 5
        }
        return priority_map.get(segment, 6)
    
    def _estimate_impact(self, segment: str, segment_df: pd.DataFrame) -> str:
        """
        Estimate potential impact of actions
        """
        if segment in ['At Risk', 'About to Sleep']:
            return f"Prevent ${segment_df['CLV'].sum():,.0f} churn"
        elif segment in ['Potential Loyalists', 'Promising']:
            potential_uplift = segment_df['CLV'].sum() * 0.3  # Assume 30% uplift
            return f"Increase CLV by ${potential_uplift:,.0f}"
        elif segment == 'New Customers':
            return f"Convert {len(segment_df)} new customers"
        else:
            return "Maintain current value"
    
    def generate_kpi_dashboard(self, rfm_df: pd.DataFrame, 
                             summary_df: Optional[pd.DataFrame] = None) -> Dict:
        """
        Generate KPIs for dashboard
        """
        kpis = {
            'overview': {
                'total_customers': len(rfm_df),
                'total_clv': rfm_df['CLV'].sum(),
                'avg_clv': rfm_df['CLV'].mean(),
                'median_clv': rfm_df['CLV'].median(),
                'clv_std': rfm_df['CLV'].std()
            },
            'segments': {},
            'demographics': {},
            'risk_metrics': {},
            'growth_metrics': {}
        }
        
        # Segment KPIs
        for segment in rfm_df['Segment'].unique():
            segment_df = rfm_df[rfm_df['Segment'] == segment]
            kpis['segments'][segment] = {
                'count': len(segment_df),
                'total_clv': segment_df['CLV'].sum(),
                'avg_clv': segment_df['CLV'].mean(),
                'pct_of_total': len(segment_df) / len(rfm_df) * 100
            }
        
        # Demographic KPIs
        for demo in ['Age', 'Gender', 'City_Category']:
            if demo in rfm_df.columns:
                kpis['demographics'][demo] = {}
                for value in rfm_df[demo].unique():
                    demo_df = rfm_df[rfm_df[demo] == value]
                    kpis['demographics'][demo][str(value)] = {
                        'count': len(demo_df),
                        'avg_clv': demo_df['CLV'].mean()
                    }
        
        # Risk metrics
        at_risk = rfm_df[rfm_df['Segment'].isin(['At Risk', 'About to Sleep'])]
        kpis['risk_metrics'] = {
            'at_risk_customers': len(at_risk),
            'at_risk_clv': at_risk['CLV'].sum(),
            'at_risk_pct': len(at_risk) / len(rfm_df) * 100
        }
        
        # Growth metrics
        growth = rfm_df[rfm_df['Segment'].isin(['New Customers', 'Promising', 'Potential Loyalists'])]
        kpis['growth_metrics'] = {
            'growth_customers': len(growth),
            'growth_clv': growth['CLV'].sum(),
            'avg_frequency': growth['Frequency'].mean()
        }
        
        # Lifetime model metrics (if available)
        if summary_df is not None:
            if 'probability_alive' in summary_df.columns:
                kpis['lifetime_metrics'] = {
                    'avg_probability_alive': summary_df['probability_alive'].mean(),
                    'customers_above_50pct': (summary_df['probability_alive'] > 0.5).sum(),
                    'avg_clv_3m': summary_df.get('CLV_3M', pd.Series()).mean(),
                    'avg_clv_6m': summary_df.get('CLV_6M', pd.Series()).mean(),
                    'avg_clv_12m': summary_df.get('CLV_12M', pd.Series()).mean()
                }
        
        return kpis
    
    def export_detailed_report(self, rfm_df: pd.DataFrame, 
                             summary_df: Optional[pd.DataFrame] = None,
                             output_path: str = 'clv_detailed_report.xlsx') -> None:
        """
        Export comprehensive report to Excel
        """
        logger.info(f"Exporting detailed report to {output_path}")
        
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            # Executive Summary (as text)
            summary_text = self.generate_executive_summary(rfm_df, summary_df)
            summary_lines = summary_text.split('\n')
            summary_df_text = pd.DataFrame({'Executive Summary': summary_lines})
            summary_df_text.to_excel(writer, sheet_name='Executive_Summary', index=False)
            
            # Full CLV Analysis
            rfm_df.to_excel(writer, sheet_name='CLV_Analysis', index=False)
            
            # Segment Summary
            segment_summary = self._create_segment_summary(rfm_df)
            segment_summary.to_excel(writer, sheet_name='Segment_Summary')
            
            # Action Plan
            action_plan = self.generate_action_plan(rfm_df)
            action_plan.to_excel(writer, sheet_name='Action_Plan', index=False)
            
            # Top Customers
            top_100 = rfm_df.nlargest(100, 'CLV')[
                ['User_ID', 'Segment', 'CLV', 'Frequency', 'Total_Purchase', 'Avg_Purchase']
            ]
            top_100.to_excel(writer, sheet_name='Top_100_Customers', index=False)
            
            # At Risk Customers
            at_risk = rfm_df[rfm_df['Segment'].isin(['At Risk', 'About to Sleep', 'Need Attention'])]
            at_risk.to_excel(writer, sheet_name='At_Risk_Customers', index=False)
            
            # Demographic Analysis
            demo_analysis = self._create_demographic_analysis(rfm_df)
            demo_analysis.to_excel(writer, sheet_name='Demographic_Analysis')
            
            # Lifetime Model Results (if available)
            if summary_df is not None:
                summary_df.to_excel(writer, sheet_name='Lifetime_Models')
        
        logger.info("Detailed report exported successfully")
    
    def _create_demographic_analysis(self, rfm_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic analysis summary
        """
        demos = []
        
        for col in ['Age', 'Gender', 'City_Category', 'Marital_Status']:
            if col in rfm_df.columns:
                for value in rfm_df[col].unique():
                    demo_df = rfm_df[rfm_df[col] == value]
                    demos.append({
                        'Demographic': col,
                        'Value': value,
                        'Customers': len(demo_df),
                        'Avg_CLV': demo_df['CLV'].mean(),
                        'Total_CLV': demo_df['CLV'].sum(),
                        'Avg_Frequency': demo_df['Frequency'].mean()
                    })
        
        return pd.DataFrame(demos)