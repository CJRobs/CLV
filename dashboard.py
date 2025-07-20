"""
Streamlit Dashboard for Customer Lifetime Value Analysis
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime
from pathlib import Path

from main import CLVAnalysisPipeline
from config import CLV_CONFIG

# Page configuration
st.set_page_config(
    page_title="CLV Analysis Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background-color: #f0f2f6;
    }
    .stAlert {
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    if 'results' not in st.session_state:
        st.session_state.results = None
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = None

def create_summary_metrics(rfm_df, summary_df=None):
    """Create summary metrics display"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_customers = len(rfm_df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    with col2:
        if 'CLV' in rfm_df.columns:
            avg_basic_clv = rfm_df['CLV'].mean()
            st.metric("Avg Basic CLV", f"${avg_basic_clv:,.2f}")
        else:
            st.metric("Avg Basic CLV", "N/A")
    
    with col3:
        if summary_df is not None and 'CLV_12M' in summary_df.columns:
            avg_lifetime_clv = summary_df['CLV_12M'].mean()
            st.metric("Avg 12M Lifetime CLV", f"${avg_lifetime_clv:,.2f}")
        else:
            st.metric("Avg 12M Lifetime CLV", "N/A")
    
    with col4:
        if 'CLV' in rfm_df.columns:
            total_clv = rfm_df['CLV'].sum()
            st.metric("Total CLV", f"${total_clv:,.0f}")
        else:
            st.metric("Total CLV", "N/A")

def create_rfm_distribution_chart(rfm_df):
    """Create RFM distribution charts"""
    # Determine available columns for subplot titles
    subplot_titles = []
    has_recency = 'Recency' in rfm_df.columns
    has_frequency = 'Frequency' in rfm_df.columns
    has_monetary = 'Monetary' in rfm_df.columns
    has_segments = 'RFM_Segment' in rfm_df.columns
    
    # Build subplot titles based on available data
    subplot_titles = [
        'Recency Distribution' if has_recency else 'No Recency Data',
        'Frequency Distribution' if has_frequency else 'No Frequency Data',
        'Monetary Distribution' if has_monetary else 'No Monetary Data',
        'Customer Segments' if has_segments else 'No Segment Data'
    ]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Recency distribution
    if has_recency:
        fig.add_trace(
            go.Histogram(x=rfm_df['Recency'], name='Recency', nbinsx=30),
            row=1, col=1
        )
    
    # Frequency distribution
    if has_frequency:
        fig.add_trace(
            go.Histogram(x=rfm_df['Frequency'], name='Frequency', nbinsx=30),
            row=1, col=2
        )
    
    # Monetary distribution
    if has_monetary:
        fig.add_trace(
            go.Histogram(x=rfm_df['Monetary'], name='Monetary', nbinsx=30),
            row=2, col=1
        )
    
    # Customer segments
    if has_segments:
        segment_counts = rfm_df['RFM_Segment'].value_counts()
        fig.add_trace(
            go.Bar(x=segment_counts.index, y=segment_counts.values, name='Segments'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="RFM Analysis Overview")
    return fig

def create_clv_analysis_chart(rfm_df, summary_df=None):
    """Create CLV analysis charts"""
    # Check what data is available
    has_clv = 'CLV' in rfm_df.columns
    has_segments = 'RFM_Segment' in rfm_df.columns
    has_rfm_score = 'RFM_Score' in rfm_df.columns
    has_lifetime_data = summary_df is not None and not summary_df.empty
    
    subplot_titles = [
        'CLV Distribution' if has_clv else 'No CLV Data',
        'CLV by Segment' if (has_clv and has_segments) else 'No Segment CLV Data',
        'CLV vs RFM Score' if (has_clv and has_rfm_score) else 'No RFM Score Data',
        'Lifetime CLV Timeline' if has_lifetime_data else 'No Lifetime Data'
    ]
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=subplot_titles,
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # CLV Distribution
    if has_clv:
        fig.add_trace(
            go.Histogram(x=rfm_df['CLV'], name='CLV Distribution', nbinsx=50),
            row=1, col=1
        )
    
    # CLV by Segment
    if has_clv and has_segments:
        clv_by_segment = rfm_df.groupby('RFM_Segment')['CLV'].mean().sort_values(ascending=True)
        fig.add_trace(
            go.Bar(x=clv_by_segment.values, y=clv_by_segment.index, 
                   orientation='h', name='Avg CLV by Segment'),
            row=1, col=2
        )
    
    # CLV vs RFM Score
    if has_clv and has_rfm_score:
        fig.add_trace(
            go.Scatter(x=rfm_df['RFM_Score'], y=rfm_df['CLV'], 
                      mode='markers', name='CLV vs RFM', opacity=0.6),
            row=2, col=1
        )
    
    # Lifetime CLV Timeline
    if has_lifetime_data:
        clv_cols = [col for col in summary_df.columns if 'CLV_' in col and 'M' in col]
        if clv_cols:
            periods = [col.replace('CLV_', '').replace('M', '') for col in clv_cols]
            avg_clvs = [summary_df[col].mean() for col in clv_cols]
            fig.add_trace(
                go.Scatter(x=periods, y=avg_clvs, mode='lines+markers', 
                          name='Avg Lifetime CLV'),
                row=2, col=2
            )
    
    fig.update_layout(height=600, showlegend=False, title_text="Customer Lifetime Value Analysis")
    return fig

def create_segment_analysis_table(rfm_df):
    """Create detailed segment analysis table"""
    if 'RFM_Segment' not in rfm_df.columns:
        return pd.DataFrame()
    
    # Build aggregation dictionary based on available columns
    agg_dict = {'User_ID': 'count'}
    column_names = ['Customer Count']
    
    if 'Recency' in rfm_df.columns:
        agg_dict['Recency'] = 'mean'
        column_names.append('Avg Recency')
    
    if 'Frequency' in rfm_df.columns:
        agg_dict['Frequency'] = 'mean'
        column_names.append('Avg Frequency')
    
    if 'Monetary' in rfm_df.columns:
        agg_dict['Monetary'] = 'mean'
        column_names.append('Avg Monetary')
    
    if 'CLV' in rfm_df.columns:
        agg_dict['CLV'] = 'mean'
        column_names.append('Avg CLV')
    
    segment_analysis = rfm_df.groupby('RFM_Segment').agg(agg_dict).round(2)
    segment_analysis.columns = column_names
    segment_analysis['Customer %'] = (segment_analysis['Customer Count'] / segment_analysis['Customer Count'].sum() * 100).round(1)
    
    return segment_analysis.reset_index()

def main():
    """Main dashboard function"""
    initialize_session_state()
    
    # Header
    st.markdown('<h1 class="main-header">📊 Customer Lifetime Value Analysis Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Data input options
        st.subheader("📁 Data Input")
        data_option = st.radio(
            "Choose data source:",
            ["Upload CSV File", "Use Sample Dataset"]
        )
        
        uploaded_file = None
        if data_option == "Upload CSV File":
            uploaded_file = st.file_uploader(
                "Choose a CSV file",
                type=['csv'],
                help="Upload your transaction data CSV file"
            )
        
        # Analysis parameters
        st.subheader("🔧 Analysis Parameters")
        sample_size = st.number_input(
            "Sample Size (customers)",
            min_value=100,
            max_value=10000,
            value=2000,
            help="Number of customers to analyze (for performance)"
        )
        
        # Run analysis button
        run_analysis = st.button("🚀 Run CLV Analysis", type="primary")
    
    # Main content area
    if run_analysis:
        with st.spinner("🔄 Running CLV Analysis... This may take a few moments."):
            try:
                # Initialize pipeline
                pipeline = CLVAnalysisPipeline()
                
                # Load data
                df = None
                if uploaded_file is not None:
                    df = pd.read_csv(uploaded_file)
                    st.success(f"✅ Loaded {len(df):,} transactions from uploaded file")
                
                # Run analysis
                results = pipeline.run_complete_analysis(
                    df=df,
                    sample_size=sample_size,
                    save_outputs=False,  # Don't save files for dashboard
                    output_directory='./dashboard_results'
                )
                
                # Store results in session state
                st.session_state.results = results
                st.session_state.pipeline = pipeline
                st.session_state.analysis_complete = True
                
                st.success("🎉 Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Error running analysis: {str(e)}")
                return
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete and st.session_state.results:
        results = st.session_state.results
        rfm_df = results.get('rfm_df')
        summary_df = results.get('summary_df')
        
        if rfm_df is not None and not rfm_df.empty:
            # Summary metrics
            st.header("📈 Key Metrics")
            create_summary_metrics(rfm_df, summary_df)
            
            # Main visualizations
            st.header("📊 Analysis Results")
            
            # Create tabs for different views
            tab1, tab2, tab3, tab4 = st.tabs(["🎯 RFM Analysis", "💰 CLV Analysis", "👥 Segment Details", "📋 Executive Summary"])
            
            with tab1:
                st.subheader("RFM Distribution and Segmentation")
                rfm_chart = create_rfm_distribution_chart(rfm_df)
                st.plotly_chart(rfm_chart, use_container_width=True)
                
                # RFM data table
                st.subheader("📋 RFM Data Sample")
                display_cols = ['User_ID', 'Recency', 'Frequency', 'Monetary', 'RFM_Score', 'RFM_Segment']
                available_cols = [col for col in display_cols if col in rfm_df.columns]
                st.dataframe(rfm_df[available_cols].head(10), use_container_width=True)
            
            with tab2:
                st.subheader("Customer Lifetime Value Analysis")
                clv_chart = create_clv_analysis_chart(rfm_df, summary_df)
                st.plotly_chart(clv_chart, use_container_width=True)
                
                # Top customers by CLV
                if 'CLV' in rfm_df.columns:
                    st.subheader("🏆 Top 10 Customers by CLV")
                    # Only include columns that actually exist
                    top_customer_cols = ['User_ID', 'CLV']
                    optional_cols = ['RFM_Segment', 'Recency', 'Frequency', 'Monetary']
                    for col in optional_cols:
                        if col in rfm_df.columns:
                            top_customer_cols.append(col)
                    
                    top_customers = rfm_df.nlargest(10, 'CLV')[top_customer_cols]
                    st.dataframe(top_customers, use_container_width=True)
            
            with tab3:
                st.subheader("Customer Segment Analysis")
                segment_table = create_segment_analysis_table(rfm_df)
                if not segment_table.empty:
                    st.dataframe(segment_table, use_container_width=True)
                
                # Segment recommendations
                st.subheader("💡 Segment Recommendations")
                recommendations = {
                    "Champions": "Reward them with exclusive offers and early access to new products",
                    "Loyal Customers": "Upsell higher value products and ask for referrals",
                    "Potential Loyalists": "Offer membership or loyalty programs",
                    "New Customers": "Provide on-boarding support and education",
                    "Promising": "Offer free trials and special promotions",
                    "Need Attention": "Make limited time offers and recommend products",
                    "About to Sleep": "Share valuable resources and recommend popular products",
                    "At Risk": "Send personalized emails and offer discounts",
                    "Cannot Lose Them": "Win them back via renewals or newer products",
                    "Hibernating": "Offer other product categories and special discounts",
                    "Lost": "Revive interest with reach out campaign and ignore otherwise"
                }
                
                for segment, recommendation in recommendations.items():
                    if not segment_table.empty and 'RFM_Segment' in segment_table.columns and segment in segment_table['RFM_Segment'].values:
                        st.write(f"**{segment}**: {recommendation}")
            
            with tab4:
                st.subheader("Executive Summary")
                try:
                    from reporting import CLVReporter
                    reporter = CLVReporter()
                    exec_summary = reporter.generate_executive_summary(rfm_df, summary_df)
                    st.text_area("Summary Report", exec_summary, height=400)
                except Exception as e:
                    st.error(f"Could not generate executive summary: {str(e)}")
                
                # Download options
                st.subheader("📥 Download Results")
                col1, col2 = st.columns(2)
                
                with col1:
                    if st.button("📊 Download RFM Data"):
                        csv = rfm_df.to_csv(index=False)
                        st.download_button(
                            label="Download CSV",
                            data=csv,
                            file_name=f"rfm_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                
                with col2:
                    if summary_df is not None and not summary_df.empty:
                        if st.button("💰 Download Lifetime CLV Data"):
                            csv = summary_df.to_csv(index=False)
                            st.download_button(
                                label="Download CSV",
                                data=csv,
                                file_name=f"lifetime_clv_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv"
                            )
        else:
            st.error("❌ No valid results to display. Please check your data and try again.")
    
    else:
        # Welcome message when no analysis has been run
        st.info("👈 Configure your analysis parameters in the sidebar and click 'Run CLV Analysis' to get started!")
        
        # Show sample data format
        st.subheader("📋 Expected Data Format")
        sample_data = pd.DataFrame({
            'User_ID': [1001, 1002, 1003, 1004],
            'Product_ID': ['P001', 'P002', 'P001', 'P003'],
            'Purchase': [150.00, 299.99, 89.50, 450.00],
            'Gender': ['M', 'F', 'M', 'F'],
            'Age': [25, 32, 28, 45]
        })
        st.dataframe(sample_data, use_container_width=True)
        
        st.markdown("""
        **Required columns:**
        - `User_ID`: Unique customer identifier
        - `Purchase`: Transaction amount
        
        **Optional columns:**
        - `Product_ID`: Product identifier
        - `Gender`: Customer gender
        - `Age`: Customer age
        - Any other customer attributes
        """)

if __name__ == "__main__":
    main()