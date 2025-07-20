"""
FastAPI application for Customer Lifetime Value (CLV) analysis
"""

import io
import json
import logging
import pandas as pd
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Union
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from main import CLVAnalysisPipeline
from config import CLV_CONFIG

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Customer Lifetime Value Analysis API",
    description="API for running CLV analysis using RFM segmentation and lifetime models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class CLVAnalysisRequest(BaseModel):
    sample_size: Optional[int] = Field(None, description="Number of customers to sample for analysis")
    save_outputs: bool = Field(True, description="Whether to save output files")
    output_directory: str = Field("./api_results", description="Directory to save results")

class CLVAnalysisResponse(BaseModel):
    message: str
    total_customers: int
    avg_basic_clv: Optional[float] = None
    avg_lifetime_clv_12m: Optional[float] = None
    model_r2: Optional[float] = None
    execution_time: float
    output_directory: str

class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str

class ErrorResponse(BaseModel):
    error: str
    detail: str

# Global variable to store analysis results
analysis_results = {}

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {"message": "Customer Lifetime Value Analysis API", "version": "1.0.0"}

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )

@app.post("/analyze", response_model=CLVAnalysisResponse)
async def run_clv_analysis(
    background_tasks: BackgroundTasks,
    request: CLVAnalysisRequest,
    file: Optional[UploadFile] = File(None)
):
    """
    Run CLV analysis with uploaded data or default Kaggle dataset
    """
    start_time = datetime.now()
    
    try:
        # Initialize pipeline
        pipeline = CLVAnalysisPipeline()
        
        # Handle data input
        df = None
        if file:
            # Read uploaded file
            if not file.filename.endswith('.csv'):
                raise HTTPException(status_code=400, detail="Only CSV files are supported")
            
            content = await file.read()
            df = pd.read_csv(io.StringIO(content.decode('utf-8')))
            logger.info(f"Loaded {len(df)} rows from uploaded file: {file.filename}")
        
        # Create output directory
        output_dir = Path(request.output_directory)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run analysis
        results = pipeline.run_complete_analysis(
            df=df,
            sample_size=request.sample_size,
            save_outputs=request.save_outputs,
            output_directory=str(output_dir)
        )
        
        # Store results globally for later retrieval
        global analysis_results
        analysis_results = results
        
        # Calculate execution time
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Extract key metrics for response
        total_customers = len(results.get('rfm_df', pd.DataFrame()))
        avg_basic_clv = None
        avg_lifetime_clv_12m = None
        model_r2 = None
        
        if 'rfm_df' in results and not results['rfm_df'].empty:
            rfm_df = results['rfm_df']
            if 'CLV' in rfm_df.columns:
                avg_basic_clv = float(rfm_df['CLV'].mean())
        
        if 'summary_df' in results and not results['summary_df'].empty:
            summary_df = results['summary_df']
            if 'CLV_12M' in summary_df.columns:
                avg_lifetime_clv_12m = float(summary_df['CLV_12M'].mean())
        
        if 'model_metrics' in results and 'r2' in results['model_metrics']:
            model_r2 = float(results['model_metrics']['r2'])
        
        return CLVAnalysisResponse(
            message="CLV analysis completed successfully",
            total_customers=total_customers,
            avg_basic_clv=avg_basic_clv,
            avg_lifetime_clv_12m=avg_lifetime_clv_12m,
            model_r2=model_r2,
            execution_time=execution_time,
            output_directory=str(output_dir.absolute())
        )
        
    except Exception as e:
        logger.error(f"Error in CLV analysis: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/results", response_model=Dict)
async def get_analysis_results():
    """
    Get the latest analysis results
    """
    if not analysis_results:
        raise HTTPException(status_code=404, detail="No analysis results available. Run analysis first.")
    
    # Convert DataFrames to dict for JSON serialization
    serializable_results = {}
    for key, value in analysis_results.items():
        if isinstance(value, pd.DataFrame):
            serializable_results[key] = value.to_dict('records')
        else:
            serializable_results[key] = value
    
    return serializable_results

@app.get("/results/executive-summary")
async def get_executive_summary():
    """
    Get executive summary from the latest analysis
    """
    if not analysis_results:
        raise HTTPException(status_code=404, detail="No analysis results available. Run analysis first.")
    
    try:
        from reporting import CLVReporter
        reporter = CLVReporter()
        
        rfm_df = analysis_results.get('rfm_df')
        summary_df = analysis_results.get('summary_df')
        
        if rfm_df is None or rfm_df.empty:
            raise HTTPException(status_code=404, detail="No RFM data available in results")
        
        summary = reporter.generate_executive_summary(rfm_df, summary_df)
        return {"executive_summary": summary}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate executive summary: {str(e)}")

@app.get("/results/download/{file_type}")
async def download_results(file_type: str):
    """
    Download specific result files
    """
    if not analysis_results:
        raise HTTPException(status_code=404, detail="No analysis results available. Run analysis first.")
    
    # Map file types to expected file names
    file_map = {
        "executive_summary": "executive_summary.txt",
        "detailed_report": "clv_detailed_report.xlsx",
        "action_plan": "action_plan.csv",
        "kpi_dashboard": "kpi_dashboard.json",
        "feature_importance": "feature_importance.csv"
    }
    
    if file_type not in file_map:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type. Available types: {list(file_map.keys())}"
        )
    
    # Look for the file in common output directories
    possible_paths = [
        Path("./api_results") / file_map[file_type],
        Path("./clv_results_kaggle") / file_map[file_type],
        Path(".") / file_map[file_type]
    ]
    
    for file_path in possible_paths:
        if file_path.exists():
            return FileResponse(
                path=str(file_path),
                filename=file_map[file_type],
                media_type='application/octet-stream'
            )
    
    raise HTTPException(status_code=404, detail=f"File {file_map[file_type]} not found")

@app.get("/config")
async def get_config():
    """
    Get current CLV analysis configuration
    """
    return CLV_CONFIG

@app.get("/favicon.ico")
async def favicon():
    """Return empty response for favicon requests"""
    return Response(status_code=204)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)