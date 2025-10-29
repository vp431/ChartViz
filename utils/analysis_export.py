"""
Analysis Export Utilities - Export analysis results in various formats.
Provides comprehensive export functionality for analysis results.
"""
import json
import csv
import io
import base64
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from PIL import Image

logger = logging.getLogger(__name__)


class AnalysisExporter:
    """Comprehensive analysis results exporter."""
    
    def __init__(self):
        """Initialize the analysis exporter."""
        self.supported_formats = ['json', 'csv', 'png', 'pdf', 'html']
    
    def export_analysis_results(self, analysis_result: Dict[str, Any], 
                               export_format: str = 'json',
                               filename: Optional[str] = None) -> Dict[str, Any]:
        """
        Export analysis results in the specified format.
        
        Args:
            analysis_result: Analysis results dictionary
            export_format: Export format ('json', 'csv', 'png', 'pdf', 'html')
            filename: Optional filename (auto-generated if None)
            
        Returns:
            Dictionary with export status and data
        """
        if export_format not in self.supported_formats:
            raise ValueError(f"Unsupported export format: {export_format}")
        
        # Generate filename if not provided
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            analysis_type = analysis_result.get('analysis_type', 'analysis')
            filename = f"{analysis_type}_{timestamp}"
        
        try:
            if export_format == 'json':
                return self._export_json(analysis_result, filename)
            elif export_format == 'csv':
                return self._export_csv(analysis_result, filename)
            elif export_format == 'png':
                return self._export_png(analysis_result, filename)
            elif export_format == 'pdf':
                return self._export_pdf(analysis_result, filename)
            elif export_format == 'html':
                return self._export_html(analysis_result, filename)
            
        except Exception as e:
            logger.error(f"Export failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'format': export_format,
                'filename': filename
            }
    
    def _export_json(self, analysis_result: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export analysis results as JSON."""
        
        # Prepare data for JSON serialization
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'export_format': 'json',
                'analysis_type': analysis_result.get('analysis_type', 'unknown'),
                'model_id': analysis_result.get('model_specific', 'unknown')
            },
            'analysis_results': {
                'title': analysis_result.get('title', ''),
                'description': analysis_result.get('description', ''),
                'metrics': analysis_result.get('metrics', {}),
                'insights': analysis_result.get('insights', [])
            }
        }
        
        # Convert to JSON string
        json_data = json.dumps(export_data, indent=2, default=str)
        
        # Encode as base64 for download
        json_bytes = json_data.encode('utf-8')
        json_b64 = base64.b64encode(json_bytes).decode('utf-8')
        
        return {
            'success': True,
            'format': 'json',
            'filename': f"{filename}.json",
            'data': json_b64,
            'mime_type': 'application/json',
            'size': len(json_bytes)
        }
    
    def _export_csv(self, analysis_result: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export analysis results as CSV."""
        
        # Prepare data for CSV
        csv_data = []
        
        # Add metadata
        csv_data.append(['Metadata', '', ''])
        csv_data.append(['Export Timestamp', datetime.now().isoformat(), ''])
        csv_data.append(['Analysis Type', analysis_result.get('analysis_type', 'unknown'), ''])
        csv_data.append(['Model', analysis_result.get('model_specific', 'unknown'), ''])
        csv_data.append(['', '', ''])
        
        # Add metrics
        if analysis_result.get('metrics'):
            csv_data.append(['Metrics', 'Value', 'Description'])
            for key, value in analysis_result['metrics'].items():
                description = self._get_metric_description(key)
                csv_data.append([key, str(value), description])
            csv_data.append(['', '', ''])
        
        # Add insights
        if analysis_result.get('insights'):
            csv_data.append(['Insights', '', ''])
            for i, insight in enumerate(analysis_result['insights'], 1):
                csv_data.append([f'Insight {i}', insight, ''])
        
        # Convert to CSV string
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerows(csv_data)
        csv_string = output.getvalue()
        output.close()
        
        # Encode as base64 for download
        csv_bytes = csv_string.encode('utf-8')
        csv_b64 = base64.b64encode(csv_bytes).decode('utf-8')
        
        return {
            'success': True,
            'format': 'csv',
            'filename': f"{filename}.csv",
            'data': csv_b64,
            'mime_type': 'text/csv',
            'size': len(csv_bytes)
        }
    
    def _export_png(self, analysis_result: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export analysis visualization as PNG."""
        
        if not analysis_result.get('visualization_data'):
            return {
                'success': False,
                'error': 'No visualization data available for PNG export',
                'format': 'png',
                'filename': filename
            }
        
        try:
            # Convert visualization data to plotly figure
            fig_dict = analysis_result['visualization_data']
            fig = go.Figure(fig_dict)
            
            # Update layout for better export
            fig.update_layout(
                width=1200,
                height=800,
                title_font_size=20,
                font_size=12
            )
            
            # Export as PNG
            png_bytes = pio.to_image(fig, format='png', width=1200, height=800)
            png_b64 = base64.b64encode(png_bytes).decode('utf-8')
            
            return {
                'success': True,
                'format': 'png',
                'filename': f"{filename}.png",
                'data': png_b64,
                'mime_type': 'image/png',
                'size': len(png_bytes)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'PNG export failed: {str(e)}',
                'format': 'png',
                'filename': filename
            }
    
    def _export_pdf(self, analysis_result: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export analysis results as PDF report."""
        
        try:
            # Create HTML content for PDF conversion
            html_content = self._generate_html_report(analysis_result)
            
            # For now, return HTML content as PDF is complex to implement
            # In a full implementation, you would use libraries like weasyprint or reportlab
            html_bytes = html_content.encode('utf-8')
            html_b64 = base64.b64encode(html_bytes).decode('utf-8')
            
            return {
                'success': True,
                'format': 'html',  # Returning HTML instead of PDF for now
                'filename': f"{filename}_report.html",
                'data': html_b64,
                'mime_type': 'text/html',
                'size': len(html_bytes),
                'note': 'PDF export returned as HTML report'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'PDF export failed: {str(e)}',
                'format': 'pdf',
                'filename': filename
            }
    
    def _export_html(self, analysis_result: Dict[str, Any], filename: str) -> Dict[str, Any]:
        """Export analysis results as HTML report."""
        
        try:
            html_content = self._generate_html_report(analysis_result)
            
            # Encode as base64 for download
            html_bytes = html_content.encode('utf-8')
            html_b64 = base64.b64encode(html_bytes).decode('utf-8')
            
            return {
                'success': True,
                'format': 'html',
                'filename': f"{filename}.html",
                'data': html_b64,
                'mime_type': 'text/html',
                'size': len(html_bytes)
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': f'HTML export failed: {str(e)}',
                'format': 'html',
                'filename': filename
            }
    
    def _generate_html_report(self, analysis_result: Dict[str, Any]) -> str:
        """Generate HTML report from analysis results."""
        
        title = analysis_result.get('title', 'Analysis Report')
        description = analysis_result.get('description', '')
        metrics = analysis_result.get('metrics', {})
        insights = analysis_result.get('insights', [])
        
        html_content = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>{title}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                    line-height: 1.6;
                }}
                .header {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                    margin-bottom: 30px;
                }}
                .metrics {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                .metric-card {{
                    background-color: #ffffff;
                    border: 1px solid #dee2e6;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                }}
                .metric-value {{
                    font-size: 2em;
                    font-weight: bold;
                    color: #007bff;
                }}
                .metric-label {{
                    color: #6c757d;
                    margin-top: 10px;
                }}
                .insights {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 8px;
                }}
                .insight-item {{
                    background-color: #ffffff;
                    border-left: 4px solid #28a745;
                    padding: 15px;
                    margin-bottom: 10px;
                    border-radius: 0 8px 8px 0;
                }}
                .timestamp {{
                    color: #6c757d;
                    font-size: 0.9em;
                    text-align: right;
                    margin-top: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>{title}</h1>
                <p>{description}</p>
            </div>
        """
        
        # Add metrics section
        if metrics:
            html_content += """
            <h2>Analysis Metrics</h2>
            <div class="metrics">
            """
            
            for key, value in metrics.items():
                formatted_value = f"{value:.3f}" if isinstance(value, float) else str(value)
                metric_label = self._get_metric_description(key)
                
                html_content += f"""
                <div class="metric-card">
                    <div class="metric-value">{formatted_value}</div>
                    <div class="metric-label">{metric_label}</div>
                </div>
                """
            
            html_content += "</div>"
        
        # Add insights section
        if insights:
            html_content += """
            <h2>Analysis Insights</h2>
            <div class="insights">
            """
            
            for insight in insights:
                html_content += f"""
                <div class="insight-item">
                    {insight}
                </div>
                """
            
            html_content += "</div>"
        
        # Add visualization if available
        if analysis_result.get('visualization_data'):
            try:
                fig_dict = analysis_result['visualization_data']
                fig = go.Figure(fig_dict)
                
                # Convert to HTML
                fig_html = pio.to_html(fig, include_plotlyjs='cdn', div_id='analysis-chart')
                html_content += f"""
                <h2>Analysis Visualization</h2>
                {fig_html}
                """
            except Exception as e:
                html_content += f"""
                <h2>Analysis Visualization</h2>
                <p>Error loading visualization: {str(e)}</p>
                """
        
        # Add timestamp
        html_content += f"""
            <div class="timestamp">
                Report generated on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            </div>
        </body>
        </html>
        """
        
        return html_content
    
    def _get_metric_description(self, metric_key: str) -> str:
        """Get human-readable description for metric keys."""
        descriptions = {
            'attention_mean': 'Average Attention',
            'attention_std': 'Attention Standard Deviation',
            'attention_max': 'Maximum Attention',
            'attention_min': 'Minimum Attention',
            'attention_entropy': 'Attention Entropy',
            'attention_gini': 'Gini Coefficient (Concentration)',
            'attention_sparsity': 'Sparsity Ratio',
            'total_attention': 'Total Attention',
            'chart_attention_ratio': 'Chart Focus Ratio',
            'flow_variance': 'Flow Variance',
            'entropy_diversity': 'Entropy Diversity',
            'concentration_diversity': 'Concentration Diversity'
        }
        
        return descriptions.get(metric_key, metric_key.replace('_', ' ').title())
    
    def create_download_link(self, export_result: Dict[str, Any]) -> str:
        """Create a download link for exported data."""
        if not export_result.get('success'):
            return ""
        
        data = export_result['data']
        filename = export_result['filename']
        mime_type = export_result['mime_type']
        
        return f"data:{mime_type};base64,{data}"


# Global exporter instance
analysis_exporter = AnalysisExporter()





