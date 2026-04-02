import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
import streamlit as st

class PerformanceVisualizer:
    """Generate graphs for performance visualization"""
    
    def __init__(self):
        # Set style for better looking graphs
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
    
    def create_bar_graph(self, metrics: dict) -> plt.Figure:
        """
        Create bar graph showing performance metrics
        Required by faculty: Bar Graph
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        metric_values = [
            metrics.get('accuracy', 0),
            metrics.get('precision', 0),
            metrics.get('recall', 0),
            metrics.get('f1_score', 0)
        ]
        
        # Define colors based on values
        colors = []
        for val in metric_values:
            if val >= 0.8:
                colors.append('#28a745')  # Green - Excellent
            elif val >= 0.6:
                colors.append('#ffc107')  # Yellow - Good
            else:
                colors.append('#dc3545')  # Red - Needs improvement
        
        # Create bars
        bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=1.5)
        
        # Add value labels on top of bars
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Customize the graph
        ax.set_ylim(0, 1.1)
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('📊 Model Performance Metrics', fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add horizontal line at 0.7 (good threshold)
        ax.axhline(y=0.7, color='gray', linestyle='--', alpha=0.5, label='Good Threshold (0.7)')
        ax.legend()
        
        plt.tight_layout()
        return fig
    
    def create_confusion_matrix_heatmap(self, y_true: list, y_pred: list) -> plt.Figure:
        """
        Create heatmap of confusion matrix
        Required by faculty: Heatmap
        """
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['BAD', 'GOOD'],
                   yticklabels=['BAD', 'GOOD'],
                   ax=ax, cbar_kws={'label': 'Count'})
        
        # Customize
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        ax.set_title('🔍 Confusion Matrix Heatmap', fontsize=16, fontweight='bold', pad=20)
        
        # Add annotations
        for i in range(2):
            for j in range(2):
                if cm[i, j] > 0:
                    ax.text(j+0.5, i+0.5, f'{cm[i, j]}', 
                           ha='center', va='center', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig
    
    def create_radar_plot(self, metrics: dict) -> plt.Figure:
        """
        Create radar plot for multi-dimensional comparison
        Optional but recommended by faculty
        """
        # Prepare data
        categories = ['Semantic\nUnderstanding', 'Keyword\nMatching', 
                     'Answer\nLength', 'Structure\nQuality', 'Example\nUsage']
        
        # Get values or use defaults
        values = [
            metrics.get('semantic_score', 0) / 100,
            metrics.get('keyword_score', 0) / 100,
            metrics.get('length_score', 0) / 100,
            metrics.get('structure_score', 0) / 100,
            metrics.get('example_score', 0) / 100
        ]
        
        # Close the loop for radar chart
        values += values[:1]
        angles = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
        angles += angles[:1]
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
        
        # Plot the radar
        ax.plot(angles, values, 'o-', linewidth=2, color='#667eea', label='Current Performance')
        ax.fill(angles, values, alpha=0.25, color='#667eea')
        
        # Set category labels
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=10)
        
        # Set y-axis limits
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['20%', '40%', '60%', '80%', '100%'], fontsize=8)
        
        # Add title
        ax.set_title('🎯 Multi-dimensional Performance Analysis', fontsize=14, fontweight='bold', pad=20)
        
        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        
        plt.tight_layout()
        return fig
    
    def create_performance_comparison(self, metrics_before: dict, metrics_after: dict) -> plt.Figure:
        """
        Create comparison bar graph (before vs after threshold adjustment)
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
        before_values = [
            metrics_before.get('accuracy', 0),
            metrics_before.get('precision', 0),
            metrics_before.get('recall', 0),
            metrics_before.get('f1_score', 0)
        ]
        after_values = [
            metrics_after.get('accuracy', 0),
            metrics_after.get('precision', 0),
            metrics_after.get('recall', 0),
            metrics_after.get('f1_score', 0)
        ]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, before_values, width, label='Before Optimization', color='#ff6b6b')
        bars2 = ax.bar(x + width/2, after_values, width, label='After Optimization', color='#51cf66')
        
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('📈 Performance Improvement After Threshold Tuning', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories)
        ax.legend()
        ax.set_ylim(0, 1.1)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Add value labels
        for bar in bars1:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        for bar in bars2:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        return fig
    
    def display_all_graphs(self, y_true, y_pred, metrics, component_scores=None):
        """
        Display all graphs in Streamlit
        """
        st.subheader("📊 Performance Visualization Dashboard")
        
        # Row 1: Bar Graph and Heatmap
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### 📊 Bar Graph")
            bar_fig = self.create_bar_graph(metrics)
            st.pyplot(bar_fig)
            st.caption("Figure 1: Performance metrics bar chart showing accuracy, precision, recall, and F1 score")
        
        with col2:
            st.markdown("### 🔥 Confusion Matrix Heatmap")
            heatmap_fig = self.create_confusion_matrix_heatmap(y_true, y_pred)
            st.pyplot(heatmap_fig)
            st.caption("Figure 2: Heatmap showing correct vs incorrect classifications")
        
        # Row 2: Radar Plot
        st.markdown("---")
        st.markdown("### 🎯 Radar Plot (Multi-dimensional Analysis)")
        
        if component_scores:
            radar_fig = self.create_radar_plot(component_scores)
        else:
            # Use metrics as fallback
            radar_fig = self.create_radar_plot(metrics)
        
        st.pyplot(radar_fig)
        st.caption("Figure 3: Radar plot showing performance across multiple dimensions")
        
        # Optional: Comparison graph if before/after data available
        return bar_fig, heatmap_fig, radar_fig
    
    def save_graphs(self, y_true, y_pred, metrics, filename_prefix="performance"):
        """
        Save graphs as image files (for report)
        """
        # Bar graph
        bar_fig = self.create_bar_graph(metrics)
        bar_fig.savefig(f"{filename_prefix}_bar.png", dpi=300, bbox_inches='tight')
        
        # Heatmap
        heatmap_fig = self.create_confusion_matrix_heatmap(y_true, y_pred)
        heatmap_fig.savefig(f"{filename_prefix}_heatmap.png", dpi=300, bbox_inches='tight')
        
        # Radar plot
        radar_fig = self.create_radar_plot(metrics)
        radar_fig.savefig(f"{filename_prefix}_radar.png", dpi=300, bbox_inches='tight')
        
        print(f"✅ Graphs saved as {filename_prefix}_bar.png, {filename_prefix}_heatmap.png, {filename_prefix}_radar.png")