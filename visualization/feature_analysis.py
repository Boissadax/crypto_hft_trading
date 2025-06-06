"""
Feature Analysis Visualizer

Creates comprehensive visualizations for feature engineering analysis:
- Feature distribution histograms and box plots
- Correlation heatmaps between features
- Feature evolution over time
- Feature importance across different time windows
- PCA/t-SNE dimensionality reduction plots
- Feature stability analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

class FeatureAnalysisVisualizer:
    """Visualize feature engineering results and feature analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8), dpi: int = 100):
        """
        Initialize visualizer.
        
        Args:
            figsize: Default figure size
            dpi: Figure DPI for quality
        """
        self.figsize = figsize
        self.dpi = dpi
        self.colors = sns.color_palette("husl", n_colors=10)
    
    def plot_feature_distributions(self,
                                 df_features: pd.DataFrame,
                                 feature_subset: Optional[List[str]] = None,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature distributions.
        
        Args:
            df_features: Features DataFrame
            feature_subset: Subset of features to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        features = feature_subset or df_features.columns[:12]  # Limit to 12 features
        n_features = len(features)
        
        # Calculate grid size
        n_cols = 4
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows), dpi=self.dpi)
        fig.suptitle('📊 Feature Distributions', fontsize=16, fontweight='bold')
        
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        for idx, feature in enumerate(features):
            row, col = idx // n_cols, idx % n_cols
            ax = axes[row, col]
            
            if feature in df_features.columns:
                data = df_features[feature].dropna()
                
                # Plot histogram
                ax.hist(data, bins=50, alpha=0.7, color=self.colors[idx % len(self.colors)], 
                       edgecolor='black')
                ax.set_title(f'{feature}')
                ax.set_ylabel('Frequency')
                ax.grid(True, alpha=0.3)
                
                # Add statistics
                mean_val = data.mean()
                std_val = data.std()
                ax.axvline(mean_val, color='red', linestyle='--', 
                          label=f'μ={mean_val:.3f}')
                ax.axvline(mean_val + std_val, color='orange', linestyle=':', alpha=0.7)
                ax.axvline(mean_val - std_val, color='orange', linestyle=':', alpha=0.7)
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f'Feature {feature}\nnot found', 
                       ha='center', va='center', transform=ax.transAxes)
        
        # Hide empty subplots
        for idx in range(n_features, n_rows * n_cols):
            row, col = idx // n_cols, idx % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"📁 Feature distributions saved to: {save_path}")
        
        return fig
    
    def plot_correlation_heatmap(self,
                               df_features: pd.DataFrame,
                               feature_subset: Optional[List[str]] = None,
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation heatmap between features.
        
        Args:
            df_features: Features DataFrame
            feature_subset: Subset of features to include
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        features = feature_subset or df_features.select_dtypes(include=[np.number]).columns[:20]
        correlation_data = df_features[features].corr()
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=self.dpi)
        fig.suptitle('🔥 Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        
        # Create heatmap
        sns.heatmap(correlation_data, annot=True, cmap='RdYlBu_r', center=0,
                   square=True, fmt='.2f', cbar_kws={"shrink": .8}, ax=ax)
        
        ax.set_title('Correlation Matrix')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"📁 Correlation heatmap saved to: {save_path}")
        
        return fig
    
    def plot_feature_evolution(self,
                             df_features: pd.DataFrame,
                             features: List[str],
                             window_size: int = 1000,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature evolution over time.
        
        Args:
            df_features: Features DataFrame with timestamp index
            features: Features to plot evolution
            window_size: Rolling window size for smoothing
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        fig.suptitle('📈 Feature Evolution Over Time', fontsize=16, fontweight='bold')
        
        # Convert timestamp index to seconds for plotting
        if df_features.index.dtype == np.int64:
            time_seconds = (df_features.index - df_features.index[0]) / 1_000_000
        else:
            time_seconds = np.arange(len(df_features))
        
        for idx, feature in enumerate(features[:4]):  # Limit to 4 features
            ax = axes[idx // 2, idx % 2]
            
            if feature in df_features.columns:
                data = df_features[feature].fillna(method='ffill')
                
                # Plot raw data
                ax.plot(time_seconds, data, alpha=0.3, color=self.colors[idx], 
                       label='Raw', linewidth=0.5)
                
                # Plot rolling mean
                rolling_mean = data.rolling(window=window_size, min_periods=1).mean()
                ax.plot(time_seconds, rolling_mean, color=self.colors[idx], 
                       label=f'Rolling Mean ({window_size})', linewidth=2)
                
                ax.set_title(f'{feature}')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, f'Feature {feature}\nnot found', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"📁 Feature evolution saved to: {save_path}")
        
        return fig
    
    def plot_pca_analysis(self,
                        df_features: pd.DataFrame,
                        n_components: int = 2,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot PCA analysis of features.
        
        Args:
            df_features: Features DataFrame
            n_components: Number of PCA components
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Select numeric features only
        numeric_features = df_features.select_dtypes(include=[np.number]).dropna()
        
        if len(numeric_features.columns) < 2:
            print("⚠️ Not enough numeric features for PCA analysis")
            return plt.figure()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(numeric_features)
        
        # Apply PCA
        pca = PCA(n_components=min(n_components, len(numeric_features.columns)))
        features_pca = pca.fit_transform(features_scaled)
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6), dpi=self.dpi)
        fig.suptitle('🎯 PCA Feature Analysis', fontsize=16, fontweight='bold')
        
        # 1. PCA scatter plot
        ax1 = axes[0]
        scatter = ax1.scatter(features_pca[:, 0], features_pca[:, 1], 
                            alpha=0.6, c=range(len(features_pca)), 
                            cmap='viridis', s=10)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        ax1.set_title('📊 PCA Scatter Plot')
        ax1.grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=ax1, label='Sample Index')
        
        # 2. Explained variance
        ax2 = axes[1]
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        ax2.bar(range(1, len(pca.explained_variance_ratio_) + 1), 
               pca.explained_variance_ratio_, alpha=0.7, 
               color=self.colors[0], label='Individual')
        ax2.plot(range(1, len(cumulative_variance) + 1), 
                cumulative_variance, 'ro-', color=self.colors[1], 
                label='Cumulative')
        
        ax2.set_xlabel('Principal Component')
        ax2.set_ylabel('Explained Variance Ratio')
        ax2.set_title('📈 Explained Variance')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"📁 PCA analysis saved to: {save_path}")
        
        return fig
    
    def plot_feature_stability(self,
                             df_features: pd.DataFrame,
                             features: List[str],
                             window_size: int = 5000,
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature stability over time windows.
        
        Args:
            df_features: Features DataFrame
            features: Features to analyze stability
            window_size: Window size for stability calculation
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=self.dpi)
        fig.suptitle('📊 Feature Stability Analysis', fontsize=16, fontweight='bold')
        
        for idx, feature in enumerate(features[:4]):
            ax = axes[idx // 2, idx % 2]
            
            if feature in df_features.columns:
                data = df_features[feature].dropna()
                
                # Calculate rolling statistics
                rolling_mean = data.rolling(window=window_size).mean()
                rolling_std = data.rolling(window=window_size).std()
                
                # Coefficient of variation (stability metric)
                cv = rolling_std / rolling_mean.abs()
                cv = cv.replace([np.inf, -np.inf], np.nan).dropna()
                
                # Convert index to time
                if data.index.dtype == np.int64:
                    time_seconds = (cv.index - data.index[0]) / 1_000_000
                else:
                    time_seconds = np.arange(len(cv))
                
                ax.plot(time_seconds, cv, color=self.colors[idx], linewidth=1.5)
                ax.set_title(f'{feature} - Coefficient of Variation')
                ax.set_xlabel('Time (seconds)')
                ax.set_ylabel('CV (std/mean)')
                ax.grid(True, alpha=0.3)
                
                # Add stability zones
                ax.axhline(y=0.1, color='green', linestyle='--', alpha=0.7, 
                          label='Stable (CV < 0.1)')
                ax.axhline(y=0.3, color='orange', linestyle='--', alpha=0.7, 
                          label='Moderate (CV < 0.3)')
                ax.legend(fontsize=8)
            else:
                ax.text(0.5, 0.5, f'Feature {feature}\nnot found', 
                       ha='center', va='center', transform=ax.transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
            print(f"📁 Feature stability saved to: {save_path}")
        
        return fig
    
    def create_feature_dashboard(self,
                               df_features: pd.DataFrame,
                               important_features: Optional[List[str]] = None,
                               save_dir: Optional[str] = None) -> Dict[str, plt.Figure]:
        """
        Create complete feature analysis dashboard.
        
        Args:
            df_features: Features DataFrame
            important_features: List of important features to focus on
            save_dir: Directory to save all figures
            
        Returns:
            Dictionary of figure objects
        """
        figures = {}
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Auto-select important features if not provided
        if important_features is None:
            numeric_cols = df_features.select_dtypes(include=[np.number]).columns
            important_features = list(numeric_cols[:8])  # Top 8 features
        
        print(f"🔄 Creating feature analysis dashboard for {len(important_features)} features...")
        
        # 1. Feature distributions
        print("📊 Creating feature distributions...")
        save_path = f"{save_dir}/feature_distributions.png" if save_dir else None
        figures['distributions'] = self.plot_feature_distributions(
            df_features, important_features, save_path=save_path)
        
        # 2. Correlation heatmap
        print("🔥 Creating correlation heatmap...")
        save_path = f"{save_dir}/feature_correlations.png" if save_dir else None
        figures['correlations'] = self.plot_correlation_heatmap(
            df_features, important_features, save_path=save_path)
        
        # 3. Feature evolution
        print("📈 Creating feature evolution...")
        save_path = f"{save_dir}/feature_evolution.png" if save_dir else None
        figures['evolution'] = self.plot_feature_evolution(
            df_features, important_features[:4], save_path=save_path)
        
        # 4. PCA analysis
        print("🎯 Creating PCA analysis...")
        save_path = f"{save_dir}/pca_analysis.png" if save_dir else None
        figures['pca'] = self.plot_pca_analysis(df_features, save_path=save_path)
        
        # 5. Feature stability
        print("📊 Creating stability analysis...")
        save_path = f"{save_dir}/feature_stability.png" if save_dir else None
        figures['stability'] = self.plot_feature_stability(
            df_features, important_features[:4], save_path=save_path)
        
        print(f"✅ Feature analysis dashboard created with {len(figures)} visualizations")
        
        return figures
