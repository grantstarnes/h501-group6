"""
Top Games by Aggregated Rating Visualization Module.

This module provides functions to visualize and analyze the highest-rated games
by year, showing trends in game ratings over time.

Functions:
    - prepare_top_games_data: Prepare and clean data for visualization
    - plot_top_games_by_rating: Create an interactive Plotly visualization
    - get_top_games_table: Get a DataFrame of top games sorted by rating
    - display_top_games_analysis: Complete Streamlit display with chart and table
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib.colors import LinearSegmentedColormap


def prepare_top_games_data(
    df: pd.DataFrame,
    min_year: int = 1998,
    required_columns: list = None  # type: ignore
) -> pd.DataFrame:
    """
    Prepare and clean game data for top-rated analysis.
    
    Args:
        df: Input DataFrame with game data
        min_year: Minimum release year to include (default: 1998)
        required_columns: List of required columns (default: ['name', 'aggregated_rating', 'first_release_date'])
    
    Returns:
        DataFrame with cleaned data or empty DataFrame if issues occur
    """
    if required_columns is None:
        required_columns = ['name', 'aggregated_rating', 'first_release_date']
    
    # Check if required columns exist
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(f"Missing required columns: {missing_cols}")
        return pd.DataFrame()
    
    # Keep only relevant columns
    df = df[required_columns].copy()
    
    # Drop rows with missing aggregated_rating or first_release_date
    df = df.dropna(subset=['aggregated_rating', 'first_release_date'])
    
    if df.empty:
        st.warning("No valid data after filtering for required columns.")
        return df
    
    # Clean aggregated_rating: convert to numeric
    df['aggregated_rating'] = pd.to_numeric(df['aggregated_rating'], errors='coerce')
    df = df.dropna(subset=['aggregated_rating'])
    
    # Round to 1 decimal place
    df['aggregated_rating'] = df['aggregated_rating'].round(1)
    
    # Clean first_release_date: convert to numeric (Unix timestamp)
    df['first_release_date'] = pd.to_numeric(df['first_release_date'], errors='coerce').astype('Int64')
    
    # Convert timestamp to year
    df['release_year'] = pd.to_datetime(
        df['first_release_date'],
        unit='s',
        errors='coerce'
    ).dt.year
    
    # Drop rows with invalid years
    df = df.dropna(subset=['release_year'])
    df['release_year'] = df['release_year'].astype(int)
    
    # Filter for games released after min_year
    df = df[df['release_year'] > min_year].reset_index(drop=True)
    
    return df


def get_top_games_by_year(df: pd.DataFrame) -> pd.DataFrame:
    """
    Get the highest-rated game for each year.
    
    Args:
        df: Cleaned DataFrame from prepare_top_games_data()
    
    Returns:
        DataFrame with one row per year, sorted by release_year
    """
    if df.empty:
        return pd.DataFrame()
    
    top_games = (
        df.sort_values('aggregated_rating', ascending=False)
          .groupby('release_year', as_index=False)
          .first()
          .sort_values('release_year')
    )
    
    return top_games


def get_top_games_table(
    top_games_df: pd.DataFrame,
    sort_by: str = 'aggregated_rating',
    ascending: bool = False,
    top_n: int = None  # type: ignore
) -> pd.DataFrame:
    """
    Get a formatted table of top games.
    
    Args:
        top_games_df: DataFrame from get_top_games_by_year()
        sort_by: Column to sort by (default: 'aggregated_rating')
        ascending: Sort order (default: False for descending)
        top_n: Return only top N rows (default: None for all)
    
    Returns:
        Formatted DataFrame ready for display
    """
    if top_games_df.empty:
        return pd.DataFrame()
    
    result = top_games_df.sort_values(sort_by, ascending=ascending)[
        ['release_year', 'name', 'aggregated_rating']
    ].reset_index(drop=True)
    
    if top_n:
        result = result.head(top_n)
    
    return result


def plot_top_games_by_rating(
    top_games_df: pd.DataFrame,
    height: int = 600,
    show_labels: bool = True,
    label_top_n: int = 3,
    colormap: str = 'viridis'
) -> go.Figure:
    """
    Create an interactive Plotly visualization of top-rated games by year.
    
    Args:
        top_games_df: DataFrame from get_top_games_by_year()
        height: Figure height in pixels (default: 600)
        show_labels: Show game name labels (default: True)
        label_top_n: Label only top N games (default: 3)
        colormap: Colormap name - 'viridis', 'plasma', 'cool', 'twilight', etc. (default: 'viridis')
    
    Returns:
        Plotly Figure object
    """
    if top_games_df.empty:
        return go.Figure().add_annotation(text="No data available")
    
    # Create scatter plot with line
    fig = go.Figure()
    
    # Add line connecting points
    fig.add_trace(go.Scatter(
        x=top_games_df['release_year'],
        y=top_games_df['aggregated_rating'],
        mode='lines',
        line=dict(color='lightgray', width=2),
        name='Trend',
        hoverinfo='skip',
        showlegend=False
    ))
    
    # Add scatter points with color gradient
    fig.add_trace(go.Scatter(
        x=top_games_df['release_year'],
        y=top_games_df['aggregated_rating'],
        mode='markers',
        marker=dict(
            size=10,
            color=top_games_df['aggregated_rating'],
            colorscale=colormap,
            showscale=True,
            colorbar=dict(title="Aggregated<br>Rating"),
            line=dict(width=1, color='black')
        ),
        text=top_games_df['name'],
        hovertemplate='<b>%{text}</b><br>Year: %{x}<br>Rating: %{y}<extra></extra>',
        name='Games'
    ))
    
    # Add labels for top N games if requested
    if show_labels and label_top_n > 0:
        top_n_games = top_games_df.nlargest(label_top_n, 'aggregated_rating')
        
        for _, row in top_n_games.iterrows():
            fig.add_annotation(
                x=row['release_year'],
                y=row['aggregated_rating'] + 1,
                text=row['name'],
                showarrow=True,
                arrowhead=2,
                arrowsize=1,
                arrowwidth=1,
                arrowcolor='gray',
                ax=0,
                ay=-30,
                font=dict(size=10, color='darkblue')
            )
    
    # Update layout
    fig.update_layout(
        title='Top-Rated Games by Year (1998+)',
        xaxis_title='Release Year',
        yaxis_title='Aggregated Rating',
        height=height,
        hovermode='closest',
        template='plotly_white',
        xaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray'
        ),
        yaxis=dict(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            range=[65, 105]  # Extra space for labels
        )
    )
    
    return fig


def display_top_games_analysis(
    df: pd.DataFrame,
    min_year: int = 1998,
    show_chart: bool = True,
    show_table: bool = True,
    show_insights: bool = True
):
    """
    Display complete analysis of top-rated games with Streamlit components.
    
    Args:
        df: Raw games DataFrame
        min_year: Minimum release year (default: 1998)
        show_chart: Display interactive chart (default: True)
        show_table: Display top games table (default: True)
        show_insights: Display insights text (default: True)
    """
    st.markdown("### 🎮 Top-Rated Games by Year Analysis")
    
    # Prepare data
    cleaned_df = prepare_top_games_data(df, min_year=min_year)
    
    if cleaned_df.empty:
        st.error("Unable to process data. Please check your input.")
        return
    
    top_games_df = get_top_games_by_year(cleaned_df)
    
    if top_games_df.empty:
        st.warning("No games found after filtering.")
        return
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Years", len(top_games_df))
    with col2:
        st.metric("Highest Rating", f"{top_games_df['aggregated_rating'].max():.1f}")
    with col3:
        st.metric("Lowest Rating", f"{top_games_df['aggregated_rating'].min():.1f}")
    with col4:
        avg_rating = top_games_df['aggregated_rating'].mean()
        st.metric("Average Rating", f"{avg_rating:.1f}")
    
    # Display chart
    if show_chart:
        st.markdown("#### Top-Rated Game Each Year")
        fig = plot_top_games_by_rating(top_games_df, height=600, label_top_n=3)
        st.plotly_chart(fig, use_container_width=True)
    
    # Display table
    if show_table:
        st.markdown("#### Top Games by Rating (Sorted)")
        table_df = get_top_games_table(top_games_df, top_n=15)
        st.dataframe(table_df, use_container_width=True)
    
    # Display insights
    if show_insights:
        st.markdown("#### 📊 Key Insights")
        
        # Find games with highest and lowest ratings
        highest = top_games_df.loc[top_games_df['aggregated_rating'].idxmax()]
        lowest = top_games_df.loc[top_games_df['aggregated_rating'].idxmin()]
        
        highest_year = int(highest['release_year'])
        lowest_year = int(lowest['release_year'])
        highest_rating = float(highest['aggregated_rating'])
        lowest_rating = float(lowest['aggregated_rating'])
        
        insights_text = f"""
        - **Highest Rated**: *{highest['name']}* ({highest_year}) with a rating of {highest_rating:.1f}
        - **Lowest Rated**: *{lowest['name']}* ({lowest_year}) with a rating of {lowest_rating:.1f}
        - **Rating Range**: {top_games_df['aggregated_rating'].max() - top_games_df['aggregated_rating'].min():.1f} points
        - **Analysis Period**: {int(top_games_df['release_year'].min())} - {int(top_games_df['release_year'].max())}
        
        The data shows the highest-rated game released in each year since {min_year}. 
        Games with aggregated ratings above 90.0 are considered excellent, indicating widespread popularity.
        Recent years show lower top ratings, which may indicate either more diverse game selection 
        or higher critical standards from the gaming community.
        """
        
        st.markdown(insights_text)


if __name__ == "__main__":
    # Example usage
    print("This module is designed to be imported into a Streamlit app.")
    print("\nExample usage:")
    print("  from top_games_visualization import display_top_games_analysis")
    print("  display_top_games_analysis(df)")
