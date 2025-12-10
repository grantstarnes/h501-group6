# Top Games by Aggregated Rating - Integration Guide

## Overview
The `top_games_visualization.py` module provides reusable functions for visualizing and analyzing the highest-rated games by year from your IGDB data.

## Available Functions

### 1. `prepare_top_games_data(df, min_year=1998, required_columns=None)`
Cleans and prepares raw game data for analysis.

**Parameters:**
- `df`: Input DataFrame with game data
- `min_year`: Minimum release year to include (default: 1998)
- `required_columns`: List of required columns (default: ['name', 'aggregated_rating', 'first_release_date'])

**Returns:** Cleaned DataFrame ready for analysis

**Example:**
```python
from s3_loader_V2 import load_games
from top_games_visualization import prepare_top_games_data

games = load_games()
cleaned_df = prepare_top_games_data(games, min_year=1998)
```

---

### 2. `get_top_games_by_year(df)`
Extracts the highest-rated game for each year.

**Parameters:**
- `df`: Cleaned DataFrame from `prepare_top_games_data()`

**Returns:** DataFrame with one row per year, sorted by release_year

**Example:**
```python
top_games = get_top_games_by_year(cleaned_df)
print(top_games)
```

---

### 3. `get_top_games_table(top_games_df, sort_by='aggregated_rating', ascending=False, top_n=None)`
Creates a formatted table for display.

**Parameters:**
- `top_games_df`: DataFrame from `get_top_games_by_year()`
- `sort_by`: Column to sort by (default: 'aggregated_rating')
- `ascending`: Sort order (default: False for descending)
- `top_n`: Return only top N rows (default: None for all)

**Returns:** Formatted DataFrame with columns: release_year, name, aggregated_rating

**Example:**
```python
table = get_top_games_table(top_games, sort_by='aggregated_rating', top_n=10)
st.dataframe(table, use_container_width=True)
```

---

### 4. `plot_top_games_by_rating(top_games_df, height=600, show_labels=True, label_top_n=3, colormap='viridis')`
Creates an interactive Plotly visualization.

**Parameters:**
- `top_games_df`: DataFrame from `get_top_games_by_year()`
- `height`: Figure height in pixels (default: 600)
- `show_labels`: Show game name labels (default: True)
- `label_top_n`: Label only top N games (default: 3)
- `colormap`: Plotly colormap name (default: 'viridis')
  - Options: 'viridis', 'plasma', 'cool', 'twilight', 'turbo', 'jet', 'summer', 'ocean', etc.

**Returns:** Plotly Figure object

**Example:**
```python
fig = plot_top_games_by_rating(top_games, height=600, label_top_n=3, colormap='plasma')
st.plotly_chart(fig, use_container_width=True)
```

---

### 5. `display_top_games_analysis(df, min_year=1998, show_chart=True, show_table=True, show_insights=True)`
Complete Streamlit display with all components integrated.

**Parameters:**
- `df`: Raw games DataFrame
- `min_year`: Minimum release year (default: 1998)
- `show_chart`: Display interactive chart (default: True)
- `show_table`: Display top games table (default: True)
- `show_insights`: Display insights text (default: True)

**Returns:** None (renders directly to Streamlit)

**Example:**
```python
from s3_loader_V2 import load_games
from top_games_visualization import display_top_games_analysis

games = load_games()
display_top_games_analysis(games, min_year=1998)
```

---

## Integration Examples

### Option 1: Complete Analysis (Recommended)
Add to `streamlit_app_V2.py` in the main tab:

```python
import streamlit as st
from s3_loader_V2 import load_games
from top_games_visualization import display_top_games_analysis

# In your main() function:
tab1, tab2, tab3 = st.tabs(["Search", "Top Games", "Explore"])

with tab2:
    games = load_games()
    display_top_games_analysis(games, min_year=1998)
```

---

### Option 2: Custom Layout
Use individual functions for more control:

```python
from s3_loader_V2 import load_games
from top_games_visualization import (
    prepare_top_games_data,
    get_top_games_by_year,
    plot_top_games_by_rating,
    get_top_games_table
)

st.markdown("## Top-Rated Games Analysis")

# Load and prepare data
games = load_games()
cleaned_df = prepare_top_games_data(games, min_year=1998)
top_games = get_top_games_by_year(cleaned_df)

# Create two columns
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("### Game Ratings Trend")
    fig = plot_top_games_by_rating(
        top_games, 
        height=500, 
        label_top_n=5, 
        colormap='turbo'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("### Statistics")
    st.metric("Years Tracked", len(top_games))
    st.metric("Highest Rating", f"{top_games['aggregated_rating'].max():.1f}")
    st.metric("Lowest Rating", f"{top_games['aggregated_rating'].min():.1f}")
    st.metric("Average", f"{top_games['aggregated_rating'].mean():.1f}")

st.markdown("### Top 10 Games by Rating")
table = get_top_games_table(top_games, top_n=10)
st.dataframe(table, use_container_width=True)
```

---

### Option 3: Add as New Page/Section
Create a dedicated function in `streamlit_app_V2.py`:

```python
def display_top_games_page():
    """Display the top-rated games analysis page."""
    from top_games_visualization import display_top_games_analysis
    from s3_loader_V2 import load_games
    
    st.markdown("# 🎮 Top-Rated Games by Year")
    st.markdown("""
    Discover which games received the highest aggregated ratings each year 
    and explore trends in gaming over the last 25 years.
    """)
    
    # Sidebar filters
    st.sidebar.markdown("### Filters")
    min_year = st.sidebar.slider("Minimum Year", 1998, 2023, 1998)
    show_chart = st.sidebar.checkbox("Show Chart", value=True)
    show_table = st.sidebar.checkbox("Show Table", value=True)
    show_insights = st.sidebar.checkbox("Show Insights", value=True)
    
    # Main display
    games = load_games()
    display_top_games_analysis(
        games,
        min_year=min_year,
        show_chart=show_chart,
        show_table=show_table,
        show_insights=show_insights
    )

# In main():
if page == "Top Games":
    display_top_games_page()
```

---

## Styling Options

### Colormaps for Plotly
The visualization supports any Plotly colormap:
- **Sequential**: viridis, plasma, inferno, magma, cividis
- **Sequential (alternate)**: turbo, jet, summer, winter, spring, autumn
- **Diverging**: RdBu, RdYlBu, RdYlGn, BrBG, PiYG, PRGn, PuOr, Spectral
- **Cyclical**: twilight, cool, hot, Electric, Bluered

Example with different colormap:
```python
fig = plot_top_games_by_rating(top_games, colormap='turbo')
st.plotly_chart(fig, use_container_width=True)
```

---

## Data Requirements
The input DataFrame must contain these columns (before cleaning):
- `name`: Game title
- `aggregated_rating`: Numeric rating (0-100)
- `first_release_date`: Unix timestamp or numeric date

The module handles missing data, type conversions, and validation automatically.

---

## Performance Notes
- Processing is optimized for datasets up to 50,000+ games
- Plotly visualization is rendered client-side for smooth interactivity
- Data caching via `@st.cache_data` is recommended for repeated loads

---

## Example Output
The module produces:
1. **Interactive line chart** with color-coded points showing rating progression
2. **Metric cards** showing statistics
3. **Sortable data table** of top games
4. **Insights text** highlighting key findings

---

## Troubleshooting

### "No data available" message
- Check that your DataFrame has the required columns
- Verify `aggregated_rating` is not all NaN
- Ensure `first_release_date` contains valid timestamps

### Chart not rendering
- Verify Plotly is installed: `pip install plotly`
- Check that top_games DataFrame is not empty
- Try increasing `height` parameter

### Performance issues
- Filter data before calling functions (use `min_year` parameter)
- Cache the cleaned data with `@st.cache_data`
- Use `top_n` parameter in `get_top_games_table()` for large datasets
