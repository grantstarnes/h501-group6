# Top Games by Aggregated Rating - Streamlit Visualization Module

## 📋 Summary

This package contains a complete, production-ready visualization module for displaying the highest-rated games by year in your Streamlit app. It's based on the analysis from your Jupyter notebook `Highest_Rated_Games_by_Year.ipynb`.

## 📁 Files Included

### Main Module
- **`top_games_visualization.py`** - The core visualization module with 5 main functions

### Documentation
- **`TOP_GAMES_INTEGRATION.md`** - Complete integration guide with examples
- **`INTEGRATION_EXAMPLES.py`** - Practical code snippets showing 4 different integration approaches
- **`README_TOP_GAMES.md`** - This file

## 🚀 Quick Start

### 1. Copy the Module
Copy `top_games_visualization.py` to your `streamlit_app/` directory.

### 2. Import and Use
In your `streamlit_app_V2.py`:

```python
from top_games_visualization import display_top_games_analysis
from s3_loader_V2 import load_games

# In your main() function, add this to a tab or section:
games = load_games()
display_top_games_analysis(games, min_year=1998)
```

### 3. Done!
The module handles everything:
- ✅ Data cleaning and validation
- ✅ Top-rated game calculation
- ✅ Interactive Plotly visualization
- ✅ Statistical metrics display
- ✅ Sortable data table
- ✅ Insights summary

## 🎯 Key Functions

### `display_top_games_analysis()` - Complete Solution
The easiest way to add this to your app. It includes everything:
```python
display_top_games_analysis(df, min_year=1998)
```

### `prepare_top_games_data()` - Data Cleaning
Cleans raw game data for analysis:
```python
cleaned_df = prepare_top_games_data(df, min_year=1998)
```

### `get_top_games_by_year()` - Aggregation
Gets the top-rated game for each year:
```python
top_games = get_top_games_by_year(cleaned_df)
```

### `plot_top_games_by_rating()` - Interactive Chart
Creates an interactive visualization:
```python
fig = plot_top_games_by_rating(top_games, colormap='turbo')
st.plotly_chart(fig, use_container_width=True)
```

### `get_top_games_table()` - Data Export
Creates a formatted table:
```python
table = get_top_games_table(top_games, top_n=15)
st.dataframe(table)
```

## 💡 Integration Examples

### Option 1: Simple Tab Integration
```python
def main():
    tab1, tab2 = st.tabs(["Search", "Top Games"])
    
    with tab1:
        # ... search code ...
    
    with tab2:
        from top_games_visualization import display_top_games_analysis
        games = load_games()
        display_top_games_analysis(games)

if __name__ == "__main__":
    main()
```

### Option 2: Page Selector
```python
def main():
    page = st.sidebar.radio("Page", ["Search", "Top Games"])
    
    if page == "Top Games":
        from top_games_visualization import display_top_games_analysis
        games = load_games()
        display_top_games_analysis(games)
```

### Option 3: Custom Layout
```python
from top_games_visualization import (
    prepare_top_games_data,
    get_top_games_by_year,
    plot_top_games_by_rating,
    get_top_games_table
)

games = load_games()
cleaned = prepare_top_games_data(games)
top = get_top_games_by_year(cleaned)

col1, col2 = st.columns([2, 1])
with col1:
    fig = plot_top_games_by_rating(top)
    st.plotly_chart(fig, use_container_width=True)
with col2:
    table = get_top_games_table(top, top_n=10)
    st.dataframe(table)
```

## 🎨 Customization

### Change Color Scheme
```python
fig = plot_top_games_by_rating(top_games, colormap='plasma')
```

Available colormaps: `'viridis'`, `'plasma'`, `'turbo'`, `'cool'`, `'twilight'`, `'jet'`, `'summer'`, `'ocean'`, `'inferno'`

### Adjust Date Range
```python
display_top_games_analysis(games, min_year=2010)
```

### Show/Hide Components
```python
display_top_games_analysis(
    games,
    show_chart=True,      # Toggle the interactive chart
    show_table=True,      # Toggle the data table
    show_insights=True    # Toggle the insights text
)
```

### Customize Labels
```python
fig = plot_top_games_by_rating(
    top_games,
    label_top_n=5,        # Label only top 5 games
    show_labels=True      # Show game name labels
)
```

## 📊 Output

The module produces:

1. **Interactive Chart**
   - Line chart showing rating trends
   - Color-coded points based on rating value
   - Hover tooltips with game info
   - Labeled top N games

2. **Statistics Metrics**
   - Total years tracked
   - Highest rating
   - Lowest rating
   - Average rating

3. **Data Table**
   - Sortable by any column
   - Formatted with release year, game name, rating
   - Customizable row limit

4. **Insights Section**
   - Key findings summary
   - Highest and lowest rated games
   - Rating range
   - Historical context

## 🔧 Requirements

These are already installed in your `requirements.txt`:
- `pandas` - Data processing
- `numpy` - Numerical operations
- `streamlit` - Web app framework
- `plotly` - Interactive visualizations
- `matplotlib` - (imported but not used; you can remove if needed)

## ⚙️ Data Requirements

Input DataFrame must contain:
- `name` - Game title
- `aggregated_rating` - Numeric rating (0-100)
- `first_release_date` - Unix timestamp

The module automatically:
- Handles missing values
- Converts data types
- Validates inputs
- Cleans outliers

## 🐛 Troubleshooting

### "No data available"
- Check your DataFrame has required columns
- Verify `aggregated_rating` isn't all NaN
- Ensure dates are valid timestamps

### Chart doesn't appear
- Verify Plotly is installed: `pip install plotly`
- Check that data is not empty
- Try increasing figure height: `height=800`

### Slow performance
- Filter data before processing: `min_year=2015`
- Use `@st.cache_data` to cache results
- Limit table rows: `top_n=20`

## 📚 Source Notebook

This module is based on the analysis in:
```
/Highest_Rated_Games_by_Year.ipynb
```

Key cells referenced:
- Cell: Data loading and cleaning
- Cell: Top games by year aggregation
- Cell: Advanced visualization with color gradient

## 🎓 Educational Notes

The notebook analysis shows:
- Games released after 1998 have aggregated ratings
- Top ratings peaked around 2005 and 2017
- Recent years show lower top ratings (possibly due to more diverse gaming or higher critical standards)
- Aggregated ratings combine user and critic scores on 0-100 scale
- Scores above 90.0 indicate widespread popularity

## 📝 License

Same as your project (check your main README for details)

## 🤝 Contributing

To extend this module:
1. Add new functions following the existing pattern
2. Keep functions focused and single-purpose
3. Use type hints for clarity
4. Document with docstrings
5. Test with both small and large datasets

## 📞 Support

For issues or questions:
1. Check `TOP_GAMES_INTEGRATION.md` for detailed docs
2. Review `INTEGRATION_EXAMPLES.py` for code samples
3. Examine `top_games_visualization.py` function docstrings

---

**Created**: December 2025  
**Status**: Production Ready  
**Version**: 1.0
