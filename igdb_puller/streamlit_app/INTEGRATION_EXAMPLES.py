"""
Example: How to integrate top_games_visualization into streamlit_app_V2.py

This file shows the exact code snippets to add to your Streamlit app.
"""

# ============================================================================
# OPTION 1: Add as a new tab (Simplest)
# ============================================================================
# Add this to your main() function where you create tabs:

def main():
    """Main Streamlit app."""
    
    # ... existing code ...
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Search", "Top Games", "Genres", "About"])
    
    # Tab 1: Search (existing code)
    with tab1:
        st.markdown("## 🔍 Game Search")
        # ... existing search code ...
    
    # Tab 2: Top Games (NEW)
    with tab2:
        from top_games_visualization import display_top_games_analysis
        
        games = load_games()
        display_top_games_analysis(games, min_year=1998)
    
    # Tab 3: Genres (existing code)
    with tab3:
        st.markdown("## 🎮 Explore by Genre")
        # ... existing genre code ...
    
    # Tab 4: About (existing code)
    with tab4:
        st.markdown("## About")
        # ... existing about code ...


# ============================================================================
# OPTION 2: Add as a sidebar page selector (More flexible)
# ============================================================================
# Modify your main() function to include page selection:

def main():
    """Main Streamlit app with page selector."""
    
    st.sidebar.markdown("## Navigation")
    page = st.sidebar.radio(
        "Select a page",
        ["🔍 Search", "🎮 Top Games", "📊 Explore", "ℹ️ About"]
    )
    
    if page == "🔍 Search":
        page_search()
    elif page == "🎮 Top Games":
        page_top_games()
    elif page == "📊 Explore":
        page_explore()
    elif page == "ℹ️ About":
        page_about()


def page_top_games():
    """Display the top games analysis page."""
    from top_games_visualization import display_top_games_analysis
    
    st.markdown("# 🎮 Top-Rated Games by Year")
    st.markdown("""
    Explore the highest-rated games released each year and discover trends 
    in gaming popularity over the last 25+ years.
    """)
    
    # Optional: Add sidebar controls
    st.sidebar.markdown("### Top Games Settings")
    min_year = st.sidebar.slider(
        "Filter from year:",
        min_value=1998,
        max_value=2023,
        value=1998
    )
    show_insights = st.sidebar.checkbox("Show Insights", value=True)
    
    # Load and display
    games = load_games()
    display_top_games_analysis(
        games,
        min_year=min_year,
        show_chart=True,
        show_table=True,
        show_insights=show_insights
    )


# ============================================================================
# OPTION 3: Advanced custom layout with filters
# ============================================================================
# For maximum control over appearance:

def page_top_games_advanced():
    """Advanced top games page with custom layout."""
    from top_games_visualization import (
        prepare_top_games_data,
        get_top_games_by_year,
        plot_top_games_by_rating,
        get_top_games_table
    )
    from s3_loader_V2 import load_games
    
    st.markdown("# 🎮 Top-Rated Games Analysis")
    
    # Sidebar controls
    st.sidebar.markdown("### Visualization Settings")
    min_year = st.sidebar.slider("Minimum Year", 1998, 2023, 1998)
    label_top_n = st.sidebar.slider("Label Top N Games", 1, 10, 3)
    colormap = st.sidebar.selectbox(
        "Color Scheme",
        ["viridis", "plasma", "turbo", "cool", "twilight", "jet", "summer"]
    )
    
    # Load data
    games = load_games()
    cleaned_df = prepare_top_games_data(games, min_year=min_year)
    top_games = get_top_games_by_year(cleaned_df)
    
    if top_games.empty:
        st.error("No data available for the selected filters.")
        return
    
    # Display statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Years", len(top_games))
    with col2:
        st.metric("Highest Rating", f"{top_games['aggregated_rating'].max():.1f}")
    with col3:
        st.metric("Lowest Rating", f"{top_games['aggregated_rating'].min():.1f}")
    with col4:
        avg = top_games['aggregated_rating'].mean()
        st.metric("Average Rating", f"{avg:.1f}")
    
    # Create two columns for layout
    col_chart, col_table = st.columns([2, 1])
    
    # Chart in left column
    with col_chart:
        st.markdown("### Ratings Trend Over Time")
        fig = plot_top_games_by_rating(
            top_games,
            height=600,
            show_labels=True,
            label_top_n=label_top_n,
            colormap=colormap
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Table in right column
    with col_table:
        st.markdown("### Top 10 Games")
        table = get_top_games_table(top_games, top_n=10)
        st.dataframe(table, use_container_width=True)
    
    # Additional insights
    st.markdown("---")
    st.markdown("### 📊 Full Ranking")
    full_table = get_top_games_table(top_games, top_n=None)
    st.dataframe(full_table, use_container_width=True)


# ============================================================================
# OPTION 4: Minimal integration - Just the imports and function call
# ============================================================================
# If you already have a tab structure, just add this to a new tab:

"""
With tab2:  # Assuming you have a tab named tab2
    from top_games_visualization import display_top_games_analysis
    from s3_loader_V2 import load_games
    
    games = load_games()
    display_top_games_analysis(games)
"""

# ============================================================================
# STEP-BY-STEP: How to Add This to streamlit_app_V2.py
# ============================================================================

"""
1. Copy the file `top_games_visualization.py` to the same directory as 
   streamlit_app_V2.py

2. At the top of streamlit_app_V2.py, add the import:
   
   from top_games_visualization import display_top_games_analysis

3. In your main() function, add this code where you want the visualization:
   
   with tab_name:
       games = load_games()
       display_top_games_analysis(games, min_year=1998)

4. That's it! The function handles all the data processing and display.

5. (Optional) For more control, use individual functions:
   - prepare_top_games_data()
   - get_top_games_by_year()
   - plot_top_games_by_rating()
   - get_top_games_table()
"""

if __name__ == "__main__":
    print(__doc__)
