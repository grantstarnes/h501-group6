#!/usr/bin/env python3
r"""
Quick Visual Guide: Top Games Visualization

This file shows what the module produces at a glance.
"""

r"""
═══════════════════════════════════════════════════════════════════════════════
WHAT THE VISUALIZATION LOOKS LIKE
═══════════════════════════════════════════════════════════════════════════════

When you call:
    display_top_games_analysis(games, min_year=1998)

You get this in your Streamlit app:

┌─────────────────────────────────────────────────────────────────────────────┐
│                  ### 🎮 Top-Rated Games by Year Analysis                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐              │
│  │   Years    │ │   Highest  │ │   Lowest   │ │  Average   │              │
│  │     25     │ │    96.5    │ │    70.1    │ │    83.2    │              │
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘              │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  #### Top-Rated Game Each Year                                             │
│                                                                              │
│  100 │                                     ●  (Top Label)                  │
│      │                                   /    \                            │
│   90 │    ●  (Zelda: OoT)        ●  ●  ●         ● (RDR2)                │
│      │   / \    /   \          /   \  /  \      /  \                     │
│   80 │  ●   ● ●     ●    ●  ●              ●  ●    ●  (Recent Low)      │
│      │ /     X        \  /  \/                   \  /                     │
│   70 │●                ●                         ●                        │
│      ├─────────────────────────────────────────────────────────────────┤ │
│      1998  2003  2008  2013  2018  2023                                  │
│                                                                              │
│  🌈 Color gradient represents rating values                               │
│  💡 Dark blue = Lower rating | Green = Higher rating                     │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  #### Top Games by Rating (Sorted)                                        │
│                                                                              │
│     release_year  │ name                       │ aggregated_rating        │
│  ─────────────────┼────────────────────────────┼─────────────────        │
│        1998       │ The Legend of Zelda: Oc... │     96.5                │
│        2017       │ Red Dead Redemption 2      │     95.8                │
│        2005       │ Metal Gear Solid III       │     94.2                │
│        2013       │ The Last of Us             │     93.1                │
│        2010       │ God of War III             │     92.7                │
│        ...        │ ...                        │     ...                 │
│                                                                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  #### 📊 Key Insights                                                     │
│                                                                              │
│  - Highest Rated: The Legend of Zelda: Ocarina of Time (1998) with       │
│    rating of 96.5                                                          │
│  - Lowest Rated: Dead Island 2 (2023) with rating of 70.1                │
│  - Rating Range: 26.4 points                                              │
│  - Analysis Period: 1998 - 2023                                           │
│                                                                              │
│  The data shows the highest-rated game released in each year since 1998.  │
│  Games with aggregated ratings above 90.0 are considered excellent...    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════
CODE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

EXAMPLE 1: Simplest Usage
───────────────────────────────────────────────────────────────────────────────
from top_games_visualization import display_top_games_analysis
from s3_loader_V2 import load_games

games = load_games()
display_top_games_analysis(games)

→ Result: Complete analysis with chart, metrics, table, and insights


EXAMPLE 2: With Filters
───────────────────────────────────────────────────────────────────────────────
display_top_games_analysis(
    games,
    min_year=2010,           # Only show games after 2010
    show_chart=True,         # Include interactive chart
    show_table=True,         # Include data table
    show_insights=True       # Include insights text
)

→ Result: Same layout but filtered to recent games


EXAMPLE 3: Custom Color Scheme
───────────────────────────────────────────────────────────────────────────────
from top_games_visualization import (
    prepare_top_games_data,
    get_top_games_by_year,
    plot_top_games_by_rating
)

cleaned = prepare_top_games_data(games)
top = get_top_games_by_year(cleaned)
fig = plot_top_games_by_rating(top, colormap='plasma')
st.plotly_chart(fig, use_container_width=True)

→ Result: Chart with plasma colormap (yellow-orange-purple gradient)


EXAMPLE 4: Complete Custom Layout
───────────────────────────────────────────────────────────────────────────────
from top_games_visualization import *

games = load_games()
cleaned = prepare_top_games_data(games, min_year=2000)
top = get_top_games_by_year(cleaned)

st.markdown("# My Custom Layout")

# Metrics in 4 columns
cols = st.columns(4)
with cols[0]:
    st.metric("Years", len(top))
with cols[1]:
    st.metric("Max", f"{top['aggregated_rating'].max():.1f}")
with cols[2]:
    st.metric("Min", f"{top['aggregated_rating'].min():.1f}")
with cols[3]:
    st.metric("Avg", f"{top['aggregated_rating'].mean():.1f}")

# Chart in main area
fig = plot_top_games_by_rating(top, height=500, colormap='turbo')
st.plotly_chart(fig, use_container_width=True)

# Table in sidebar
with st.sidebar:
    st.markdown("### Top 10")
    table = get_top_games_table(top, top_n=10)
    st.dataframe(table)

→ Result: Fully customized layout with your design


═══════════════════════════════════════════════════════════════════════════════
COLOR SCHEMES AVAILABLE
═══════════════════════════════════════════════════════════════════════════════

Sequential (Low → High):
  'viridis'      : Purple → Green      (default)
  'plasma'       : Purple → Yellow
  'inferno'      : Black → Yellow
  'magma'        : Black → Purple
  'cividis'      : Blue → Yellow (colorblind friendly)

Sequential (Warm):
  'turbo'        : Blue → Green → Yellow
  'hot'          : Black → Red → Yellow
  'summer'       : Green → Yellow
  'autumn'       : Yellow → Red → Brown
  'spring'       : Pink → Yellow
  'winter'       : Blue → Green

Sequential (Cool):
  'cool'         : Cyan → Magenta
  'ocean'        : Dark Blue → Light Blue → Green

Diverging:
  'twilight'     : Purple → Pink → Yellow → Green (cyclical)
  'RdBu'         : Red ← White → Blue
  'RdYlGn'       : Red ← White → Green

Example usage:
    fig = plot_top_games_by_rating(top, colormap='turbo')


═══════════════════════════════════════════════════════════════════════════════
INTEGRATION CHECKLIST
═══════════════════════════════════════════════════════════════════════════════

□ Copy top_games_visualization.py to streamlit_app/ folder
□ Add import: from top_games_visualization import display_top_games_analysis
□ Add function call in main()
□ Test by running: streamlit run streamlit_app_V2.py
□ Navigate to the new section
□ Verify chart, table, and metrics display correctly
□ (Optional) Customize colors, filters, layout
□ Deploy to Streamlit Cloud (no changes needed!)


═══════════════════════════════════════════════════════════════════════════════
DATA FLOW
═══════════════════════════════════════════════════════════════════════════════

Raw Games DataFrame
        ↓
prepare_top_games_data()    ← Cleans, validates, filters
        ↓
Cleaned DataFrame
        ↓
get_top_games_by_year()     ← Aggregates to 1 row per year
        ↓
Top Games DataFrame (year, name, rating)
        ↓ ┌─────────────────────────────┬──────────────────────────┐
          ↓                               ↓
    plot_top_games_by_rating()    get_top_games_table()
          ↓                               ↓
    Plotly Figure            Formatted DataFrame
          ↓                               ↓
    st.plotly_chart()          st.dataframe()


═══════════════════════════════════════════════════════════════════════════════
PARAMETERS & OPTIONS
═══════════════════════════════════════════════════════════════════════════════

display_top_games_analysis()
├─ df : DataFrame (required)
│  └─ The games data from load_games()
├─ min_year : int (default: 1998)
│  └─ Earliest year to include
├─ show_chart : bool (default: True)
│  └─ Display interactive visualization
├─ show_table : bool (default: True)
│  └─ Display sortable data table
└─ show_insights : bool (default: True)
   └─ Display insights section

plot_top_games_by_rating()
├─ top_games_df : DataFrame (required)
│  └─ From get_top_games_by_year()
├─ height : int (default: 600)
│  └─ Chart height in pixels
├─ show_labels : bool (default: True)
│  └─ Show game name annotations
├─ label_top_n : int (default: 3)
│  └─ How many games to label
└─ colormap : str (default: 'viridis')
   └─ Color scheme name

get_top_games_table()
├─ top_games_df : DataFrame (required)
│  └─ From get_top_games_by_year()
├─ sort_by : str (default: 'aggregated_rating')
│  └─ Column to sort by
├─ ascending : bool (default: False)
│  └─ Sort order (False = highest first)
└─ top_n : int (default: None)
   └─ Return only top N rows


═══════════════════════════════════════════════════════════════════════════════
FEATURES AT A GLANCE
═══════════════════════════════════════════════════════════════════════════════

✅ Interactive Chart
   • Hover tooltips with game info
   • Color gradient based on rating
   • Zoom and pan support
   • Export as PNG

✅ Responsive Design
   • Works on desktop and mobile
   • Adjusts to container width
   • Touch-friendly interactions

✅ Data Processing
   • Automatic type conversion
   • Missing value handling
   • Date range filtering
   • Data validation

✅ Customization
   • 15+ color schemes
   • Adjustable date range
   • Show/hide components
   • Custom layouts

✅ Performance
   • Optimized for 10k+ games
   • Fast data aggregation
   • Client-side rendering
   • Memory efficient

✅ Documentation
   • Complete docstrings
   • Type hints for IDE support
   • Multiple integration guides
   • Code examples


═══════════════════════════════════════════════════════════════════════════════
EXPECTED DATA FORMAT
═══════════════════════════════════════════════════════════════════════════════

Input DataFrame Columns (from load_games()):
  name                 : str    (game title)
  aggregated_rating    : float  (0-100 scale)
  first_release_date   : float  (Unix timestamp or numeric date)

Example:
  name                    aggregated_rating  first_release_date
  The Legend of Zelda      96.5               894076800
  Red Dead Redemption 2    95.2               1540425600
  ...

The module automatically:
  • Converts timestamps to years
  • Handles missing values
  • Validates data types
  • Cleans outliers


═══════════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════════

Problem: "No data available" message
Solution: 
  1. Check DataFrame has required columns
  2. Verify aggregated_rating is not all NaN
  3. Check date format is numeric

Problem: Chart doesn't render
Solution:
  1. Ensure Plotly is installed: pip install plotly
  2. Check top_games DataFrame is not empty
  3. Verify data types are correct
  4. Try increasing height: height=800

Problem: Performance is slow
Solution:
  1. Filter data before processing: min_year=2010
  2. Cache results: @st.cache_data
  3. Limit table rows: top_n=20
  4. Run streamlit with --logger.level=warning

═══════════════════════════════════════════════════════════════════════════════
SUCCESS CRITERIA
═══════════════════════════════════════════════════════════════════════════════

You know it's working when you see:
  ✓ Four metric cards (Years, Highest, Lowest, Average)
  ✓ Interactive line chart with color gradient
  ✓ Game name labels on top games
  ✓ Sortable data table below chart
  ✓ Insights section with key findings
  ✓ Hover tooltips on chart points
  ✓ No error messages in console

═══════════════════════════════════════════════════════════════════════════════
"""

print(__doc__)
