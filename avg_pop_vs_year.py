{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "6c06fb9e",
   "metadata": {},
   "outputs": [
    {
     "ename": "FileNotFoundError",
     "evalue": "[Errno 2] No such file or directory: 'games.csv'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mFileNotFoundError\u001b[0m                         Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 4\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mpandas\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mpd\u001b[39;00m\n\u001b[0;32m      2\u001b[0m \u001b[38;5;28;01mimport\u001b[39;00m \u001b[38;5;21;01mplotly\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mexpress\u001b[39;00m \u001b[38;5;28;01mas\u001b[39;00m \u001b[38;5;21;01mpx\u001b[39;00m\n\u001b[1;32m----> 4\u001b[0m games_df \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mread_csv(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mgames.csv\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m      5\u001b[0m popularity_df \u001b[38;5;241m=\u001b[39m pd\u001b[38;5;241m.\u001b[39mread_csv(\u001b[38;5;124m'\u001b[39m\u001b[38;5;124mpopularity.csv\u001b[39m\u001b[38;5;124m'\u001b[39m)\n\u001b[0;32m      6\u001b[0m \u001b[38;5;66;03m# Preprocess games data\u001b[39;00m\n",
      "File \u001b[1;32mc:\\Users\\davin\\.conda\\Lib\\site-packages\\pandas\\util\\_decorators.py:211\u001b[0m, in \u001b[0;36mdeprecate_kwarg.<locals>._deprecate_kwarg.<locals>.wrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    209\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    210\u001b[0m         kwargs[new_arg_name] \u001b[38;5;241m=\u001b[39m new_arg_value\n\u001b[1;32m--> 211\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32mc:\\Users\\davin\\.conda\\Lib\\site-packages\\pandas\\util\\_decorators.py:331\u001b[0m, in \u001b[0;36mdeprecate_nonkeyword_arguments.<locals>.decorate.<locals>.wrapper\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m    325\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;28mlen\u001b[39m(args) \u001b[38;5;241m>\u001b[39m num_allow_args:\n\u001b[0;32m    326\u001b[0m     warnings\u001b[38;5;241m.\u001b[39mwarn(\n\u001b[0;32m    327\u001b[0m         msg\u001b[38;5;241m.\u001b[39mformat(arguments\u001b[38;5;241m=\u001b[39m_format_argument_list(allow_args)),\n\u001b[0;32m    328\u001b[0m         \u001b[38;5;167;01mFutureWarning\u001b[39;00m,\n\u001b[0;32m    329\u001b[0m         stacklevel\u001b[38;5;241m=\u001b[39mfind_stack_level(),\n\u001b[0;32m    330\u001b[0m     )\n\u001b[1;32m--> 331\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m func(\u001b[38;5;241m*\u001b[39margs, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwargs)\n",
      "File \u001b[1;32mc:\\Users\\davin\\.conda\\Lib\\site-packages\\pandas\\io\\parsers\\readers.py:950\u001b[0m, in \u001b[0;36mread_csv\u001b[1;34m(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, squeeze, prefix, mangle_dupe_cols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, error_bad_lines, warn_bad_lines, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options)\u001b[0m\n\u001b[0;32m    935\u001b[0m kwds_defaults \u001b[38;5;241m=\u001b[39m _refine_defaults_read(\n\u001b[0;32m    936\u001b[0m     dialect,\n\u001b[0;32m    937\u001b[0m     delimiter,\n\u001b[1;32m   (...)\u001b[0m\n\u001b[0;32m    946\u001b[0m     defaults\u001b[38;5;241m=\u001b[39m{\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdelimiter\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m,\u001b[39m\u001b[38;5;124m\"\u001b[39m},\n\u001b[0;32m    947\u001b[0m )\n\u001b[0;32m    948\u001b[0m kwds\u001b[38;5;241m.\u001b[39mupdate(kwds_defaults)\n\u001b[1;32m--> 950\u001b[0m \u001b[38;5;28;01mreturn\u001b[39;00m _read(filepath_or_buffer, kwds)\n",
      "File \u001b[1;32mc:\\Users\\davin\\.conda\\Lib\\site-packages\\pandas\\io\\parsers\\readers.py:605\u001b[0m, in \u001b[0;36m_read\u001b[1;34m(filepath_or_buffer, kwds)\u001b[0m\n\u001b[0;32m    602\u001b[0m _validate_names(kwds\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnames\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m))\n\u001b[0;32m    604\u001b[0m \u001b[38;5;66;03m# Create the parser.\u001b[39;00m\n\u001b[1;32m--> 605\u001b[0m parser \u001b[38;5;241m=\u001b[39m TextFileReader(filepath_or_buffer, \u001b[38;5;241m*\u001b[39m\u001b[38;5;241m*\u001b[39mkwds)\n\u001b[0;32m    607\u001b[0m \u001b[38;5;28;01mif\u001b[39;00m chunksize \u001b[38;5;129;01mor\u001b[39;00m iterator:\n\u001b[0;32m    608\u001b[0m     \u001b[38;5;28;01mreturn\u001b[39;00m parser\n",
      "File \u001b[1;32mc:\\Users\\davin\\.conda\\Lib\\site-packages\\pandas\\io\\parsers\\readers.py:1442\u001b[0m, in \u001b[0;36mTextFileReader.__init__\u001b[1;34m(self, f, engine, **kwds)\u001b[0m\n\u001b[0;32m   1439\u001b[0m     \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhas_index_names\u001b[39m\u001b[38;5;124m\"\u001b[39m] \u001b[38;5;241m=\u001b[39m kwds[\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mhas_index_names\u001b[39m\u001b[38;5;124m\"\u001b[39m]\n\u001b[0;32m   1441\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles: IOHandles \u001b[38;5;241m|\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m \u001b[38;5;241m=\u001b[39m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[1;32m-> 1442\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_engine \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39m_make_engine(f, \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mengine)\n",
      "File \u001b[1;32mc:\\Users\\davin\\.conda\\Lib\\site-packages\\pandas\\io\\parsers\\readers.py:1735\u001b[0m, in \u001b[0;36mTextFileReader._make_engine\u001b[1;34m(self, f, engine)\u001b[0m\n\u001b[0;32m   1733\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mb\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m mode:\n\u001b[0;32m   1734\u001b[0m         mode \u001b[38;5;241m+\u001b[39m\u001b[38;5;241m=\u001b[39m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mb\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m-> 1735\u001b[0m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles \u001b[38;5;241m=\u001b[39m get_handle(\n\u001b[0;32m   1736\u001b[0m     f,\n\u001b[0;32m   1737\u001b[0m     mode,\n\u001b[0;32m   1738\u001b[0m     encoding\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mencoding\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m),\n\u001b[0;32m   1739\u001b[0m     compression\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcompression\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m),\n\u001b[0;32m   1740\u001b[0m     memory_map\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmemory_map\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mFalse\u001b[39;00m),\n\u001b[0;32m   1741\u001b[0m     is_text\u001b[38;5;241m=\u001b[39mis_text,\n\u001b[0;32m   1742\u001b[0m     errors\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mencoding_errors\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mstrict\u001b[39m\u001b[38;5;124m\"\u001b[39m),\n\u001b[0;32m   1743\u001b[0m     storage_options\u001b[38;5;241m=\u001b[39m\u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39moptions\u001b[38;5;241m.\u001b[39mget(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mstorage_options\u001b[39m\u001b[38;5;124m\"\u001b[39m, \u001b[38;5;28;01mNone\u001b[39;00m),\n\u001b[0;32m   1744\u001b[0m )\n\u001b[0;32m   1745\u001b[0m \u001b[38;5;28;01massert\u001b[39;00m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles \u001b[38;5;129;01mis\u001b[39;00m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;28;01mNone\u001b[39;00m\n\u001b[0;32m   1746\u001b[0m f \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mself\u001b[39m\u001b[38;5;241m.\u001b[39mhandles\u001b[38;5;241m.\u001b[39mhandle\n",
      "File \u001b[1;32mc:\\Users\\davin\\.conda\\Lib\\site-packages\\pandas\\io\\common.py:856\u001b[0m, in \u001b[0;36mget_handle\u001b[1;34m(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)\u001b[0m\n\u001b[0;32m    851\u001b[0m \u001b[38;5;28;01melif\u001b[39;00m \u001b[38;5;28misinstance\u001b[39m(handle, \u001b[38;5;28mstr\u001b[39m):\n\u001b[0;32m    852\u001b[0m     \u001b[38;5;66;03m# Check whether the filename is to be opened in binary mode.\u001b[39;00m\n\u001b[0;32m    853\u001b[0m     \u001b[38;5;66;03m# Binary mode does not support 'encoding' and 'newline'.\u001b[39;00m\n\u001b[0;32m    854\u001b[0m     \u001b[38;5;28;01mif\u001b[39;00m ioargs\u001b[38;5;241m.\u001b[39mencoding \u001b[38;5;129;01mand\u001b[39;00m \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mb\u001b[39m\u001b[38;5;124m\"\u001b[39m \u001b[38;5;129;01mnot\u001b[39;00m \u001b[38;5;129;01min\u001b[39;00m ioargs\u001b[38;5;241m.\u001b[39mmode:\n\u001b[0;32m    855\u001b[0m         \u001b[38;5;66;03m# Encoding\u001b[39;00m\n\u001b[1;32m--> 856\u001b[0m         handle \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mopen\u001b[39m(\n\u001b[0;32m    857\u001b[0m             handle,\n\u001b[0;32m    858\u001b[0m             ioargs\u001b[38;5;241m.\u001b[39mmode,\n\u001b[0;32m    859\u001b[0m             encoding\u001b[38;5;241m=\u001b[39mioargs\u001b[38;5;241m.\u001b[39mencoding,\n\u001b[0;32m    860\u001b[0m             errors\u001b[38;5;241m=\u001b[39merrors,\n\u001b[0;32m    861\u001b[0m             newline\u001b[38;5;241m=\u001b[39m\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m    862\u001b[0m         )\n\u001b[0;32m    863\u001b[0m     \u001b[38;5;28;01melse\u001b[39;00m:\n\u001b[0;32m    864\u001b[0m         \u001b[38;5;66;03m# Binary mode\u001b[39;00m\n\u001b[0;32m    865\u001b[0m         handle \u001b[38;5;241m=\u001b[39m \u001b[38;5;28mopen\u001b[39m(handle, ioargs\u001b[38;5;241m.\u001b[39mmode)\n",
      "\u001b[1;31mFileNotFoundError\u001b[0m: [Errno 2] No such file or directory: 'games.csv'"
     ]
    }
   ],
   "source": [
    "import pandas as pd\n",
    "import plotly.express as px\n",
    "\n",
    "games_df = pd.read_csv('games.csv')\n",
    "popularity_df = pd.read_csv('popularity.csv')\n",
    "# Preprocess games data\n",
    "games_df = games_df[['id', 'first_release_date']].dropna()\n",
    "games_df['first_release_date'] = pd.to_datetime(games_df['first_release_date'], errors='coerce')\n",
    "games_df = games_df.dropna(subset=['first_release_date'])\n",
    "\n",
    "# Preprocess popularity data\n",
    "popularity_df = popularity_df[['game_id', 'value']]\n",
    "\n",
    "popularity_agg = popularity_df.groupby('game_id').mean().reset_index()\n",
    "\n",
    "merged_df = pd.merge(games_df, popularity_agg, left_on='id', right_on='game_id')\n",
    "\n",
    "merged_df['release_year'] = merged_df['first_release_date'].dt.year\n",
    "\n",
    "# ---- Plot 1: Line Plot (Average Popularity by Year) ----\n",
    "yearly_popularity = merged_df.groupby('release_year')['value'].mean().reset_index()\n",
    "fig_line = px.line(yearly_popularity, x='release_year', y='value',\n",
    "                   title='Average Game Popularity by Release Year',\n",
    "                   labels={'release_year': 'Release Year', 'value': 'Average Popularity'})\n",
    "fig_line.show()\n",
    "\n",
    "# Convert release date from Unix timestamp\n",
    "games_df['first_release_date'] = pd.to_datetime(games_df['first_release_date'], unit='s', errors='coerce')\n",
    "\n",
    "# Drop rows with missing release year and exclude 1970\n",
    "games_df = games_df[games_df['first_release_date'].dt.year != 1970]\n",
    "games_df['release_year'] = games_df['first_release_date'].dt.year\n",
    "\n",
    "# Group by year and name to find average aggregated rating\n",
    "df_avg_rating = games_df.groupby(['release_year', 'name'])['aggregated_rating'].mean().reset_index()\n",
    "\n",
    "# Create a larger figure\n",
    "fig = px.line(\n",
    "    df_avg_rating,\n",
    "    x='release_year',\n",
    "    y='aggregated_rating',\n",
    "    color='name',\n",
    "    title='Aggregated Game Rating by Release Year and Title',\n",
    "    labels={\n",
    "        'release_year': 'Release Year',\n",
    "        'aggregated_rating': 'Aggregated Rating'\n",
    "    },\n",
    "    width=1400,   # make the figure wider\n",
    "    height=800    # make the figure taller\n",
    ")\n",
    "\n",
    "# Update axes and traces\n",
    "fig.update_yaxes(automargin=True, layer='above traces')\n",
    "fig.update_xaxes(automargin=True, layer='above traces')\n",
    "fig.update_traces(mode='markers+lines')\n",
    "\n",
    "fig.update_layout(\n",
    "    xaxis_title='Release Year',\n",
    "    yaxis_title='Aggregated Rating',\n",
    "    title_font_size=24,\n",
    "    xaxis=dict(tickmode='linear', dtick=1)\n",
    ")\n",
    "\n",
    "fig.show()\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "base",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.4"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
