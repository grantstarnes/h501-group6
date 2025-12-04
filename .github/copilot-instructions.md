# Copilot Instructions for h501-group6

## Project Overview
This repository is a modular data pipeline and recommender system for video games, built for INFO-H501. It consists of:
- **IGDB Puller**: A package for downloading IGDB tables via CLI or Python API, exporting to CSV, NDJSON, or Parquet.
- **Content-Based Recommender**: Python scripts for recommending games using IGDB data and similarity metrics.
- **Streamlit App**: (in `igdb_puller/streamlit_app/`) for interactive recommendations and data exploration.

## Key Components & Data Flow
- **IGDB Data Fetching**: Use the CLI (`igdb-puller ...`) or `pull_table` API to fetch and export tables. Registry-driven schemas in `igdb_puller/registry.py`.
- **Environment Setup**: Requires Python >=3.9. Install dependencies with `pip install -e .` in `igdb_puller/`.
- **Secrets**: Set `TWITCH_CLIENT_ID` and `TWITCH_CLIENT_SECRET` in a `.env` file for IGDB API access.
- **Data Files**: Main recommender expects `games.csv`, `genres.csv`, and `game_time_to_beats.csv` in the working directory.

## Developer Workflows
- **Pulling Data**:
  - CLI: `igdb-puller games --max-rows 10000 --fmt csv --out games.csv`
  - API: `from igdb_puller import pull_table; pull_table("games", ...)`
- **Extending IGDB Tables**: Add new `TableDef` to `igdb_puller/registry.py`.
- **Running Recommender**: Execute `content_recommender.py` directly. Example usage is in the file.
- **Streamlit App**: Run `streamlit run igdb_puller/streamlit_app/streamlit_app.py`.

## Project-Specific Patterns
- **Fixed CSV Schema**: Each export run uses a fixed schema to avoid column drift.
- **NDJSON for Normalization**: Use NDJSON format for intermediate steps to avoid schema issues.
- **Parquet for Analytics**: Parquet is preferred for fast analytics workflows.
- **Genre Processing**: Genres are pipe-separated strings, converted to lists and binarized for ML.
- **Similarity Calculation**: Uses cosine similarity on combined genre and scaled numeric features.

## Integration Points
- **External APIs**: IGDB via Twitch credentials.
- **Python Packages**: `requests`, `pandas`, `python-dotenv`, `pyarrow`, `scikit-learn`, `numpy`.
- **Streamlit**: For interactive UI (see `streamlit_app/`).

## Example: Adding a New IGDB Table
1. Edit `igdb_puller/registry.py` to add a new table definition.
2. Pull data via CLI or API as shown above.

## References
- Main logic: `igdb_puller/igdb_puller/` (data fetching, exporting)
- Recommender: `content_recommender.py`
- Streamlit UI: `igdb_puller/streamlit_app/`
- Registry: `igdb_puller/igdb_puller/registry.py`

---
For unclear or missing conventions, review `README.md` files and code comments in key modules. Please ask for feedback or clarification if any section is incomplete or ambiguous.
