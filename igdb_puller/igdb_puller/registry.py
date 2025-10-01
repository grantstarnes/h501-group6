from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class TableDef:
    endpoint: str
    fields: str
    sort: str = "id asc"

# Central place to add/modify tables (you can extend this anytime)
TABLES: Dict[str, TableDef] = {
    "games": TableDef(
        endpoint="games",
        fields=(
            "id,name,slug,first_release_date,updated_at,"
            "total_rating,total_rating_count,aggregated_rating,aggregated_rating_count,"
            "follows,hypes,platforms,genres,status"
        ),
    ),
    "game_time_to_beats": TableDef(
        endpoint="game_time_to_beats",
        fields="id,game_id,normally,hastily,completely,count,created_at,updated_at",
    ),
    "popularity_primitives": TableDef(
        endpoint="popularity_primitives",
        fields="id,game_id,external_popularity_source,popularity_type,value,calculated_at,updated_at",
    ),
    "popularity_types": TableDef(
        endpoint="popularity_types",
        fields="id,name,external_popularity_source,created_at,updated_at",
    ),
    "genres": TableDef(
        endpoint="genres",
        fields="checksum,created_at,name,slug,updated_at,url",
    ),
    "platforms": TableDef(
        endpoint="platforms",
        fields=(
            "abbreviation,alternative_name,checksum,created_at,generation,name,platform_family,"
            "platform_logo,platform_type,slug,summary,updated_at,url,versions,websites"
        ),
    ),
    "platform_families": TableDef(
        endpoint="platform_families",
        fields="checksum,name,slug",
    ),
    "platform_logos": TableDef(
        endpoint="platform_logos",
        fields="alpha_channel,animated,checksum,height,image_id,url,width",
    ),
    "platform_types": TableDef(
        endpoint="platform_types",
        fields="checksum,created_at,name,updated_at",
    ),
    "platform_versions": TableDef(
        endpoint="platform_versions",
        fields=(
            "checksum,companies,connectivity,cpu,graphics,main_manufacturer,media,memory,name,os,"
            "output,platform_logo,platform_version_release_dates,resolutions,slug,sound,storage,summary,url"
        ),
    ),
}


# ---------- dependency configuration ----------
DIRECT_BY_GAME = [
    "artworks","covers","external_games","game_localizations","involved_companies",
    "language_supports","multiplayer_modes","release_dates","screenshots","game_videos",
    "websites",
]
DIRECT_BY_GAME_ID = ["game_time_to_beats","popularity_primitives"]

LOOKUPS_FROM_GAMES = {
    # endpoint            column on games
    "age_ratings":        "age_ratings",
    "genres":             "genres",
    "keywords":           "keywords",
    "platforms":          "platforms",
    "player_perspectives":"player_perspectives",
    "game_modes":         "game_modes",
    "franchises":         "franchises",   # adjust to 'franchise' if your schema uses singular
    "collections":        "collection",   # games.collection is a single id
}

# second-order lookups we can optionally pull after first-order tables are loaded
SECOND_ORDER = {
    # from involved_companies -> companies, then company_logos, company_websites
    "companies": {"source_df": "df_involved_companies", "source_col": "company", "filter": "id"},
    "company_logos": {"source_df": "df_companies", "source_col": "logo", "filter": "id"},
    "company_websites": {"source_df": "df_companies", "source_col": "id", "filter": "company"},

    # from platforms -> platform_* (pull selectively as needed)
    "platform_logos": {"source_df": "df_platforms", "source_col": "logo", "filter": "id"},
    "platform_versions": {"source_df": "df_platforms", "source_col": "versions", "filter": "id"},
    "platform_version_release_dates": {"source_df": "df_platform_versions", "source_col": "id", "filter": "platform_version"},
    "platform_websites": {"source_df": "df_platforms", "source_col": "id", "filter": "platform"},
    "platform_families": {"source_df": "df_platforms", "source_col": "platform_family", "filter": "id"},
    "platform_types": {"source_df": "df_platforms", "source_col": "category", "filter": "id"},  # if needed

    # from release_dates -> release_date_* (regions/statuses/date_formats)
    "release_date_regions": {"source_df": "df_release_dates", "source_col": "region", "filter": "id"},
    "release_date_statuses": {"source_df": "df_release_dates", "source_col": "status", "filter": "id"},
    "date_formats": {"source_df": "df_release_dates", "source_col": "date_format", "filter": "id"},

    # from websites -> website_types
    "website_types": {"source_df": "df_websites", "source_col": "category", "filter": "id"},
}