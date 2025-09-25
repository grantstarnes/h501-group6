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