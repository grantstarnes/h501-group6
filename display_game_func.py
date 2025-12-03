from IPython.display import HTML


def display_game_cover_git(game_id, df_games_param, df_covers_param):
    """
    Displays the cover image and information for a given game ID.
    This version takes df_games_param and df_covers_param as arguments
    and does not calculate genre popularity internally.

    Args:
        game_id (int): The ID of the game.
        df_games_param (pd.DataFrame): The DataFrame containing game information.
        df_covers_param (pd.DataFrame): The DataFrame containing cover information.
    """
    # Find the cover information for the given game ID
    cover_info = df_covers_param[df_covers_param['game'] == game_id]
    game_info = df_games_param[df_games_param['id'] == game_id]

    if not cover_info.empty and not game_info.empty:
        # Get the URL of the first cover found
        image_url = cover_info['url'].iloc[0].replace('t_thumb', 't_cover_big') # Use a larger size

        # Get game information
        name = game_info['name'].iloc[0]
        popularity = game_info['popularity_value'].iloc[0]
        log_popularity = game_info['log_popularity_value'].iloc[0]
        age_ratings = game_info['age_rating_list'].iloc[0]
        genres_str = game_info['genres'].iloc[0]
        release_year = game_info['release_year'].iloc[0]

        # Display the image and information using HTML
        display(HTML(f'''
            <div style="display: flex;">
                <img src="https:{image_url}" width="150px" style="margin-right: 20px;">
                <div>
                    <h3>{name}</h3>
                    <p><strong>Popularity:</strong> {popularity:.8f}</p>
                    <p><strong>Log Popularity:</strong> {log_popularity:.8f}</p>
                    <p><strong>Age Ratings:</strong> {age_ratings}</p>
                    <p><strong>Genres:</strong> {genres_str}</p>
                    <p><strong>Release Year:</strong> {release_year}</p>
                </div>
            </div>
        '''))
    else:
        print(f"No cover or information found for game ID: {game_id}")