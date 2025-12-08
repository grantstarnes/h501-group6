# h501-group6

Abstract: The main purpose of the project was to identify if there is a relationship between how fast players beat video games and the ratings they receive. After creating visuals to analyze the results, there didn't seem to be much correlation between time to beat and ratings. However, the information on the app remains applicable to both video game consumers and producers. Consumers can find recommended games based on details they input, see the results, and other games they might consider playing. Videogame producers can use this site to see how well their games compare to competitors or to other games they previously produced, using visuals and trends across different videogames and genres, as well as their success rates. 

Data Description: The data used was from the IGDB database, which provided downloadable CSV files containing extensive video game data. We used the applicable CSVs from the website, removed unnecessary columns, and cleaned up null values by either removing them completely or replacing them with alternative values.  

Algorithm Description: The app is a tool that allows any user to find information about video games. Users can type in a desired game and find its ratings, release date, genre, and other details. The main goal of our app is to recommend games based on the user's input and share statistics to help them decide whether to buy the game.

Tools Used:
  - Streamlit: Transforms the team's Python code into a web application so all our work is visible on the output web app.
  - Python: The coding language used to develop the Streamlit app, create the recommendation algorithm, and present visuals on the app. 
  - AWS: The cloud service to connect the IGDB data to our local devices, so visualizations are compatible and able to upload to the Streamlit app.
  - Git: Tracked all the code changes for the team, so that in troubleshooting situations, they can potentially refer back to previous code versions.
  - GitHub: The cloud-based, centralized hub for the team to share everyone's code in the same space, so we stay up to date on everyone's progress.

Ethical Concerns: Our code raises a few ethical concerns. One example of evaluation bias was during the data cleaning process, the "age_rating" column was removed, which is the European rating system for video games. While the argument for removing this column was to keep the ratings US-based, since most hypothetical users are from the US, we are completely dismissing another way to evaluate video games that could potentially strengthen our analysis and app credibility. In the future, we will conduct the same analysis and create separate visuals/results for the US and European ratings. In other words, do not drop the "age_rating" column.
