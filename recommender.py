from datetime import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from persona import admin, child, dad
import warnings
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
warnings.simplefilter(action='ignore', category=FutureWarning)


def get_persona():
    """This function accepts no arguments. It welcomes the user, and returns their choice of user persona"""
    return admin
    # Welcome User
    print("Welcome to Nick's recommendation system.")
    print("Which persona would you like to test?\n\t[A]dmin\n\t[C]hild\n\t[D]ad")
    choice = input("Enter the string; the first character will suffice.").strip(' ').lower()
    while choice[0] not in ['a', 'c', 'd']:
        choice = input("Enter the string; the first character will suffice.").strip(' ').lower()
    if choice[0] == 'a':
        return admin
    elif choice[0] == 'c':
        return child
    elif choice[0] == 'd':
        return dad
    else:
        pass
    return


def encode_genres(df, profile=False):
    """This function accepts the streaming data, and returns encoded columns based on the genres each respective movie
    was listed under."""

    # Copy dataframe and select relevant columns and data
    temp_df = df.copy()
    temp_df = temp_df.loc[temp_df.type == "Movie"]
    temp_df = temp_df.loc[temp_df.country == "United States"]
    temp_df = temp_df[['title', 'country', 'listed_in']]

    all_genres = []
    for index, data in temp_df.iterrows():

        # Access each rows genres
        weight = 1.1
        genres = temp_df.loc[index, 'listed_in']
        genres = [x.strip(' ') for x in genres.split(',')]
        # As long as its not redundant genre, apply weight and append the column to master genre list
        for genre in genres:
            if genre != 'Movies':
                temp_df.loc[index, genre] = weight
                if genre not in all_genres:
                    all_genres.append(genre)
                weight -= 0.05
            else:
                pass



    # Drop unneeded data
    temp_df = temp_df.drop('listed_in', 1).drop('country', 1)

    # If columns are not present in a movie or profile, add any missing columns for NA handling
    if profile:
        for col in all_genres:
            if col not in temp_df.columns.to_list():
                temp_df[col] = " "

    temp_df = temp_df.fillna(0)

    return temp_df


def generate_profile(movies, ratings, encoded_genres):
    """This function accepts a persona's watch history: titles and ratings, gathers the encoded genres data
     for those movies, performs a transposition and dot calculation between the ratings. A series is returned with
     the form Genre: Weighted Dot product"""

    # Locate encoded data for user's movies
    encoded_user_genres = encoded_genres.loc[encoded_genres.title.isin(movies)].drop('title', 1)
    # Perform transposition
    encoded_user_genres = encoded_user_genres.transpose()
    # Calculate dot product between movie vectors and rating for each movie, and sum by column.
    user_profile = encoded_user_genres.dot(ratings)

    return user_profile


def generate_recs(encoded_genres, user_profile):
    """This function accepts the encoded genres df and a user profile, and returns the top recommendations
    for that user profile based on their most highly reviewed genres."""

    new_df = pd.DataFrame()
    new_df['title'] = encoded_genres.title
    encoded_genres = encoded_genres.drop('title', 1)

    similarities = []
    user_vector = np.array([user_profile.values])

    for i in range(len(new_df.title)):
        row_vector = np.array([encoded_genres.iloc[i].values])
        similarity = cosine_similarity(row_vector, user_vector)[0][0]
        similarities.append(similarity)

    new_df['similarity'] = similarities

    new_df = new_df.sort_values(by='similarity', ascending=False)

    return new_df


def merge_data(df_recommended, df_master_data, previously_watched):
    temp = df_recommended.merge(df_master_data[['title', 'description', 'director', 'cast']], on='title').set_index(
        'title')
    try:
        temp = temp.drop(previously_watched)
    except KeyError:
        pass

    temp['similarity'] = temp['similarity'].round(4)
    temp['similarity'] = temp['similarity'].apply(lambda row: '{:.2%}'.format(row))

    return temp.iloc[0:100, ]


def generate_plot(recs):
    # Generate plot of similarities
    similarities = [x for x in list(recs.similarity) if x > 0.4]

    # with plt.style.context('dark_background'):
    fig, ax = plt.subplots(figsize=(18, 18))

    plt.plot(similarities, color='r')

    # ax.get_xaxis().set_visible(False)
    ax.yaxis.set_major_formatter(mticks.PercentFormatter(1))

    # Set plot settings
    plt.title('Similarity To User Profile', size=30)
    plt.ylabel('Cosine Similarity', size=26)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel('Movies Processed', size=26)
    # Save plot for report
    plt.savefig('similarity_plot.png')


def generate_HTML(recs):
    # Generate html of dataframe to add to report
    recs_html = recs.to_html()

    # The img in style is superceded by in line style
    html_style = """
    <style>
        h1 {float:left; font-family: sans-serif; text-align:center; margin-left:10%;}
        img {height:90px; width:90px; float:right; position:relative; margin-right:5%;}
        table {border-collapse:collapse;}
        th {color: white; background: red; text-align:center;}
        tr {text-align:center;}
    </style>
    """

    # Insert title and logo here
    html_header = """
        <header><h1>FlickFinder.</h1>
        <img src='logo.png'>
    """

    # In line here controls style rather than style block
    html_plot = """
        <img src='similarity_plot.png' style='height:87%; width:90%; margin-left: 5%;'></header>
    """

    # Open html file
    with open('movie_recommendations.html', 'w') as file:

        # Set CSS Style Configurations
        file.write(html_style)
        # Create header logo and title
        file.write(html_header)
        # Add visualization to report
        file.write(html_plot)
        # Add dataframe to report
        file.write(recs_html)
        # Close file
        file.close()


def main():
    """This is the workflow of the program, invoked by the main() at the end of the script."""

    # Get user persona for demo...
    persona = get_persona()

    # Set Timer for runtime analysis
    init_timestamp = datetime.now()

    # Read data
    df = pd.read_csv('/Users/nicholascampa/Desktop/Datasets/DS496/netflix_titles.csv')

    # Encode Genres
    encoded_genres = encode_genres(df, profile=False)

    # Create User Profile
    user_profile = generate_profile(persona['titles'], persona['ratings'], encoded_genres)

    # Generate Recommendations
    df_recommendations = generate_recs(encoded_genres, user_profile)

    # Generate report
    df_merge = merge_data(df_recommendations, df, persona['titles'])

    generate_HTML(df_merge)

    # Report runtime
    print(datetime.now() - init_timestamp)

main()
