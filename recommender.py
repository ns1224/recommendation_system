import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from persona import admin, child, dad


def gets_persona():
    """This function accepts no arguments. It welcomes the user, and returns their choice of user persona"""

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


def report_recs(df_recommended, df_master_data):
    """This function accepts a dataframe of recommendations based on descending similarities 0-1.
    It returns detailed data about each movie and its cast, director, etc."""
    join = df_master_data.loc[df_master_data.title.isin(list(df_recommended.title))]
    return join


def main():
    """This is the workflow of the program, invoked by the main() at the end of the script."""

    # Get user persona for demo...
    persona = gets_persona()

    # Read data
    df = pd.read_csv('/Users/nicholascampa/Desktop/Datasets/DS496/netflix_titles.csv')

    # Encode Genres
    encoded_genres = encode_genres(df, profile=False)

    # Create User Profile
    user_profile = generate_profile(persona['titles'], persona['ratings'], encoded_genres)

    # Generate Recommendations
    df_recommendations = (generate_recs(encoded_genres, user_profile))

    print(report_recs(df_recommendations, df))


main()
