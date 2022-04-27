from datetime import *
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from persona import admin, child, dad
import warnings
from fpdf import *
import matplotlib.pyplot as plt
import matplotlib.ticker as mticks
import pdfkit

warnings.simplefilter(action='ignore', category=FutureWarning)


def gets_persona():
    """This function accepts no arguments. It welcomes the user, and returns their choice of user persona"""
    return child
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


def generate_report(df_recommended, df_master_data, previously_watched):
    print(previously_watched)
    temp = df_recommended.merge(df_master_data[['title', 'description', 'director', 'cast']], on='title').set_index(
        'title')
    temp = temp.drop(previously_watched)

    file = open('report.txt', 'w')
    for n in range(10):
        title = temp.index[n]
        director = temp.iloc[n, 2]
        cast = temp.iloc[n, 3].split(',')[0:3]

        file.write(f"The #{n+1} recommended movie was: {title}.\n")
        file.write(f"It was directed by {director} and features {cast[0]}, {cast[1]}, and {cast[2]}.\n")
        file.write(f"{temp.iloc[n, 1]}.\n")
        file.write("\n")
    file.close()

    print(temp.head())
    return temp


def generatePDF(recs):

    # Generate plot of similarities
    similarities = [x for x in list(recs.similarity) if x > 0.4]

    #with plt.style.context('dark_background'):
    fig, ax = plt.subplots(figsize=(24, 18))

    plt.plot(similarities, color='r')

    # ax.get_xaxis().set_visible(False)
    ax.yaxis.set_major_formatter(mticks.PercentFormatter(1.0))

    plt.title('Similarity To User Profile', size=30)
    plt.ylabel('Cosine Similarity (shown as %)', size=26)
    plt.yticks(fontsize=20)
    plt.xticks(fontsize=20)
    plt.xlabel('Movies Processed', size=26)
    plt.savefig('test.svg')

    # Generate html of dataframe to add to pdf
    recs_html = recs.to_html()
    html_header = "<h1 style='text-align:center;'>FlickFinder</h1>"
    html_byline = "<h2 style='text-align:center;'>Created by Nick Campa 2022</h2>"
    html = "<img src='test.svg' style='height:80%; width:100%'> <br><br>"

    with open('test.html', 'w') as file:
        file.write(html_header)
        file.write(html_byline)
        file.write(html)
        file.write(recs_html)
        file.close()


def main():
    """This is the workflow of the program, invoked by the main() at the end of the script."""

    # Get user persona for demo...
    persona = gets_persona()

    # Set Timer for runtime analysis
    init_timestamp = datetime.now()

    # Read data
    df = pd.read_csv('/Users/nicholascampa/Desktop/Datasets/DS496/netflix_titles.csv')

    # Encode Genres
    encoded_genres = encode_genres(df, profile=False)

    # Create User Profile
    user_profile = generate_profile(persona['titles'], persona['ratings'], encoded_genres)

    # Generate Recommendations
    df_recommendations = (generate_recs(encoded_genres, user_profile))

    # Generate report
    df_merge = generate_report(df_recommendations, df, persona['titles'])

    generatePDF(df_merge)

    # Report runtime
    print(datetime.now() - init_timestamp)

main()
