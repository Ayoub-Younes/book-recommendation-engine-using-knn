# import libraries (you may add additional imports but you may not have to)
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import requests
import zipfile
import os

# get data files

current_dir = os.path.join(os.getcwd(), "Book Recommendation Engine using KNN")
file_name = os.path.join(current_dir, "book-crossings.zip")
extracted_dir = os.path.join(current_dir, "book-crossings")
books_filename = os.path.join(extracted_dir, 'BX-Books.csv')
ratings_filename = os.path.join(extracted_dir, 'BX-Book-Ratings.csv')



'''# URL of the zip file
url = "https://cdn.freecodecamp.org/project-data/books/book-crossings.zip"
response = requests.get(url)

with open(file_name, "wb") as file:
  file.write(response.content)

# Unzip the file
with zipfile.ZipFile(file_name, 'r') as zip_ref:
    zip_ref.extractall(extracted_dir)

# Optionally, remove the zip file after extraction
os.remove(file_name)'''

# import csv data into dataframes

df_books = pd.read_csv(
    books_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['isbn', 'title', 'author'],
    usecols=['isbn', 'title', 'author'],
    dtype={'isbn': 'str', 'title': 'str', 'author': 'str'})

df_ratings = pd.read_csv(
   ratings_filename,
    encoding = "ISO-8859-1",
    sep=";",
    header=0,
    names=['user', 'isbn', 'rating'],
    usecols=['user', 'isbn', 'rating'],
    dtype={'user': 'int32', 'isbn': 'str', 'rating': 'float32'})

# Remove duplicated books
df_books.drop_duplicates(subset=['title'], keep='first', inplace=True)

# remove from the dataset users with less than 200 ratings and books with less than 100 ratings.
books_filter = df_ratings['isbn'].value_counts().reset_index()
books_filter  = books_filter[books_filter['count'] >= 100]['isbn'].to_numpy()
users_filter = df_ratings['user'].value_counts().reset_index()
users_filter  = users_filter[users_filter['count'] >= 200]['user'].to_numpy()
df_ratings = df_ratings[df_ratings['user'].isin(users_filter) & df_ratings['isbn'].isin(books_filter)]

# Group the data by isbn
df = df_ratings.pivot_table(index='isbn', columns='user', values='rating').fillna(0)
df = df.reset_index()

def get_recommends(book):

    # Preparing the model
    isbn = df_books[df_books['title'] == book]['isbn'].values[0]
    index = df[df['isbn'] == isbn].index[0]
    ratings_matrix = df.drop('isbn', axis=1).values
    model = NearestNeighbors(n_neighbors=6, metric='cosine')
    model.fit(ratings_matrix)
    distances, indices = model.kneighbors([ratings_matrix[index]])

    #preparing the output
    distances = np.flip(distances[0],0)[:4]
    indices = np.flip(indices[0],0)[:4]
    books_isbn = [df.loc[i]['isbn'] for i in indices]
    books = [df_books[df_books['isbn'] == isbn]['title'].values[0] for isbn in books_isbn]
    result = [[x, y] for x, y in zip(books, distances)]
    return [book, result]


print(get_recommends("Where the Heart Is (Oprah's Book Club (Paperback))"))