import pandas as pd
import numpy as np


def prepare_data():
    songsCsv = pd.read_csv('songs.csv')
    songsCsv.rename(columns={'_id': 'songId'}, inplace=True)

    listenedCsv = pd.read_csv('listened.csv')
    likedCsv = pd.read_csv('liked.csv')

    ratingsRawCsv = pd.concat([listenedCsv, likedCsv])
    ratingsRawCsv = ratingsRawCsv.dropna(axis=1)

    songsDataset = songsCsv.merge(ratingsRawCsv, on="songId")
    songsDataset['rating'] = 0
    datasetWithDuplicates = pd.DataFrame(songsDataset, columns=[
        "songId", "name", "genre._id", "genre.name", "artist._id", "artist.name", "userId", "rating", "image"])
    dataset = datasetWithDuplicates.drop_duplicates(keep='first').reset_index()

    datasetWithDuplicates.to_csv('raw_ratings.csv', index=False)
    print('updated raw_ratings.csv')
    
    numberOfDuplicatesInsideOfDataset = datasetWithDuplicates.pivot_table(
        index=['songId', 'userId'], aggfunc='size')

    for index in range(len(numberOfDuplicatesInsideOfDataset)):
        dataset.loc[index, 'rating'] = numberOfDuplicatesInsideOfDataset[index]

    dataset.to_csv('rating.csv', index=False)
    print('updated ratings.csv')
    return 
