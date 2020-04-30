import numpy as np
import pandas as pd

from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise import SVD

from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

import pymongo
from pymongo import MongoClient

client = MongoClient()
db = client.yelp

businesses_collection = db.businesses

def neighbours_for_business(business_id):
    
    business = db.businesses.find_one({ 'business_id': business_id })

    # Get categories array
    categories = business.get('categories')

    # Get Businesses in categories ($all for all in categories, $in for any in categories)
    businesses_in_categories_df = pd.DataFrame(list(businesses_collection.find({ "categories": { "$all": categories } } )))
    business_ids_in_categories = businesses_in_categories_df['business_id'].tolist()

    # Now obtain all useful (greater than zero) reviews from the group of businesses_in_categories_df
    reviews_df = pd.DataFrame(list(db.reviews.find({'business_id': {'$in': business_ids_in_categories}})))
    reviews_df.stars = reviews_df.stars.astype('int')
    
    # group reviews dataframe by number of reviews and remove those businesses that have just one review
    reviews_group = reviews_df.groupby('business_id')['stars'].agg({'count'}).sort_values(by='count', ascending=False)
    reviews_group = reviews_group[reviews_group['count'] < 2].reset_index()
    reviews_df_filtered = reviews_df.merge(reviews_group, how="inner", on="business_id").drop('count', axis=1)
    reviews_df_filtered = pd.concat([reviews_df,reviews_df_filtered])
    reviews_df_filtered = reviews_df_filtered.drop_duplicates(keep=False)

    # Create Surprise data
    reader = Reader( rating_scale= (1,5))
    train_df, test_df = train_test_split(reviews_df_filtered[['business_id','user_id','stars']], test_size=.3, stratify=reviews_df_filtered['business_id'])
    train_set = Dataset.load_from_df(train_df, reader)
    train_set = train_set.build_full_trainset()

    # Setup algorithm using SVD
    svd = SVD( n_factors = 5, n_epochs = 200, biased = True, lr_all = 0.005, reg_all = 0, init_mean = 0, init_std_dev = 0.01, verbose = False )
    svd.fit(train_set)

    # Obtain pu matrix and get our business vector
    pu = svd.pu
    business_vector_innerid = train_set.to_inner_uid(business.get('business_id'))
    business_vector = pu[business_vector_innerid]

    # Calculate the euclidean distance between our business and everything else
    distances = pairwise_distances(pu,business_vector.reshape(1, -1),'euclidean')
    # Obtain an indirect array with the indices of the sorted distances array
    ordered_distances_index = np.argsort(distances.flatten())

    # If number of neighbours is greater than number_of_neighbours, return first number_of_neighbours, otherwise return what we have
    number_of_neighbours = 10
    neighbour_indices = []
    if ordered_distances_index.shape[0] > number_of_neighbours:
        neighbour_indices = ordered_distances_index[1:number_of_neighbours+1]
    else:
        neighbour_indices = ordered_distances_index

    # Get neighbour ids of selected business
    business_neighbours_ids = (train_set.to_raw_uid(rid) for rid in neighbour_indices)
    business_neighbours = []
    for neighbour in business_neighbours_ids:
        business_neighbours.append(neighbour)

    return list(businesses_collection.find({ "business_id": {"$in": business_neighbours} }))