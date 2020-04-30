from django.core.management.base import BaseCommand, CommandError

import numpy as np
import pandas as pd

from surprise import Reader
from surprise import Dataset
from surprise import accuracy
# from surprise.model_selection import train_test_split
from surprise import SVD

from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

import pymongo
from pymongo import MongoClient

client = MongoClient()
db = client.yelp

class Command(BaseCommand):
    help = 'Recalculate neighbours and reviews'

    def handle(self, *args, **options):

        business_recomendation_records = []
        
        # Gather businesses data from DB
        # create new collection with split categories
        # db.temp_businesses.drop()
        # agr = [{'$project': {'business_id':1,'name':1,'address':1,'city':1,'state':1,'postal_code':1,'latitude':1,'longitude':1,'stars':1,'review_count':1,'is_open':1,'attributes':1,'categories': {'$split':["$categories",","]},'hours':1}},{'$out': 'temp_businesses'}]
        # businesses_agr = db.businesses.aggregate(agr)
        # businesses_collection = db.temp_businesses
        businesses_collection = db.businesses
        businesses_df = pd.DataFrame(list(businesses_collection.find({ "review_count": { "$gt": 1 }, 'categories': { '$ne': None } })))
        # businesses_count = businesses_df.shape[0]

        for index in range(len(businesses_df)):
            
            business = businesses_df.iloc[index]

            self.stdout.write(business['business_id'])

            # Get categories array
            categories = business.get('categories')
            
            # Get Businesses in categories ($all for all in categories, $in for any in categories)
            businesses_in_categories_df = pd.DataFrame(list(businesses_collection.find({ "categories": {"$all": categories} })))
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

            if reviews_df_filtered.empty:
                continue

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

            business_neighbours_to_save = businesses_collection.find({ "business_id": {"$in": business_neighbours} })
            
            # Get reviews of the business neightbours selected
            business_reviews_to_save = db.reviews.find({ "business_id": {"$in": business_neighbours} })

            # Insert new record

            neighbours_list = list(business_neighbours_to_save)
            neighbours_reviews_list = list(business_reviews_to_save)

            business_recomendation_record = {
                'business_id': business.get('business_id'),
                'neighbours': neighbours_list,
                'reviews': neighbours_reviews_list
            }

            business_recomendation_records.append(business_recomendation_record)
            
            info_message = '[%d] Recalculated business with (%d) neighbours and (%d) reviews' % (
                index,
                len(neighbours_list), 
                len(neighbours_reviews_list)
            )

            self.stdout.write(self.style.WARNING(info_message))

        
        result = db.businessReviews.insert_many(business_recomendation_records)

        success_message = 'Finished inserting (%d) ids' % len(result.inserted_ids)

        self.stdout.write(self.style.SUCCESS(success_message))