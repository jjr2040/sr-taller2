import numpy as np
import pandas as pd
import re

from surprise import Reader
from surprise import Dataset
from surprise import accuracy
from surprise import SVD

from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import chi2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

import pymongo
from pymongo import MongoClient

client = MongoClient(host='mongodb://mongo')
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
    if reviews_group.shape[0] > 0:
        reviews_df_filtered = reviews_df.merge(reviews_group, how="inner", on="business_id").drop('count', axis=1)
        reviews_df_filtered = pd.concat([reviews_df,reviews_df_filtered])
        reviews_df_filtered = reviews_df_filtered.drop_duplicates(keep=False)
    else:
        reviews_df_filtered = reviews_df

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

    # Get reviews of the business neightbours selected
    business_reviews_to_save = db.reviews.find({ "business_id": {"$in": business_neighbours} })

    business['recommended_reviews'] = list(business_reviews_to_save)

    result = db.businesses.replace_one({ 'business_id': business_id }, business)

    # success_message = 'Modified businesses (%d) ids' % len(result.modified_count)

    return list(businesses_collection.find({ "business_id": {"$in": business_neighbours} }))


def business_recommended_reviews(business_id):

    business = businesses_collection.find_one({ 'business_id': business_id })

    recommended_reviews = list(business.get('recommended_reviews'))

    kn_reviews_df = pd.DataFrame(recommended_reviews)
    kn_reviews_df['useful_mean'] = kn_reviews_df['useful'].mean()
    kn_reviews_df['is_useful'] = kn_reviews_df['useful'] > kn_reviews_df['useful_mean']

    # Build TF-IDF for reviews

    reviews_texts=kn_reviews_df['text']

    # Clean especial characters

    clear_text_list=[]
    for x in reviews_texts:
        y=re.sub(r'[,.!-?¿¡"&$%#\n\t]','',x.lower())
        clear_text_list.append(y)
    
    # Se crea una matrix de conteo de (textos x palabras), se ignoran las palabras que aparezcan en el 85% de los documentos(no relevantes)

    cv=CountVectorizer(max_df=0.85, stop_words="english")
    word_count_vector=cv.fit_transform(clear_text_list)

    # Se convierte la matriz dispersa a dataframe

    keywords_df = pd.DataFrame(word_count_vector.toarray())

    #Calculo de TF- IDF sobre la matriz dispersa, smooth_idf modifica la formula matematica False para no ignorar completamente los terminos que aparecen en todos los textos
    #Se utiliza normalizacion coseno
    #use_idf true para calcular la ponderacion inversa de frecuencia

    Tfidf_transformer=TfidfTransformer(smooth_idf=False,use_idf=True)
    Tfidf_transformer.fit(word_count_vector)

    keyword_tf_idf_df = pd.DataFrame(Tfidf_transformer.transform(word_count_vector).toarray())

    # Find reviews for a business

    keywords_df_with_class = pd.concat([kn_reviews_df['review_id'], kn_reviews_df['is_useful'], keyword_tf_idf_df], axis=1)

    #Con esta configuración se utilizan los 3 vecinos más cercanos, con distancia euclidiana
    knn_clasif=KNeighborsClassifier(3)

    # Fit recibe la matriz de entrenamiento y la clase objetivo
    knn_clasif.fit(keyword_tf_idf_df, keywords_df_with_class['is_useful'])

    # llamamos predict sobre  los test , creando una nueva columna en el dataframe de test
    keywords_df_with_class['predict'] = knn_clasif.predict(keyword_tf_idf_df)
    reviews_ids = list(keywords_df_with_class[ keywords_df_with_class.is_useful & keywords_df_with_class.predict]['review_id'])
    useful_reviews = db.reviews.find({ "review_id": {"$in": reviews_ids} })

    #Get the number of occurrences of a keyword in all reviews
    lista_de_tuplas=list(zip(cv.get_feature_names(),keywords_df.sum().to_list()))
    #sort
    lista_de_tuplas.sort(key=lambda tup: tup[1], reverse=True)
    list_of_lists = [list(elem) for elem in lista_de_tuplas]

    # Add relevant keywords to reviews
    feature_names=cv.get_feature_names()
    reviews_keywords=[]
    for text in clear_text_list:
        tf_idf_vector=Tfidf_transformer.transform(cv.transform([text]))
        sorted_items=sort_coo(tf_idf_vector.tocoo())
        keywords=extract_topn_from_vector(feature_names,sorted_items,10)
        reviews_keywords.append( list( keywords.keys() ) )
    
    relevant_keywords_df = pd.DataFrame(reviews_keywords)
    relevant_keywords_df = pd.concat([keywords_df_with_class['review_id'], relevant_keywords_df], axis=1)

    for index, row in relevant_keywords_df.iterrows():
        db.reviews.update_one({ 'review_id': row['review_id'] }, 
                              { '$set': { 'relevant_keywords': reviews_keywords[index] }})
    
    return list(useful_reviews), list_of_lists[:100]


def sort_coo(coo_matrix):
    tuples=zip(coo_matrix.col,coo_matrix.data)
    return sorted(tuples,key=lambda x:(x[1],x[0]),reverse=True)


#Extra de todas las keywords las n-keywords mas relevantes(TF-IDF)
def extract_topn_from_vector(feature_names, sorted_items,topn=10):
    sorted_items=sorted_items[:topn]
    score_vals=[]
    feature_vals=[]   
    for idx,score in sorted_items:
        score_vals.append(round(score,3))
        feature_vals.append(feature_names[idx])
    results={}
    for idx in range(len(feature_vals)):
        results[feature_vals[idx]]=score_vals[idx]
    return results 
