import mongoengine

from mongoengine import ReferenceField

class Business (mongoengine.Document):
    business_id = mongoengine.StringField()
    name = mongoengine.StringField()
    address = mongoengine.StringField()
    city = mongoengine.StringField()
    state = mongoengine.StringField()
    postal_code = mongoengine.StringField()
    latitude = mongoengine.DecimalField()
    longitude = mongoengine.DecimalField()
    stars = mongoengine.FloatField()
    review_count = mongoengine.IntField()
    is_open = mongoengine.IntField()
    attributes = mongoengine.ListField()
    categories = mongoengine.ListField()
    hours = mongoengine.ListField()
    recommended_reviews = mongoengine.ListField(ReferenceField('Review'))
    meta = {
        'collection': 'businesses',
        'indexes': [
            'name',
            'business_id'
        ]
    }

class Review (mongoengine.Document):
    review_id = mongoengine.StringField()
    user_id = mongoengine.StringField()
    business_id = mongoengine.StringField()
    stars = mongoengine.FloatField()
    useful = mongoengine.IntField()
    funny = mongoengine.IntField()
    cool = mongoengine.IntField()
    text = mongoengine.StringField()
    date = mongoengine.StringField()
    relevant_keywords = mongoengine.ListField()
    meta = {
        'collection': 'reviews',
        'indexes': [
            'business_id',
            'review_id'
        ]
    }


