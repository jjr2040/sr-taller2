from django.urls import path
from . import views

urlpatterns = [
    path('', views.HomeView.as_view(), name='index'),
    path('reviews/<business_id>/', views.business_reviews, name='business_reviews'),
    path('neighbours/<business_id>/', views.business_neighbours, name='business_neighbours'),
    path('recommended-reviews/<business_id>/', views.business_recommended_reviews_view, name='business_recommended_reviews'),
    path('addBusiness', views.add_business),
    path('addReview', views.add_review),
]