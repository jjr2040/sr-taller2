from django.shortcuts import render
from django.views.generic.list import ListView
from django.core.paginator import Paginator
from django.http import Http404, JsonResponse
from datetime import date
from mongoengine import connect
import uuid
from .recommenders import neighbours_for_business, business_recommended_reviews

from .models import Business, Review
connect('yelp', host='mongodb://mongo')

businesses = Business.objects.order_by('name').all()

# Create your views here.
class HomeView(ListView):
    template_name = 'index.html'
    paginate_by = 10
    
    def get_queryset(self):
        business_name = self.request.GET.get('name')

        if business_name:
            return businesses.filter(name=business_name)

        return businesses

def business_reviews(request, business_id):
    business = businesses.filter(business_id=business_id).first()
    if(business):
        reviews = Review.objects(business_id=business_id)
        context = {
            'business_id': business.business_id,
            'business_name': business.name,
            'reviews': reviews
        }
        return render(request, 'business_reviews.html', context=context)
    else:
        return Http404("Negocio no encontrado.")

def business_neighbours(request, business_id):
    business = businesses.filter(business_id=business_id).first()
    neighbours = neighbours_for_business(business_id)
    context = {
        'business_id': business_id,
        'business_name': business.name,
        'neighbours': neighbours
    }
    return render(request, 'business_neighbours.html', context=context)


def business_recommended_reviews_view(request, business_id):
    business = businesses.filter(business_id=business_id).first()

    reviews, keyword_list = business_recommended_reviews(business_id)

    context = {
        'business_name': business.name,
        'reviews': reviews,
        'keyword_list': keyword_list
    }
    return render(request, 'business_recommended_reviews.html', context=context)


def add_business(request):
    new_business = Business(business_id=str(uuid.uuid4()),name=request.GET['name'], address=request.GET['address'], city=request.GET['city'], state=request.GET['state'], postal_code=request.GET['postalcode'], categories=request.GET['categories'].split(','), review_count=0, stars=0)
    new_business.save()

    data = { 'allright': True }
    return JsonResponse(data)

def add_review(request):
    business_id = request.GET['business_id']
    stars = float(request.GET['stars'])

    business = businesses.filter(business_id=business_id).first()
    reviews = reviews = Review.objects(business_id=business_id)

    total_reviews = len(reviews) + 1
    total_stars = 0.0
    for review in reviews:
        total_stars += review.stars
    total_stars += stars
    average_stars = total_stars / total_reviews

    new_review = Review(review_id=str(uuid.uuid4()), user_id='V34qejxNsCbcgD8C0HVk-Q', business_id=business_id , stars=stars, useful=0, funny=0, cool=0, text=request.GET['text'], date=date.today().strftime("%m/%d/%Y, %H:%M:%S"))
    new_review.save()

    business.review_count = total_reviews
    business.stars = average_stars
    business.save()

    data = { 'allright': True }
    return JsonResponse(data)