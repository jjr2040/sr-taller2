from django.shortcuts import render
from django.views.generic.list import ListView
from django.core.paginator import Paginator
from django.http import Http404, JsonResponse
from mongoengine import connect
import uuid

from .models import Business, Review
connect('yelp')

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
    context = {}
    return render(request, 'business_neighbours.html', context=context)

def add_business(request):
    new_business = Business(business_id=str(uuid.uuid4()),name=request.GET['name'], address=request.GET['address'], city=request.GET['city'], state=request.GET['state'], postal_code=request.GET['postalcode'], categories=request.GET['categories'], review_count=0, stars=0)
    new_business.save()

    data = { 'allright': True }
    return JsonResponse(data)

def add_review(request):
    new_review = Review(review_id=str(uuid.uuid4()), business_id=request.GET['business_id'], stars=float(request.GET['stars']), text=request.GET['text'])
    new_review.save()

    data = { 'allright': True }
    return JsonResponse(data)