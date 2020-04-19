from django.shortcuts import render
from django.views.generic.list import ListView
from django.core.paginator import Paginator
from django.http import Http404
from mongoengine import connect

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
            'business_name': business.name,
            'reviews': reviews
        }
        return render(request, 'business_reviews.html', context=context)
    else:
        return Http404("Negocio no encontrado.")

def business_neighbours(request, business_id):
    context = {}
    return render(request, 'business_neighbours.html', context=context)