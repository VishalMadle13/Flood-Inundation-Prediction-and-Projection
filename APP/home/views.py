from django.shortcuts import render, HttpResponse, redirect

# Create your views here.
def home(request):
    # print("mpas function working")
    return render(request,'home.html')

def about(request):
    return render(request,'about.html')