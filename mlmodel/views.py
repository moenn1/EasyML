from django.http import JsonResponse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import numpy as np
from django.shortcuts import render, redirect
from sklearn.model_selection import train_test_split
from django.core.files.storage import default_storage
from django.conf import settings
from django.http import HttpResponse
from sklearn.linear_model import LinearRegression
from sklearn import metrics

import csv
import io
import os

def index(request):
    return render(request, 'index.html')

# def load_dataset(request):
#     if request.method == 'POST':
#         csv_file = request.FILES['csv_file']
#         file_name = default_storage.save('tmp/' + csv_file.name, csv_file)
#         request.session['file_name'] = file_name
#         data_set = csv_file.read().decode('UTF-8')
#         io_string = io.StringIO(data_set)
#         reader = csv.reader(io_string, delimiter=',', quotechar="|")
#         header = next(reader)  # Gets the first row

#         # Save the header in the session
#         request.session['header'] = header

#         # Now you can pass the header to your template
#         return render(request, 'dataset.html', {'header': header})
#     else:
#         # Display the form
#         return render(request, 'index.html')


def load_dataset(request):
    if request.method == 'POST':
        csv_file = request.FILES['csv_file']
        
        data_set = csv_file.read().decode('UTF-8')
        io_string = io.StringIO(data_set)
        reader = csv.reader(io_string, delimiter=',', quotechar="|")
        header = next(reader)  # Gets the first row  
        file_name = default_storage.save('tmp/' + csv_file.name, csv_file)
        request.session['file_name'] = file_name  # Save the file name to the session
        request.session['header'] = header
        # Now you can pass the header to your template
        return render(request, 'dataset.html', {'header': header})
    else:
        # Display the form
        return render(request, 'index.html')

    
def select_features(request):
    if request.method == 'POST':
        selected_features = request.POST.getlist('features')
        # Save selected features in the session
        request.session['selected_features'] = selected_features
        return redirect('select_target')
    else:
        return redirect('load_dataset')
    

def select_target(request):
    selected_features = request.session.get('selected_features', [])
    header = request.session.get('header', [])  # Retrieve the header from the session

    remaining_features = [feature for feature in header if feature not in selected_features]
    return render(request, 'select_target.html', {'header': remaining_features})


def select_model(request):
    if request.method == 'POST':
        target_feature = request.POST.getlist('features')
        selected_features = request.session.get('selected_features', [])
        request.session['target_feature'] = target_feature
        supervised_models = ['Linear Regression', 'Logistic Regression', 'Decision Tree', 'Random Forest', 'Support Vector Machine']
        unsupervised_models = ['K-Means', 'Hierarchical Clustering', 'PCA', 'SVD', 'LDA']
        request.session['selected_features'] = selected_features
        request.session['target_feature'] = target_feature
        return render(request, 'select_model.html', {'supervised_models': supervised_models, 'unsupervised_models': unsupervised_models, 'selected_features': selected_features, 'target_feature': target_feature})
    else:
        return redirect('select_target')


# import other necessary libraries depending on your models

def output(request):
    if request.method == 'POST':
        selected_supervised_model = request.POST['selected_supervised_model']
        #selected_unsupervised_model = request.POST['selected_unsupervised_model']
        train_split_size = float(request.POST['train_split_size'])
        test_split_size = float(request.POST['test_split_size'])

        selected_features = request.session.get('selected_features', [])
        target_feature = request.session.get('target_feature')

        file_name = request.session.get('file_name', None)
        if file_name:
            file_path = os.path.join(settings.MEDIA_ROOT, file_name)
            df = pd.read_csv(file_path)

            # Extract relevant columns
            X = df[selected_features]
            y = df[target_feature]

            # Split the data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_split_size, random_state=42)

            # Train your model
            # Depending on the selected model, you would train it differently
            # This is just an example with a linear regression model from sklearn
            if selected_supervised_model == 'Linear Regression':
                print("Linear Regression")
                print(X_train.head())
                print(y_train.head())
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                   #CALCULATE THE ACCURACY, recall, precision, f1 score
                accuracy = model.score(X_test, y_test)
                print("Accuracy: ", accuracy)
                modelname = "Linear Regression"
                
            
                

            # If it is unsupervised learning, you might cluster the data for example
            # elif selected_unsupervised_model == 'K-Means':
            #     from sklearn.cluster import KMeans
            #     model = KMeans(n_clusters=3)
            #     model.fit(X)

            # You can continue this process for other models

            # Now, you can use your model to make predictions, evaluate it, etc.
            # For example, you might want to render a new template showing the model's performance

            context = {
                'model': modelname,
                'X_test': X_test,
                'y_test': y_test,
                'selected_features': selected_features,
                'target_feature': target_feature,
                'selected_supervised_model': selected_supervised_model,
                'train_split_size': train_split_size,
                'test_split_size': test_split_size,
                'accuracy': accuracy,
            }

            return render(request, 'model_performance.html', context)
        else:
            return redirect('load_dataset')
    else:
        return redirect('select_model')
