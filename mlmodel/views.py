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
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import csv
import io
import os

def index(request):
    return render(request, 'index.html')


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
        supervised_models = ['Linear Regression', 'KNN', 'SVM', 'CNN', 'Decision Tree', 'Random Forest', 'Gradient Boosting']
        # unsupervised_models = ['K-Means', 'Hierarchical Clustering', 'PCA', 'SVD', 'LDA']
        request.session['selected_features'] = selected_features
        request.session['target_feature'] = target_feature
        return render(request, 'select_model.html', {'supervised_models': supervised_models, 'selected_features': selected_features, 'target_feature': target_feature})
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
                mse = mean_squared_error(y_test, y_pred)
                # The root mean squared error
                rmse = np.sqrt(mse)
                # The coefficient of determination: 1 is perfect prediction
                r2 = r2_score(y_test, y_pred)
                y_test = y_test.iloc[:, 0]  # assuming y_test is a DataFrame with one column
                y_pred = y_pred.ravel()  # assuming y_pred is a numpy array
                sort_indices = np.argsort(y_test)
                sorted_y_test = y_test.iloc[sort_indices]
                sorted_y_pred = y_pred[sort_indices]


                plt.figure(figsize=(10, 5))
                plt.plot(range(len(sorted_y_test)), sorted_y_test, label='Actual Stress')
                plt.plot(range(len(sorted_y_pred)), sorted_y_pred, label='Predicted Stress')
                plt.xlabel('Index')
                plt.ylabel('Stress')
                plt.title('Actual vs. Predicted Stress (Linear Regression)')
                plt.legend()
                plt.savefig('media/line_plot.png')

                # 2. Scatter plot
                plt.figure(figsize=(10, 5))
                plt.scatter(y_test, y_pred)
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.savefig('media/scatter_plot.png')
                
            elif selected_supervised_model == 'Decision Tree':
                from sklearn.tree import DecisionTreeRegressor
                model = DecisionTreeRegressor(random_state=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)
                modelname = "Decision Tree"
                mse = mean_squared_error(y_test, y_pred)
                # The root mean squared error
                rmse = np.sqrt(mse)
                # The coefficient of determination: 1 is perfect prediction
                r2 = r2_score(y_test, y_pred)
                y_test = y_test.iloc[:, 0]  # assuming y_test is a DataFrame with one column
                y_pred = y_pred.ravel()  # assuming y_pred is a numpy array
                sort_indices = np.argsort(y_test)
                sorted_y_test = y_test.iloc[sort_indices]
                sorted_y_pred = y_pred[sort_indices]


                plt.figure(figsize=(10, 5))
                plt.plot(range(len(sorted_y_test)), sorted_y_test, label='Actual Stress')
                plt.plot(range(len(sorted_y_pred)), sorted_y_pred, label='Predicted Stress')
                plt.xlabel('Index')
                plt.ylabel('Stress')
                plt.title('Actual vs. Predicted Stress (Decision Tree)')
                plt.legend()


                plt.savefig('media/line_plot.png')

                # 2. Scatter plot
                plt.figure(figsize=(10, 5))
                plt.scatter(y_test, y_pred)
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.savefig('media/scatter_plot.png')

            elif selected_supervised_model == 'Random Forest':
                from sklearn.ensemble import RandomForestRegressor
                model = RandomForestRegressor(random_state=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)
                modelname = "Random Forest"
                mse = mean_squared_error(y_test, y_pred)
                # The root mean squared error
                rmse = np.sqrt(mse)
                # The coefficient of determination: 1 is perfect prediction
                r2 = r2_score(y_test, y_pred)
                y_test = y_test.iloc[:, 0]  # assuming y_test is a DataFrame with one column
                y_pred = y_pred.ravel()  # assuming y_pred is a numpy array
                sort_indices = np.argsort(y_test)
                sorted_y_test = y_test.iloc[sort_indices]
                sorted_y_pred = y_pred[sort_indices]


                plt.figure(figsize=(10, 5))
                plt.plot(range(len(sorted_y_test)), sorted_y_test, label='Actual Stress')
                plt.plot(range(len(sorted_y_pred)), sorted_y_pred, label='Predicted Stress')
                plt.xlabel('Index')
                plt.ylabel('Stress')
                plt.title('Actual vs. Predicted Stress (Random Forest)')
                plt.legend()


                plt.savefig('media/line_plot.png')

                # 2. Scatter plot
                plt.figure(figsize=(10, 5))
                plt.scatter(y_test, y_pred)
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.savefig('media/scatter_plot.png')

            elif selected_supervised_model == 'Gradient Boosting':
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(random_state=0)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)
                modelname = "Gradient Boosting"
                mse = mean_squared_error(y_test, y_pred)
                # The root mean squared error
                rmse = np.sqrt(mse)
                # The coefficient of determination: 1 is perfect prediction
                r2 = r2_score(y_test, y_pred)
                y_test = y_test.iloc[:, 0]  # assuming y_test is a DataFrame with one column
                y_pred = y_pred.ravel()  # assuming y_pred is a numpy array
                sort_indices = np.argsort(y_test)
                sorted_y_test = y_test.iloc[sort_indices]
                sorted_y_pred = y_pred[sort_indices]


                plt.figure(figsize=(10, 5))
                plt.plot(range(len(sorted_y_test)), sorted_y_test, label='Actual Stress')
                plt.plot(range(len(sorted_y_pred)), sorted_y_pred, label='Predicted Stress')
                plt.xlabel('Index')
                plt.ylabel('Stress')
                plt.title('Actual vs. Predicted Stress (Gradient Boosting)')
                plt.legend()


                plt.savefig('media/line_plot.png')

                # 2. Scatter plot
                plt.figure(figsize=(10, 5))
                plt.scatter(y_test, y_pred)
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.savefig('media/scatter_plot.png')
                
            elif selected_supervised_model == 'KNN':
                from sklearn.neighbors import KNeighborsRegressor
                model = KNeighborsRegressor(n_neighbors=3)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)
                modelname = "KNN"
                mse = mean_squared_error(y_test, y_pred)
                # The root mean squared error
                rmse = np.sqrt(mse)
                # The coefficient of determination: 1 is perfect prediction
                
                r2 = r2_score(y_test, y_pred)
                y_test = y_test.iloc[:, 0]  # assuming y_test is a DataFrame with one column
                y_pred = y_pred.ravel()  # assuming y_pred is a numpy array
                sort_indices = np.argsort(y_test)
                sorted_y_test = y_test.iloc[sort_indices]
                sorted_y_pred = y_pred[sort_indices]


                plt.figure(figsize=(10, 5))
                plt.plot(range(len(sorted_y_test)), sorted_y_test, label='Actual Stress')
                plt.plot(range(len(sorted_y_pred)), sorted_y_pred, label='Predicted Stress')
                plt.xlabel('Index')
                plt.ylabel('Stress')
                plt.title('Actual vs. Predicted Stress (KNN Regression)')
                plt.legend()


                plt.savefig('media/line_plot.png')

                # 2. Scatter plot
                plt.figure(figsize=(10, 5))
                plt.scatter(y_test, y_pred)
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.savefig('media/scatter_plot.png')

            elif selected_supervised_model == 'SVM':
                from sklearn.svm import SVR
                model = SVR(kernel='linear')
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                accuracy = model.score(X_test, y_test)
                modelname = "SVM"
                mse = mean_squared_error(y_test, y_pred)
                # The root mean squared error
                rmse = np.sqrt(mse)
                # The coefficient of determination: 1 is perfect prediction
                r2 = r2_score(y_test, y_pred)
                y_test = y_test.iloc[:, 0]  # assuming y_test is a DataFrame with one column
                y_pred = y_pred.ravel()  # assuming y_pred is a numpy array
                sort_indices = np.argsort(y_test)
                sorted_y_test = y_test.iloc[sort_indices]
                sorted_y_pred = y_pred[sort_indices]


                plt.figure(figsize=(10, 5))
                plt.plot(range(len(sorted_y_test)), sorted_y_test, label='Actual Stress')
                plt.plot(range(len(sorted_y_pred)), sorted_y_pred, label='Predicted Stress')
                plt.xlabel('Index')
                plt.ylabel('Stress')
                plt.title('Actual vs. Predicted Stress (SVM)')
                plt.legend()


                plt.savefig('media/line_plot.png')

                # 2. Scatter plot
                plt.figure(figsize=(10, 5))
                plt.scatter(y_test, y_pred)
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.savefig('media/scatter_plot.png')
                
            elif selected_supervised_model == 'CNN':
                from keras.models import Sequential
                from keras.layers import Dense, Conv2D, Flatten
                model = Sequential()
                model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,1)))
                model.add(Conv2D(32, kernel_size=3, activation='relu'))
                model.add(Flatten())
                model.add(Dense(1))
                model.compile(optimizer='adam', loss='mean_squared_error')
                model.fit(X_train, y_train, epochs=3)
                y_pred = model.predict(X_test)
                accuracy = model.evaluate(X_test, y_test)
                modelname = "CNN"
                mse = mean_squared_error(y_test, y_pred)
                # The root mean squared error
                rmse = np.sqrt(mse)
                # The coefficient of determination: 1 is perfect prediction
                r2 = r2_score(y_test, y_pred)
                y_test = y_test.iloc[:, 0]  # assuming y_test is a DataFrame with one column
                y_pred = y_pred.ravel()  # assuming y_pred is a numpy array
                sort_indices = np.argsort(y_test)
                sorted_y_test = y_test.iloc[sort_indices]
                sorted_y_pred = y_pred[sort_indices]


                plt.figure(figsize=(10, 5))
                plt.plot(range(len(sorted_y_test)), sorted_y_test, label='Actual Stress')
                plt.plot(range(len(sorted_y_pred)), sorted_y_pred, label='Predicted Stress')
                plt.xlabel('Index')
                plt.ylabel('Stress')
                plt.title('Actual vs. Predicted Stress (CNN)')
                plt.legend()


                plt.savefig('media/line_plot.png')

                # 2. Scatter plot
                plt.figure(figsize=(10, 5))
                plt.scatter(y_test, y_pred)
                plt.xlabel('True Values')
                plt.ylabel('Predicted Values')
                plt.savefig('media/scatter_plot.png')
                history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, verbose=0)  # change this as per your requirements
                plt.figure(figsize=(10, 5))
                plt.plot(history.history['loss'], label='Training Loss')
                plt.plot(history.history['val_loss'], label='Validation Loss')
                plt.legend()
                plt.savefig('media/training_vs_validation_loss_plot.png')

            
            
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
                    'mse': mse,
                    'rmse': rmse,
                    'r2': r2,

                }

            return render(request, 'model_performance.html', context)
        else:
            return redirect('load_dataset')
    else:
        return redirect('select_model')
