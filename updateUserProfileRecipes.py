from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import *
from scipy.sparse import coo_matrix, vstack, csr_matrix
from users.models import PersonalizedSuggestions
from django.contrib.auth.models import User
from recipe.models import Recipe
import tensorflow.compat.v1 as tf
from libreco.data import random_split, DatasetPure, split_by_ratio, DataInfo
from libreco.algorithms import NCF  # pure data, algorithm SVD++ and NCF
from libreco.evaluation import evaluate
from recipe.models import Recipe, Rating
from django.contrib.auth.models import User
import pandas as pd
import numpy as np
import joblib
import time
import warnings
from scipy import stats
warnings.filterwarnings("ignore")
start_time = time.time()

user = User.objects.all()[0]  # Assume Spagb to be the user
recipe20IdList = joblib.load('allRecipe_20IDs.obj')

path = 'Collaborative Filtering Files'

# Recommender method

def give_rec(userid, title, cos, indices, recipes):
    idx = indices[title]

    cos_scores = list(enumerate(cos[idx]))

    cos_scores = sorted(cos_scores, key=lambda x: x[1], reverse = True)

    cos_scores = cos_scores[1:16]

    print('CBR: Checking ncf model')

    recipe_indices = [i[0] for i in cos_scores]

    recipe_indices = check_ncf(userid, recipe_indices)  

    return recipes['Name'].iloc[recipe_indices]



def check_ncf(userid, list_of_itemids):

    # Import new user data to retrain the model
    print('CBR: Importing new user data')

    new_data = pd.DataFrame(None, columns=['user', 'item', 'label'])

    user = User.objects.all().filter(id=userid)[0]
    rating_list = user.rating_history.all()

    for review in rating_list:
      userid = review.author.id
      recipeid = review.recipe.recipe_id
      label = review.rating
      new_data = new_data.append({'user': userid, 'item': recipeid, 'label': label}, ignore_index=True)

    train, test = split_by_ratio(new_data, test_size=0.2)
    for i in range(3):
        test = test.append(train.iloc[i])

    # Loading Data Info
    print('CBR: Loading data info')
    tf.reset_default_graph()
    data_info = DataInfo.load(path)

    train_data, data_info = DatasetPure.build_trainset(train, revolution=True, data_info=data_info, merge_behavior=True)
    test_data = DatasetPure.build_testset(test, revolution=True, data_info=data_info)

    print('CBR: New data_info: ', data_info)

    # Rebuilding with new dimensions
    print('CBR: Rebuilding ncf with new dimensions')

    model = NCF('rating', data_info, embed_size=16, n_epochs=3, lr=0.001, lr_decay=False, reg=None, 
              batch_size=256, num_neg=1, use_bn=True, dropout_rate=None, hidden_units="128,64,32",
              tf_sess_config=None)

    model.rebuild_graph(path=path, model_name='ncf', full_assign=True)

    model.fit(train_data, verbose=2, shuffle=True, eval_data=test_data,
              metrics=['rmse', 'mae', 'r2'])

    # Predictions
    print('CBR: Performing predictions')

    list_of_preds = []
    # list_of_itemids = [52417, 107559, 191318, 515167, 54983]  #test

    for item in list_of_itemids:
      pred = model.predict(user=userid, item=item)
      print(f'prediction: {(item, pred)}')
      list_of_preds.append((item, pred))

    sorted_preds = sorted(list_of_preds, key=lambda x: x[1], reverse = True)
    
    return_items = []
    i = 0
    for pair in sorted_preds:
        if i == 20:
            break
        else:
            return_items.append(pair[0])
            i+=1

    return return_items

def start():
    # Beginning

    print('CBR: Checking if the user has rated 3 recipes')

    user = User.objects.all().filter(id=1)[0]
    rating_list = user.rating_history.all()
    if len(rating_list) < 3:
        print('CBR: No changes made - need to rate 3 recipes first')

    print('CBR: Loading the data')

    tfv = TfidfVectorizer(strip_accents='unicode', analyzer='word', stop_words = 'english', token_pattern=r'(?u)\b\w\w\w+\b')

    df_recipes = pd.read_csv('dataset\\recipes_20 split\\disk1.csv', engine='python', encoding='utf-8', error_bad_lines=False)
    df_recipes.columns = ['Recipe_ID', 'Name', 'minutes', 'contributor_id', 'submitted', 'tags', 'nutrition','steps', 'description', 'ingredients']

    df = df_recipes.assign(info = df_recipes.ingredients.astype(str) + ' ' + df_recipes.steps.astype(str))
    df.drop(['Name', 'tags', 'steps', 'ingredients', 'description', 'Recipe_ID', 'minutes', 'submitted', 'nutrition'], axis=1, inplace=True)

    tfv_matrix = tfv.fit_transform(df['info'])

    print('CBR: Fetching user ratings')
    id_ratings = {}     # Key: Recipe ID, Value: Rating
    num = 0
    limit = 15

    for rating in user.rating_history.all().order_by('-date_added'):
        id_ratings[rating.recipe.recipe_id] =  rating.rating
        num += 1
        if num>limit: break

    #id_ratings = {15116: 5.0, 148135: 2.0, 14945: 4.0, 187401: 2.0, 223883: 3.0}    #Test data


    raw_ratings = list(id_ratings.values())    # Create a list of all the rating values for aggregates
    
    mean = np.mean(raw_ratings)
    median = np.median(raw_ratings)
    mode = stats.mode(raw_ratings)[0][0]

    if mean and median and mode < 2:    # User has not liked any recipes on the website. Find similar users instead
        # LOAD NCF MODEL

        print(f"CBR: User's ratings are mostly negative. Using ncf model instead.")

        # Important to reset graph if model is loaded in the same shell.
        tf.compat.v1.reset_default_graph()

        # load data_info
        data_info = DataInfo.load(path)
        print(f'CBR: Data info of ncf model: {data_info}')

        # load model, should specify the model name, e.g., DeepFM
        model = NCF.load(path=path, model_name="ncf",
                            data_info=data_info, manual=True)

        list_of_preds = []

        for pair in model.recommend_user(user=1, n_rec=200):
            list_of_preds.append(pair[0])

        all_recipes = Recipe.objects.all()
        counter = 0

        user.suggestions.all().delete()

        for recipeid in list_of_preds:
            try:
                if counter == 20:
                    break
                recip = all_recipes.filter(recipe_id=recipeid)[0]
                p = PersonalizedSuggestions(user_id=user.id, suggestion=recip)
                p.save()             
                counter += 1
            except:
                pass

        print('CBR: Done!')
        print('CBR: Updated user recipes using ncf, rather than user profile')
        return


    # OTHERWISE, continue

    # Normalize the ratings, taking to account the rating tendencies for a given user
    # Recipes that have ratings that are below the user mean, will not be zeroed - not considered.

    print('CBR: Normalizing ratings')

    normalized_ratings = {}

    for item in id_ratings.items():
        normalized_ratings[item[0]] = item[1]-mean    # item[0] is key, item[1] is value


    weights = {}
    for item in normalized_ratings.items():
        if item[1]<0:
            weights[item[0]] = 0
        else: 
            weights[item[0]] = item[1]


    for index, row in df_recipes.iterrows():
        id = row['Recipe_ID']
        if id in weights:   # If the recipe ID is in the normalized ratings (weights) key list
            continue
        else:
            weights[id] = 0


    # Convert all recipe IDs to index values

    index_ratings = {}
    i = 0
    for value in weights.values():
        index_ratings[i] = value
        i+=1
     

    # Apply weights onto each recipe
    print('CBR: Applying weights onto recipes')
    tfv_matrix_coo = coo_matrix( tfv_matrix )   # For iterating
    tfv_matrix_csr = csr_matrix( tfv_matrix_coo )   # For alterations

    for i,j,v in zip(tfv_matrix_coo.row, tfv_matrix_coo.col, tfv_matrix_coo.data):
        tfv_matrix_csr[i,j] = v * index_ratings[i]

    new_coo_matrix = coo_matrix( tfv_matrix_csr )


    # Create the user profile recipe
    # Takes the weighted sum for each feature that was above average rating
    print('CBR: Creating user profile')
    sum_row = [0 for i in range(tfv_matrix_coo.shape[1])]
    counter = 0

    for i,j,v in zip(new_coo_matrix.row, new_coo_matrix.col, new_coo_matrix.data):
        sum_row[j] = sum_row[j] + v   # for each feature j, we want to sum up all of the values associated with it

    user_row = coo_matrix(sum_row) # Create a sparse matrix row


    # Now we need to add this user profile to the original tfv_matrix, so that we can compute the cosine similarity
    print('CBR: Stacking user profile')
    final_tfv_matrix = vstack([tfv_matrix, user_row])


    tfv = final_tfv_matrix.astype(np.float16)
    print('CBR: Creating cosine similarity matrix')
    cosine_sim = cosine_similarity(tfv, tfv)

    # Dummy data - so that, User Profile is now technically a recipe
    d = {'Recipe_ID': 200000000000000, 'Name': 'User Profile', 'minutes': 0, 'contributor_id': 0, 'submitted': '', 'steps': '', 'description': '', 'ingredients': ''}
    ser = pd.Series(data=d)
    # Appending this User Profile recipe to the recipe matrix
    df_recipes = df_recipes.append(ser, ignore_index=True)

    recipes = df_recipes.drop(columns=['Recipe_ID', 'minutes', 'contributor_id', 'submitted', 'nutrition'])

    indices = pd.Series(recipes.index, index=recipes['Name']).drop_duplicates()


    # Recommendation part
    print('CBR: Feeding in user profile')
    recommendation = give_rec(1, 'User Profile', cosine_sim, indices, recipes)    # User with id 1: spagb (admin)
    print('CBR: Almost done...')
    list_ids = recommendation.index
    list_names = recommendation.values
    length = len(list_ids)
    user.suggestions.all().delete()
    all_recipes = Recipe.objects.all()
    i=0
    successful = False

    for j in range(length):
        # Sort out disks to include recipes_20
        if i == 20:
            successful = True
            break

        if list_names[j] in recipe20IdList: ######## CHECK, IF RECIPE ID BELONGS TO RECIPES_20 IDs FILE
            print(f'CBR: Recipe ID: {list_names[j], list_ids[j]} is in recipe_20_list')
            try:
                recip = all_recipes.filter(recipe_name=list_names[j])[0]
                p = PersonalizedSuggestions(user_id=user.id, suggestion=recip)
                p.save()                
                i += 1
            except:
                continue
        else:
            print(f'CBR: Recipe ID: {list_names[j], list_ids[j]} is NOT recipe_20_list')

    if successful == True:
        print(f'CBR: Done!')
        print(f'CBR: Suggested recipes have been updated for {user}!')
    end = time.time()
    runtime = end - start_time
    print(f"\nCBR: Runtime: {runtime:.2f} seconds)")

# TO EXECUTE IN DJANGO SHELL:
# exec(open('updateUserProfileRecipes.py').read())