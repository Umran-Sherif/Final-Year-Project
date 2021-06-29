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
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

import joblib, time, warnings, os, sys, math, random

warnings.filterwarnings("ignore")
start_time = time.time()

limit = 16
list_of_thresholds = [0.3, 0.4, 0.5]
allusers = User.objects.all()
num_of_users = 4

# userlist = [None for i in range(num_of_users)]
# counter = 0
# while counter != num_of_users:
#     user = allusers[random.randint(1, 68330)]
#     #if len(user.rating_history.all()) == 1:
#     if len(user.rating_history.all()) > 3:
#         userlist[counter] = user
#         counter+=1

userlist = [None for i in range(num_of_users)]
counter = 0
# userl = [28, 38, 281, 326] # low mean
# userl = [899, 977, 3052, 6092] # low mean & > 16 ratings
userl = [6, 271, 25, 36] # high mean & 16 ratings
for userid in userl:
    user = allusers.filter(id=userid)[0]
    userlist[counter] = user
    counter+=1

#user = User.objects.all()[0]  # Assume Spagb to be the user
recipe20IdList = joblib.load('allRecipe_20IDs.obj')

# Removes erronous error detection when calling model.predict

class HiddenPrints:
    def __enter__(self):
     self._original_stdout = sys.stdout
     sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
     sys.stdout.close()
     sys.stdout = self._original_stdout

# RECOMMENDS A SET OF SIMILAR RECIPES

def give_rec(userid, title, cos, indices, recipes, k, use_model):
    idx = indices[title]

    cos_scores = list(enumerate(cos[idx]))

    cos_scores = sorted(cos_scores, key=lambda x: x[1], reverse = True)

    cos_scores = cos_scores[1:31]

    recipe_indices = [i[0] for i in cos_scores]

    return_items = []
    
    if use_model == True:
        list_of_preds = []

        for item in recipe_indices:
            try:
                with HiddenPrints():
                    pred = model.predict(user=userid, item=item)
                # print(f'prediction: {(item, pred)}')
                list_of_preds.append((item, pred))
            except:
                pass

        sorted_preds = sorted(list_of_preds, key=lambda x: x[1], reverse = True)

        i = 0
        for pair in sorted_preds:
            if i == k:
                break
            else:
                return_items.append(pair[0])
                i+=1

    else:
        return_items = []

        i = 0
        for pair in recipe_indices:
            if i == k:
                break
            else:
                return_items.append(pair)
                i+=1


    return recipes['Name'].iloc[return_items]   # iloc returns the recipe name and index value - as this is how pandas series works
    # recipes also works on an index basis. so if you want the recipe id, you need to loc it from df_recipes


# CREATE A PROFILE FOR EACH USER

def createProfile(user):

    num = 0

    print(f"HYBRID: Fetching user: ID: {user.id}, Name: {user.username}'s ratings")
    id_ratings = {}     # Key: Recipe ID, Value: Rating
    for rating in user.rating_history.all():
        id_ratings[rating.recipe.recipe_id] = rating.rating
        num += 1
        if num+1 > limit: break


    # NORMALIZE RATINGS


    print('HYBRID: Normalizing ratings')

    raw_ratings = list(id_ratings.values())    # Create a list of all the rating values for aggregates

    mean = np.mean(raw_ratings)
    normalized_ratings = {}

    for item in id_ratings.items():
        normalized_ratings[item[0]] = item[1]-mean    # item[0] is key, item[1] is value


    # CREATE WEIGHTS


    weights = {}
    for item in normalized_ratings.items():
        if item[1]<0:
            weights[item[0]] = 0
        else: 
            #weights[item[0]] = item[1]
            weights[item[0]] = 1


    for index, row in df_recipes.iterrows():
        id = row['Recipe_ID']
        if id in weights:   # If the recipe ID is in the normalized ratings (weights) key list
            continue
        else:
            weights[id] = 0


    # CONVERT RECIPE IDS TO INDEX VALUES


    index_ratings = {}
    i = 0
    for value in weights.values():
        index_ratings[i] = value
        i+=1
     

    # APPLY USER WEIGHTS TO EACH RECIPE


    print('HYBRID: Applying weights onto recipes')
    tfv_matrix_coo = coo_matrix( tfv_matrix )   # For iterating
    tfv_matrix_csr = csr_matrix( tfv_matrix_coo )   # For alterations

    for i,j,v in zip(tfv_matrix_coo.row, tfv_matrix_coo.col, tfv_matrix_coo.data):
        tfv_matrix_csr[i,j] = v * index_ratings[i]

    new_coo_matrix = coo_matrix( tfv_matrix_csr )


    # CREATE USER PROFILE


    print('HYBRID: Creating user profile')
    sum_row = [0 for i in range(tfv_matrix_coo.shape[1])]
    counter = 0

    for i,j,v in zip(new_coo_matrix.row, new_coo_matrix.col, new_coo_matrix.data):
        sum_row[j] = sum_row[j] + v   # for each feature j, we want to sum up all of the values associated with it

    user_row = coo_matrix(sum_row) # Create a sparse matrix row

    return user_row


# BEGINNING


print('HYBRID: Loading the data')

tfv = TfidfVectorizer(strip_accents='unicode', analyzer='word', stop_words = 'english', token_pattern=r'(?u)\b\w\w\w+\b')

df_recipes = pd.read_csv('dataset\\recipes_20 split\\disk1.csv', engine='python', encoding='utf-8', error_bad_lines=False)
df_recipes.columns = ['idx', 'Name', 'Recipe_ID', 'minutes', 'submitted', 'tags', 'nutrition','steps', 'description', 'ingredients']

#df = df_recipes.assign(info = df_recipes.steps.astype(str) + ' ' + df_recipes.ingredients.astype(str) + ' ' + df_recipes.description.astype(str) + ' ' + df_recipes.nutrition.astype(str))
df = df_recipes.assign(info = df_recipes.steps.astype(str) + ' ' + df_recipes.ingredients.astype(str))
df.drop(['idx', 'Name', 'tags', 'steps', 'ingredients', 'description', 'Recipe_ID', 'minutes', 'submitted', 'nutrition'], axis=1, inplace=True)

tfv_matrix = tfv.fit_transform(df['info'])


# ADD USER PROFILE TO TFV_MATRIX


print('HYBRID: Retrieving and stacking user profiles')

c = 0
for user in userlist:
    user_row = createProfile(user)

    if c == 0:
        final_tfv_matrix = vstack([tfv_matrix, user_row])
    else:
        final_tfv_matrix = vstack([final_tfv_matrix, user_row])
    c+=1

final_tfv = final_tfv_matrix.astype(np.float16)


# CREATE SIMILARITY MATRIX


print('HYBRID: Creating cosine similarity matrix')
cosine_sim = cosine_similarity(final_tfv, final_tfv)



# CONVERT USER PROFILES INTO RECIPES. UPDATE RECIPE AND INDICE LISTS.


for i, user in enumerate(userlist):
    d = {'Recipe_ID': 200000000000000+i, 'Name': f'User Profile{i}', 'minutes': 0, 'contributor_id': 0, 'submitted': '', 'steps': '', 'description': '', 'ingredients': ''}
    ser = pd.Series(data=d)
    df_recipes = df_recipes.append(ser, ignore_index=True)

recipes = df_recipes.drop(columns=['Recipe_ID', 'minutes', 'contributor_id', 'submitted', 'nutrition'])
indices = pd.Series(recipes.index, index=recipes['Name']).drop_duplicates()



# LOAD NCF MODEL


# print('HYBRID: Loading ncf model')
# # Important to reset graph if model is loaded in the same shell.
# tf.compat.v1.reset_default_graph()
# # load data_info
# data_info = DataInfo.load(path)
# print(f'Data info of ncf model: {data_info}')

# # load model, should specify the model name, e.g., DeepFM
# model = NCF.load(path=path, model_name="ncf",
#                     data_info=data_info, manual=True)



# LOAD NCF MODEL AND RETRAIN

path = 'Collaborative Filtering Files'

print('CBR: Importing new user data')

new_data = pd.DataFrame(None, columns=['user', 'item', 'label'])

for user in userlist:
    rating_list = user.rating_history.all()

    for review in rating_list:
        userid = review.author.id
        recipeid = review.recipe.recipe_id
        label = review.rating
        new_data = new_data.append({'user': userid+1, 'item': recipeid, 'label': label}, ignore_index=True)

train, test = split_by_ratio(new_data, test_size=0.2)

random_samples = random.sample(range(1, train.shape[0]), train.shape[0]//8)     # Select one 8th of the training set
for sample_index in random_samples:
    test = test.append(train.iloc[sample_index])


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



# CREATE A DEFAULT SET OF RECOMMENDED RECIPES FOR A USER


#print('HYBRID: Feeding in user profile')
#recommendation_cos_sim = give_rec(USER, 'User Profile', cosine_sim, indices, recipes, k=10, use_model=False)    # User with id 1: spagb (admin)
#recommendation_lin_ker = give_rec(1, 'User Profile', linear_ker, indices, recipes, k=10, use_model=True)    # User with id 1: spagb (admin)



# EVALUATION METRICS: 

# AVERAGE PRECISION AT K, RECALL & NORMALIZED DISCOUNTED CUMMULATIVE GAIN.

print('HYBRID: Computing Average Precision @ K score')

arr = final_tfv_matrix.toarray()

def apatk(i, userid, k=20, threshold=0.2, type='cos', use_model=False):    # Defaults to AP@20
    sum_of_precisions = 0
    print('\n')

    for m in range(1, k+1):
        if type == 'cos':
            recommendation = give_rec(userid, f'User Profile{i}', cosine_sim, indices, recipes, k=m, use_model=use_model)
        elif type == 'lin':
            #recommendation = give_rec(1, 'User Profile', linear_ker, indices, recipes, k=m, use_model=use_model)
            return
        else:
            print('Invalid recommendation type')
            return

        precision_at_m = precision_report(recommendation, threshold)
        
        # print(f'Precision at K={m}: {precision_at_m}')

        sum_of_precisions += precision_at_m

    average_precision_at_k = sum_of_precisions / k

    print(f'\nThe AP@{k} score is: {average_precision_at_k}')
    return average_precision_at_k


def precision_report(recommendation, relevance_threshold):
    rec_features = {}   # List of features for each recommended recipe
    features_per_user_rated_recipe = {}   # List of features for each recipe the user has rated
    percentage_tuples = []   # Shows the number of common features between both the user recommended recipe and a user history recipe

    for rating in user.rating_history.all():
        recipe_name = rating.recipe.recipe_name   # We want the name, so that we can find the index value for df_recipes
        #rated_recipe_index = df_recipes.loc[df_recipes['Name'] == recipe_name]   # find the index value for a recipe name
        rated_recipe_index = recipes.loc[recipes['Name'] == recipe_name]   # find the index value for a recipe name

        if rated_recipe_index.empty == True:
            continue    # ignores recipes from disk2.csv

        rated_recipe_index = rated_recipe_index.index[0]

        features_per_user_rated_recipe[rated_recipe_index] = []   # initialize a list for the user id key

        for j in range(len( arr[rated_recipe_index] )):
            
            value = arr[rated_recipe_index][j]
            if value > 0:
                features_per_user_rated_recipe[rated_recipe_index].append(j)  # append the features that are present for each recipe

    # Append all features to each recommendation recipe index
    for index in recommendation.index:
        rec_features[index] = []
        for j in range(len(arr[index])):
            if arr[index][j] > 0:
                rec_features[index].append(j)

    for rec_index_list in rec_features:
        for feature_list_key in features_per_user_rated_recipe:
            common = list(set(rec_features[rec_index_list]) & set(features_per_user_rated_recipe[feature_list_key])) # id_ratings = {314: 5, 39606: 5, 153616: 5, 13082: 4, 3723: 1, 54193: 2, 66: 2, 50: 4, 40: 4}
            percentage = len(common) / len(features_per_user_rated_recipe[feature_list_key])
            # print( f'{(rec_index_list, feature_list_key, percentage)}' )
            percentage_tuples.append( (rec_index_list, feature_list_key, percentage) )


    list_of_relevant_ids = {}

    for index in recommendation.index:
        list_of_relevant_ids[index] = 0 # Initially zero

        for tuple in percentage_tuples:
            if tuple[0] == index:
                if tuple[2] > relevance_threshold: # The relevance check
                    list_of_relevant_ids[index] = 1

    precision_at_m = sum(list_of_relevant_ids.values()) /len(list_of_relevant_ids)
    return precision_at_m


# apatk()
# apatk(10)
# apatk(10, 0.25)
# apatk(20, 0.25)
# apatk(20, 0.2, 'lin')

def recall(i, userid, relevance_threshold=0.2, use_model=False, thresh=20):
    recommendation = give_rec(userid, f'User Profile{i}', cosine_sim, indices, recipes, k=6, use_model=use_model)
    rec_features = {}   # List of features for each recommended recipe
    features_per_user_rated_recipe = {}   # List of features for each recipe the user has rated
    percentage_tuples = []   # Shows the number of common features between both the user recommended recipe and a user history recipe
    t2 = 0

    for recip in Recipe.objects.all():
        t2+=1
        if t2 == thresh:
            break
        recipe_name = recip.recipe_name   # We want the name, so that we can find the index value for df_recipes
        #rated_recipe_index = df_recipes.loc[df_recipes['Name'] == recipe_name]   # find the index value for a recipe name
        rated_recipe_index = recipes.loc[recipes['Name'] == recipe_name]   # find the index value for a recipe name

        if rated_recipe_index.empty == True:
            continue    # ignores recipes from disk2.csv

        rated_recipe_index = rated_recipe_index.index[0]

        features_per_user_rated_recipe[rated_recipe_index] = []   # LIST FOR A DICTIONARY KEY - initialize a list for the user id key

        for j in range(len( arr[rated_recipe_index] )):
            
            value = arr[rated_recipe_index][j]
            if value > 0:
                features_per_user_rated_recipe[rated_recipe_index].append(j)  # append the features that are present for each recipe

    print('part 1 done')

    t3 = 0

    # Append all features to each recommendation recipe index
    for index in recipes.index:
        t3+=1
        if t3 == thresh:
            break
        rec_features[index] = []
        for j in range(len(arr[index])):
            if arr[index][j] > 0:
                rec_features[index].append(j)

    print('part 2 done')

    for rec_index_list in rec_features:
        for feature_list_key in features_per_user_rated_recipe:
            common = list(set(rec_features[rec_index_list]) & set(features_per_user_rated_recipe[feature_list_key])) # id_ratings = {314: 5, 39606: 5, 153616: 5, 13082: 4, 3723: 1, 54193: 2, 66: 2, 50: 4, 40: 4}
            percentage = len(common) / len(features_per_user_rated_recipe[feature_list_key])
            #print( f'{(rec_index_list, feature_list_key, percentage)}' )
            percentage_tuples.append( (rec_index_list, feature_list_key, percentage) )

    print('part 3 done')

    list_of_relevant_ids = {}

    t1 = 0

    for index in recipes.index:
        t1+=1
        if t1 == thresh:
            break
        list_of_relevant_ids[index] = 0 # Initially zero

        for tuple in percentage_tuples:
            if tuple[0] == index:
                if tuple[2] > 0.2: # The relevance check
                    list_of_relevant_ids[index] = 1

    print('part 4 done')

    print(f'Sum of relevants: {sum(list_of_relevant_ids.values())}')
    print(f'Length of relevants: {len(list_of_relevant_ids.values())}')
    print(f'List: {list_of_relevant_ids.values()}')
    recall = sum(list_of_relevant_ids.values()) / len(list_of_relevant_ids)     # Need to do: number_of_relevant_in_rec / sum (total number of relevants)
    return recall


def NDCG(ii, userid, k=6, use_model=False):
    recommendation = give_rec(userid, f'User Profile{ii}', cosine_sim, indices, recipes, k=k, use_model=use_model)
    rec_features = {}   # List of features for each recommended recipe
    features_per_user_rated_recipe = {}   # List of features for each recipe the user has rated
    percentage_tuples = []   # Shows the number of common features between both the user recommended recipe and a user history recipe

    for rating in user.rating_history.all():
        recipe_name = rating.recipe.recipe_name   # We want the name, so that we can find the index value for df_recipes
        #rated_recipe_index = df_recipes.loc[df_recipes['Name'] == recipe_name]   # find the index value for a recipe name
        rated_recipe_index = recipes.loc[recipes['Name'] == recipe_name]   # find the index value for a recipe name

        if rated_recipe_index.empty == True:
            continue    # ignores recipes from disk2.csv

        rated_recipe_index = rated_recipe_index.index[0]

        features_per_user_rated_recipe[rated_recipe_index] = []   # initialize a list for the user id key

        for j in range(len( arr[rated_recipe_index] )):
            
            value = arr[rated_recipe_index][j]
            if value > 0:
                features_per_user_rated_recipe[rated_recipe_index].append(j)  # append the features that are present for each recipe

    # Append all features to each recommendation recipe index
    for index in recommendation.index:
        rec_features[index] = []
        for j in range(len(arr[index])):
            if arr[index][j] > 0:
                rec_features[index].append(j)

    for rec_index_list in rec_features:
        for feature_list_key in features_per_user_rated_recipe:
            common = list(set(rec_features[rec_index_list]) & set(features_per_user_rated_recipe[feature_list_key])) # id_ratings = {314: 5, 39606: 5, 153616: 5, 13082: 4, 3723: 1, 54193: 2, 66: 2, 50: 4, 40: 4}
            percentage = len(common) / len(features_per_user_rated_recipe[feature_list_key])
            #print( f'{(rec_index_list, feature_list_key, percentage)}' )
            percentage_tuples.append( (rec_index_list, feature_list_key, percentage) )


    list_of_relevant_ids = {}

    for index in recommendation.index:
        
        list_of_relevant_ids[index] = 0 # Initially zero
        for tuple in percentage_tuples:
            if tuple[0] == index: # The relevance check
                if tuple[2] > 0.3 and tuple[2] < 0.4: 
                    list_of_relevant_ids[index] = 1
                elif tuple[2] > 0.4 and tuple[2] < 0.5: 
                    list_of_relevant_ids[index] = 2
                elif tuple[2] > 0.5: 
                    list_of_relevant_ids[index] = 3

    discounted_cumulative_gain = 0

    list_of_relevant_ids = list(list_of_relevant_ids.values())

    for i, val in enumerate(list_of_relevant_ids):
        discounted_cumulative_gain += val / math.log2(i+2)

    sorted_list_of_relevant_ids = sorted(list_of_relevant_ids, reverse=True)

    idealized_discounted_cumulative_gain = 0

    for i, val in enumerate(sorted_list_of_relevant_ids):
        idealized_discounted_cumulative_gain += val / math.log2(i+2)

    if idealized_discounted_cumulative_gain == 0: return 0

    normalized_discounted_cumulative_gain = discounted_cumulative_gain / idealized_discounted_cumulative_gain

    print(f'\nDiscounted cumulative gain relevance score list: {list_of_relevant_ids}, model = {use_model}')
    print(f'Idealized discounted cumulative gain relevance score list: {sorted_list_of_relevant_ids}, model = {use_model}\n')

    return normalized_discounted_cumulative_gain

#print(recall())

def plot(i, user):

    # PLOTS FOR AP@K and NDCG

    plot_data = pd.DataFrame(columns=['k', 'threshold', 'apatk_score', 'apatk_score_with_model'])
    plot_data = plot_data.fillna(0) # with 0s rather than NaNs
    ndcg_plot_data = pd.DataFrame(columns=['k', 'ndcg_score', 'ndcg_score_with_model'])
    ndcg_plot_data = plot_data.fillna(0) # with 0s rather than NaNs

    for threshold in list_of_thresholds:
        for k in range(1,20+1):
            apatk_score = apatk(i, user.id, k=k, threshold=threshold, type='cos')
            apatk_score_with_model = apatk(i, user.id, k=k, threshold=threshold, type='cos', use_model=True)

            row = pd.Series(data={'k': k, 'threshold': threshold, 'apatk_score': apatk_score, 'apatk_score_with_model': apatk_score_with_model})
            plot_data = plot_data.append(row, ignore_index=True)

    for k in range(1, 20+1):
        ndcg_score = NDCG(i, user.id, k, False)
        ndcg_score_with_model = NDCG(i, user.id, k, True)

        row = pd.Series(data={'k': k, 'ndcg_score': ndcg_score, 'ndcg_score_with_model': ndcg_score_with_model})
        ndcg_plot_data = ndcg_plot_data.append(row, ignore_index=True)

    #display(plot_data)

    point2 = plot_data.loc[ plot_data['threshold'] == list_of_thresholds[0] ]
    point25 = plot_data.loc[ plot_data['threshold'] == list_of_thresholds[1] ]
    point3 = plot_data.loc[ plot_data['threshold'] == list_of_thresholds[2] ]


    # Data

    k_values = point2['k']
    apatk_2 = point2['apatk_score']
    apatk_with_model_2 = point2['apatk_score_with_model']

    apatk_25 = point25['apatk_score']
    apatk_with_model_25 = point25['apatk_score_with_model']

    apatk_3 = point3['apatk_score']
    apatk_with_model_3 = point3['apatk_score_with_model']

    ndcg = ndcg_plot_data['ndcg_score']
    ndcg_with_model = ndcg_plot_data['ndcg_score_with_model']

    # Plot

    plt.style.use('seaborn')

    fig1, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=1)

    p1, = ax1.plot(k_values, apatk_2, label='Without ncf', color='salmon')
    p2, = ax1.plot(k_values, apatk_with_model_2, label='With ncf', color='darkcyan')

    p3, = ax2.plot(k_values, apatk_25, label='Without ncf', color='darkgreen')
    p4, = ax2.plot(k_values, apatk_with_model_25, label='With ncf', color='firebrick')

    p5, = ax3.plot(k_values, apatk_3, label='With ncf', color='indigo')
    p6, = ax3.plot(k_values, apatk_with_model_3, label='Without ncf', color='goldenrod')

    p7, = ax4.plot(k_values, ndcg, label='With ncf', color='darkblue')
    p8, = ax4.plot(k_values, ndcg_with_model, label='Without ncf', color='mediumorchid')


    ax1.legend(handles=[p2, p1], title=f'Threshold: {list_of_thresholds[0]}', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.legend(handles=[p4, p3], title=f'Threshold: {list_of_thresholds[1]}', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax3.legend(handles=[p5, p6], title=f'Threshold: {list_of_thresholds[2]}', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax4.legend(handles=[p7, p8], title='NDCG Scores', bbox_to_anchor=(1.05, 1), loc='upper left')


    ax1.set_title(f'Average Precision At K (AP@K) - User ID: {user.id} - Limit: {limit} - Thresholds: [{list_of_thresholds[0]}, {list_of_thresholds[1]}, {list_of_thresholds[2]}]')
    ax1.set_xlabel('K')
    ax1.set_ylabel('AP@K Score')

    ax2.set_xlabel('K')
    ax2.set_ylabel('AP@K Score')

    ax3.set_xlabel('K')
    ax3.set_ylabel('AP@K Score')

    ax4.set_xlabel('K')
    ax4.set_ylabel('NDCG Score')

    plt.tight_layout()

    plt.show()

# for i, user in enumerate(userlist):
#     plot(i, user)

plot(0, userlist[0])

# TO EXECUTE IN DJANGO SHELL:
# exec(open('Evaluating with multiple users.py').read())