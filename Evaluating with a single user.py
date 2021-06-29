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
import joblib, time, warnings, os, sys, math

warnings.filterwarnings("ignore")
start_time = time.time()

# USER = 71-2 # Janet with ID 71
# # USER = 4659 # Janet with ID 4660
# user = User.objects.all()[USER]  # Assume Spagb to be the user

# USER = 6
USER = 271
user = User.objects.all().filter(id=USER)[0]
recipe20IdList = joblib.load('allRecipe_20IDs.obj')
limit = 16
list_of_thresholds = [0.3, 0.4, 0.5]

# Removes erronous error detection when calling model.predict

class HiddenPrints:
    def __enter__(self):
     self._original_stdout = sys.stdout
     sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
     sys.stdout.close()
     sys.stdout = self._original_stdout

# Recommender method

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


# Beginning

print('CBR: Loading the data')

tfv = TfidfVectorizer(strip_accents='unicode', analyzer='word', stop_words = 'english', token_pattern=r'(?u)\b\w\w\w+\b')

df_recipes = pd.read_csv('dataset\\recipes_20 split\\disk1.csv', engine='python', encoding='utf-8', error_bad_lines=False)
df_recipes.columns = ['idx', 'Name', 'Recipe_ID', 'minutes', 'submitted', 'tags', 'nutrition','steps', 'description', 'ingredients']

#df = df_recipes.assign(info = df_recipes.steps.astype(str) + ' ' + df_recipes.ingredients.astype(str) + ' ' + df_recipes.description.astype(str) + ' ' + df_recipes.nutrition.astype(str))
df = df_recipes.assign(info = df_recipes.steps.astype(str) + ' ' + df_recipes.ingredients.astype(str))
df.drop(['idx', 'Name', 'tags', 'steps', 'ingredients', 'description', 'Recipe_ID', 'minutes', 'submitted', 'nutrition'], axis=1, inplace=True)

tfv_matrix = tfv.fit_transform(df['info'])

num = 0

print('CBR: Fetching user ratings')
id_ratings = {}     # Key: Recipe ID, Value: Rating
for rating in user.rating_history.all().order_by('-date_added'):
    id_ratings[rating.recipe.recipe_id] = rating.rating
    num += 1
    if num>limit: break

#id_ratings = {15116: 5.0, 148135: 2.0, 14945: 4.0, 187401: 2.0, 223883: 3.0}    #Test data

# Normalize the ratings, taking to account the rating tendencies for a given user
# Recipes that have ratings that are below the user mean, will not be zeroed - not considered.

print('CBR: Normalizing ratings')

raw_ratings = list(id_ratings.values())    # Create a list of all the rating values for aggregates

mean = np.mean(raw_ratings)

normalized_ratings = {}

for item in id_ratings.items():
    normalized_ratings[item[0]] = item[1]-mean    # item[0] is key, item[1] is value
    # num += 1
    # if num>limit: break

weights = {}
for item in normalized_ratings.items():
    if item[1]>0:
        weights[item[0]] = 1
        # num+=1
        # if num+1>limit: break

for index, row in df_recipes.iterrows():
    id = row['Recipe_ID']
    if id in weights:   # If the recipe ID is in the normalized ratings (weights) key list
        continue
    else:
        weights[id] = 0


# Convert all recipe IDs to index values

index_ratings = {}
for i, value in enumerate(weights.values()):
    index_ratings[i] = value
 

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
final_tfv = final_tfv_matrix.astype(np.float16)



print('CBR: Creating cosine similarity matrix')
cosine_sim = cosine_similarity(final_tfv, final_tfv)
#print('CBR: Creating linear kernel matrix')
#linear_ker = linear_kernel(final_tfv, final_tfv)
#linear_ker = sigmoid_kernel(final_tfv, final_tfv)



# Dummy data - so that, User Profile is now technically a recipe
d = {'Recipe_ID': 200000000000000, 'Name': 'User Profile', 'minutes': 0, 'contributor_id': 0, 'submitted': '', 'steps': '', 'description': '', 'ingredients': ''}
ser = pd.Series(data=d)
# Appending this User Profile recipe to the recipe matrix
df_recipes = df_recipes.append(ser, ignore_index=True)

recipes = df_recipes.drop(columns=['Recipe_ID', 'minutes', 'contributor_id', 'submitted', 'nutrition'])

indices = pd.Series(recipes.index, index=recipes['Name']).drop_duplicates()




# Import new user data to retrain the model
path = 'Collaborative Filtering Files'

print('CBR: Importing new user data')

new_data = pd.DataFrame(None, columns=['user', 'item', 'label'])

# userid = USER+2

user = User.objects.all().filter(id=USER)[0]
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




# Recommendation part
print('CBR: Feeding in user profile')
recommendation_cos_sim = give_rec(USER, 'User Profile', cosine_sim, indices, recipes, k=10, use_model=False)    # User with id 1: spagb (admin)
#recommendation_lin_ker = give_rec(1, 'User Profile', linear_ker, indices, recipes, k=10, use_model=True)    # User with id 1: spagb (admin)



# Evaluation
print('CBR: Computing Average Precision @ K score')

arr = final_tfv_matrix.toarray()

def apatk(k=20, threshold=0.2, type='cos', use_model=False):    # Defaults to AP@20
    sum_of_precisions = 0

    for m in range(1, k+1):
        if type == 'cos':
            recommendation = give_rec(USER, 'User Profile', cosine_sim, indices, recipes, k=m, use_model=use_model)
        elif type == 'lin':
            #recommendation = give_rec(1, 'User Profile', linear_ker, indices, recipes, k=m, use_model=use_model)
            return
        else:
            print('Invalid recommendation type')
            return

        precision_at_m = precision_report(recommendation, threshold)
        
        #print(f'Precision at K={m}: {precision_at_m}')

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
            #print( f'{(rec_index_list, feature_list_key, percentage)}' )
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

def recall(relevance_threshold=0.2, use_model=False, thresh=20):
    recommendation = give_rec(USER, 'User Profile', cosine_sim, indices, recipes, k=6, use_model=use_model)
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


def NDCG(k=6, use_model=False):
    recommendation = give_rec(USER, 'User Profile', cosine_sim, indices, recipes, k=k, use_model=use_model)
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
                if tuple[2] > list_of_thresholds[0] and tuple[2] < list_of_thresholds[1]: 
                    list_of_relevant_ids[index] = 1
                elif tuple[2] > list_of_thresholds[1] and tuple[2] < list_of_thresholds[2]: 
                    list_of_relevant_ids[index] = 2
                elif tuple[2] > list_of_thresholds[2]: 
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


df_recipes2 = pd.read_csv('dataset\\recipes_20 split\\recipes_20.csv', engine='python', encoding='utf-8', error_bad_lines=False)
df_recipes2.columns = ['idx', 'Name', 'Recipe_ID', 'minutes', 'submitted', 'tags', 'nutrition','steps', 'description', 'ingredients']

df_whole = df_recipes2.assign(info = df_recipes2.steps.astype(str) + ' ' + df_recipes2.ingredients.astype(str))
df_whole.drop(['idx', 'Name', 'tags', 'steps', 'ingredients', 'description', 'Recipe_ID', 'minutes', 'submitted', 'nutrition'], axis=1, inplace=True)

tfv_matrix_whole = tfv.fit_transform(df_whole['info'])


def checkSimilarityForUser(userid):     # Assuming userid is an index for user objects, which is actual id - 1.
    user = User.objects.all()[userid]

    listOfUserRecipes = []

    for rating in user.rating_history.all():
        listOfUserRecipes.append(rating.recipe_id)

    user_recipes = df_recipes2.loc[df_recipes2['Recipe_ID'].isin(listOfUserRecipes)]
    num_ratings = user_recipes.shape[0]

    df_condensed_user_recipes = user_recipes.assign(info = user_recipes.steps.astype(str) + ' ' + user_recipes.ingredients.astype(str))
    df_condensed_user_recipes.drop(['idx', 'Name', 'tags', 'steps', 'ingredients', 'description', 'Recipe_ID', 'minutes', 'submitted', 'nutrition'], axis=1, inplace=True)

    tfv_matrix = tfv.fit_transform(df_condensed_user_recipes['info'])

    user_cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)

    total_mean_similarity = 0

    for i in range(num_ratings):
        el = list(enumerate(user_cosine_sim[i]))

        ith_recipe_name = user_recipes['Name'].iloc[i]

        highest_sim_score = 0
        highest_sim_recipe_name = ''
        j = 0
        for pair in el:
            if pair[1] > highest_sim_score and pair[1] < 0.99:
                highest_sim_score = pair[1]
                highest_sim_recipe_name = user_recipes['Name'].iloc[pair[0]]    # Retreive the recipe name for the jth element in el
                j = pair[0]

        print(f'{ith_recipe_name}({i}) is most similar to {highest_sim_recipe_name}({j})')
        print('The similarity score between them is {:.2f}%'.format(100*(highest_sim_score)))
        
        sum_mean = 0
        for pair in el:
            sum_mean += pair[1]

        sum_mean = 100*(sum_mean/num_ratings)

        print('The mean similarity for recipe index {} is {:.2f}%\n'.format(i, sum_mean))

        total_mean_similarity += sum_mean

    total_mean_similarity = total_mean_similarity/num_ratings

    print(f'The total mean similarity for user {userid} is {total_mean_similarity:.2f}%')


def listTotalMeanSim(userid):     # Assuming userid is an index for user objects, which is actual id - 1.
    user = User.objects.all()[userid]

    listOfUserRecipes = []

    for rating in user.rating_history.all():
        listOfUserRecipes.append(rating.recipe_id)

    user_recipes = df_recipes2.loc[df_recipes2['Recipe_ID'].isin(listOfUserRecipes)]
    num_ratings = user_recipes.shape[0]


    df_condensed_user_recipes = user_recipes.assign(info = user_recipes.steps.astype(str) + ' ' + user_recipes.ingredients.astype(str))
    df_condensed_user_recipes.drop(['idx', 'Name', 'tags', 'steps', 'ingredients', 'description', 'Recipe_ID', 'minutes', 'submitted', 'nutrition'], axis=1, inplace=True)

    tfv_matrix = tfv.fit_transform(df_condensed_user_recipes['info'])

    user_cosine_sim = cosine_similarity(tfv_matrix, tfv_matrix)

    total_mean_similarity = 0

    for i in range(num_ratings):
        el = list(enumerate(user_cosine_sim[i]))

        ith_recipe_name = user_recipes['Name'].iloc[i]
        
        sum_mean = 0
        for pair in el:
            sum_mean += pair[1]

        sum_mean = 100*(sum_mean/num_ratings)

        total_mean_similarity += sum_mean

    total_mean_similarity = total_mean_similarity/num_ratings

    print(f'The total mean similarity for user {userid} is {total_mean_similarity:.2f}%. This user rated {num_ratings} recipes')

# Find common features between recipes in the entire dataset

def findCommonFeaturesWhole(id1, id2):  
    list_of_tups, featuresid1, featuresid2 = [], [], []

    new_coo_matrix = coo_matrix( tfv_matrix_whole )

    i1 = df_recipes2[df_recipes2['Recipe_ID'] == id1].index[0] 
    i2 = df_recipes2[df_recipes2['Recipe_ID'] == id2].index[0] 

    for i,j,v in zip(new_coo_matrix.row, new_coo_matrix.col, new_coo_matrix.data):
        if i == i1 or i == i2:
            list_of_tups.append((i,j,v))

    vocabulary = tfv.vocabulary_

    for triple in list_of_tups:
        if triple[0] == i1:
            featuresid1.append(list(vocabulary.keys())[list(vocabulary.values()).index(triple[1])])
        if triple[0] == i2:
            featuresid2.append(list(vocabulary.keys())[list(vocabulary.values()).index(triple[1])])

    featuresid1.sort()    
    featuresid2.sort()    
    common_features = set(featuresid1).intersection(featuresid2)

    print(
    f'Number of features in Recipe 1: {len(featuresid1)}\nNumber of features in Recipe 2: {len(featuresid2)}\nNumber of features in common: {len(common_features)}'
    )

    return featuresid1, featuresid2, common_features


# Find common features within user's rated recipes
def findCommonFeatures(userid, id1, id2):
    user = User.objects.all()[userid]

    listOfUserRecipes = []

    for rating in user.rating_history.all():
        listOfUserRecipes.append(rating.recipe_id)

    user_recipes = df_recipes2.loc[df_recipes2['Recipe_ID'].isin(listOfUserRecipes)]
    num_ratings = user_recipes.shape[0]


    #df_condensed_user_recipes = user_recipes.assign(info = user_recipes.steps.astype(str) + ' ' + user_recipes.ingredients.astype(str) + ' ' + user_recipes.description.astype(str))
    df_condensed_user_recipes = user_recipes.assign(info = user_recipes.steps.astype(str) + ' ' + user_recipes.ingredients.astype(str))
    df_condensed_user_recipes.drop(['idx', 'Name', 'tags', 'steps', 'ingredients', 'description', 'Recipe_ID', 'minutes', 'submitted', 'nutrition'], axis=1, inplace=True)

    tfv_matrix = tfv.fit_transform(df_condensed_user_recipes['info'])

    
    list_of_tups, featuresid1, featuresid2 = [], [], []

    new_coo_matrix = coo_matrix( tfv_matrix )

    for i,j,v in zip(new_coo_matrix.row, new_coo_matrix.col, new_coo_matrix.data):
        list_of_tups.append((i,j,v))

    vocabulary = tfv.vocabulary_

    for triple in list_of_tups:
        if triple[0] == id1:
            featuresid1.append(list(vocabulary.keys())[list(vocabulary.values()).index(triple[1])])
        if triple[0] == id2:
            featuresid2.append(list(vocabulary.keys())[list(vocabulary.values()).index(triple[1])])

    featuresid1.sort()
    featuresid2.sort()    
    common_features = set(featuresid1).intersection(featuresid2)

    print(
    f'Number of features in Recipe 1: {len(featuresid1)}\nNumber of features in Recipe 2: {len(featuresid2)}\nNumber of features in common: {len(common_features)}'
    )

    return featuresid1, featuresid2, common_features


features1, features2, common_features = findCommonFeatures(userid=70, id1=112, id2=78)


# Plots

plot_data = pd.DataFrame(columns=['k', 'threshold', 'apatk_score', 'apatk_score_with_model'])
plot_data = plot_data.fillna(0) # with 0s rather than NaNs

for threshold in list_of_thresholds:
    for k in range(1,20+1):
        apatk_score = apatk(k=k, threshold=threshold, type='cos')
        apatk_score_with_model = apatk(k=k, threshold=threshold, type='cos', use_model=True)

        row = pd.Series(data={'k': k, 'threshold': threshold, 'apatk_score': apatk_score, 'apatk_score_with_model': apatk_score_with_model})
        plot_data = plot_data.append(row, ignore_index=True)

#display(plot_data)

point2 = plot_data.loc[ plot_data['threshold'] == list_of_thresholds[0] ]
point25 = plot_data.loc[ plot_data['threshold'] == list_of_thresholds[1] ]
point3 = plot_data.loc[ plot_data['threshold'] == list_of_thresholds[2] ]



import matplotlib.pyplot as plt

# Data

k_values_2 = point2['k']
apatk_2 = point2['apatk_score']
apatk_with_model_2 = point2['apatk_score_with_model']

k_values_25 = point25['k']
apatk_25 = point25['apatk_score']
apatk_with_model_25 = point25['apatk_score_with_model']

k_values_3 = point3['k']
apatk_3 = point3['apatk_score']
apatk_with_model_3 = point3['apatk_score_with_model']



ndcg_plot_data = pd.DataFrame(columns=['k', 'ndcg_score', 'ndcg_score_with_model'])

for k in range(1, 20+1):
    ndcg_score = NDCG(k, False)
    ndcg_score_with_model = NDCG(k, True)

    row = pd.Series(data={'k': k, 'ndcg_score': ndcg_score, 'ndcg_score_with_model': ndcg_score_with_model})
    ndcg_plot_data = ndcg_plot_data.append(row, ignore_index=True)

ndcg = ndcg_plot_data['ndcg_score']
ndcg_with_model = ndcg_plot_data['ndcg_score_with_model']


# Plot

plt.style.use('seaborn')

fig1, (ax1,ax2,ax3,ax4) = plt.subplots(nrows=4, ncols=1)

p1, = ax1.plot(k_values_2, apatk_2, label='Without ncf', color='salmon')
p2, = ax1.plot(k_values_2, apatk_with_model_2, label='With ncf', color='darkcyan')

p3, = ax2.plot(k_values_25, apatk_25, label='Without ncf', color='darkgreen')
p4, = ax2.plot(k_values_25, apatk_with_model_25, label='With ncf', color='firebrick')

p5, = ax3.plot(k_values_3, apatk_3, label='Without ncf', color='goldenrod')
p6, = ax3.plot(k_values_3, apatk_with_model_3, label='With ncf', color='indigo')

p7, = ax4.plot(k_values_2, ndcg, label='Without ncf', color='mediumorchid')
p8, = ax4.plot(k_values_2, ndcg_with_model, label='With ncf', color='darkblue')


ax1.legend(handles=[p2, p1], title='Threshold = ' + str(list_of_thresholds[0]), bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.legend(handles=[p4, p3], title='Threshold = ' + str(list_of_thresholds[1]), bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.legend(handles=[p6, p5], title='Threshold = ' + str(list_of_thresholds[2]), bbox_to_anchor=(1.05, 1), loc='upper left')
ax4.legend(handles=[p8, p7], title='NDCG Scores', bbox_to_anchor=(1.05, 1), loc='upper left')

# ax1.set_title(f'Average Precision At K (AP@K) - User ID: {USER}')
ax1.set_title(f'Average Precision At K - User ID: {USER} - Limit: {limit} - Thresholds: [{list_of_thresholds[0]}, {list_of_thresholds[1]}, {list_of_thresholds[2]}]')
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


# TO EXECUTE IN DJANGO SHELL:
# exec(open('Evaluating with a single user.py').read())