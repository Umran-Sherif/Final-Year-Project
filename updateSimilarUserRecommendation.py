import tensorflow.compat.v1 as tf
from libreco.data import random_split, DatasetPure, split_by_ratio, DataInfo
from libreco.algorithms import NCF  # pure data, algorithm SVD++ and NCF
from libreco.evaluation import evaluate
from recipe.models import Recipe, Rating
from django.contrib.auth.models import User
from users.models import Profile
from recipe.models import Recipe
import pandas as pd
import joblib
from pandas.util.testing import assert_frame_equal

def ncf_recommendation():
	path = 'Collaborative Filtering Files'

	# Import new user data to retrain the model
	print('Similar User Recommendation Model: Importing new user data')

	new_data = pd.DataFrame(None, columns=['user', 'item', 'label'])

	user = User.objects.all()[0]	# Assume user is 'spagb'
	rating_list = user.rating_history.all()

	if len(rating_list) < 3: 
		print('Similar User Recommendation Model: No changes made. Rate at least 3 recipes.')
		return 

	for review in rating_list:
		recipeid = review.recipe.recipe_id
		label = review.rating
		new_data = new_data.append({'user': user.id, 'item': recipeid, 'label': label}, ignore_index=True)

	print('Similar User Recommendation Model: Checking old user data')

	old_data = joblib.load('old_user_rating_history.obj')

	if new_data.equals(old_data):
		print('Similar User Recommendation Model: No changes were made. Rating scores have not changed.')
		return

	train, test = split_by_ratio(new_data, test_size=0.2)
	for i in range(3):
	    test = test.append(train.iloc[i])

	# Resetting graph
	print('Similar User Recommendation Model: Resetting graph')
	tf.reset_default_graph()

	# Loading Data Info
	print('Similar User Recommendation Model: Loading data info')
	data_info = DataInfo.load(path)

	train_data, data_info = DatasetPure.build_trainset(train, revolution=True, data_info=data_info, merge_behavior=True)
	test_data = DatasetPure.build_testset(test, revolution=True, data_info=data_info)

	print('Similar User Recommendation Model: New data_info: ', data_info)

	# Rebuilding with new dimensions
	print('Similar User Recommendation Model: Rebuilding ncf with new dimensions')

	model = NCF('rating', data_info, embed_size=16, n_epochs=3, lr=0.001, lr_decay=False, reg=None, 
	          batch_size=256, num_neg=1, use_bn=True, dropout_rate=None, hidden_units="128,64,32",
	          tf_sess_config=None)

	model.rebuild_graph(path=path, model_name='ncf', full_assign=True)

	model.fit(train_data, verbose=2, shuffle=True, eval_data=test_data,
	          metrics=['rmse', 'mae', 'r2'])

	# Predictions
	print('Similar User Recommendation Model: Checking model for recommendations')

	list_of_preds = []
	# list_of_itemids = [52417, 107559, 191318, 515167, 54983]  #test

	for pair in model.recommend_user(user=user.id, n_rec=200):
		list_of_preds.append(pair[0])

	print("Similar User Recommendation Model: Deleting " + user.username + "'s old similar list")
	for sim in user.profile.similar.all():
		user.profile.similar.remove(sim)
	all_recipes = Recipe.objects.all()
	counter = 0

	for recipeid in list_of_preds:
		try:
			if counter == 5:
				break
			recip = all_recipes.filter(recipe_id=recipeid)[0]
			user.profile.similar.add(recip)
			counter += 1
		except:
			pass

	print('Similar User Recommendation Model: ' + user.username + "'s similar user recommendation list, has been updated")

	print("Similar User Recommendation Model: Saving user's rating history data")
	joblib.dump(new_data, 'old_user_rating_history.obj')
	print('Similar User Recommendation Model: Done!')
	return

# TO EXECUTE IN DJANGO SHELL:
# exec(open('updateSimilarUserRecommendation.py').read())