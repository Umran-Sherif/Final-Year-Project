from pytrends.request import TrendReq
from IPython.display import display 
import pymannkendall as mk
import pandas as pd
import joblib
import time

start_time = time.time()

def run():
	print('PBR: Initializing')

	pytrends = TrendReq(hl='en-US')
	all_keywords = ['Chicken', 'Beef', 'Fish', 'Soup', 'Potato', 'Salmon', 'Spicy', 'Kebab', 'Turkey', 'Lamb', 'Pork', 'Taco', 'Pasta', 'Vegeterian', 'Shrimp', 'Sea Bass', 'Roast']
	keywords = []
	cat = 0
	geo = ''
	gprop = ''
	timeframe = ['today 5-y', 'now 1-d', 'now 2-d', 'now 7-d']
	highest_interest = []

	for kw in all_keywords:
		keywords.append(kw)

		try:
			print('PBR: Building payload for ' + kw)

			pytrends.build_payload(keywords, cat, timeframe[3], geo, gprop)

			print('PBR: Computing interest over time')
			data = pytrends.interest_over_time()
			data.drop(['isPartial'], axis=1, inplace=True)

			print('PBR: Applying Mann Kendall Seasonal Test on ' + kw)
			_,_,_,_,s,_,_,_,_ = mk.seasonal_test(data)
			highest_interest.append((kw,s))
		
		except Exception as e:
			print('Problem occurred when building payload for ' + kw)

		keywords.pop()

	sorted_values = sorted(highest_interest, key=lambda tup: tup[1], reverse=True)
	trending_ingredient = sorted_values[0][0]
	trending_ingredient2 = sorted_values[1][0]

	print('PBR: The cooking terms that have been trending over the past few days is ' + trending_ingredient + ' and ' + trending_ingredient2)

	#trending_ingredients = ['Chicken', 'Fish', 'Pasta'] # Test
	all_possible_candidates = joblib.load('all_possible_candidates_20.obj')
	candidate_recipes = []  # A list of recipes that contain the trending ingredient1
	candidate_recipes2 = []  # A list of recipes that contain the trending ingredient2

	for row in all_possible_candidates:
		try:
			if trending_ingredient.lower() in row[0]:
				candidate_recipes.append(row)
			elif trending_ingredient2.lower() in row[0]:
				candidate_recipes2.append(row)
		except:
			pass

	candidate_recipes_df = pd.DataFrame(candidate_recipes,columns=['Name', 'Recipe_ID', 'Number of Ratings', 'Mean Score', 'Popularity Score'])
	candidate_recipes_df2 = pd.DataFrame(candidate_recipes2,columns=['Name', 'Recipe_ID', 'Number of Ratings', 'Mean Score', 'Popularity Score'])
	candidate_recipes_df.dropna(inplace=True)
	candidate_recipes_df2.dropna(inplace=True)
	candidate_recipes_df.drop_duplicates(inplace=True)
	candidate_recipes_df2.drop_duplicates(inplace=True)
	sorted_candidate_recipes = candidate_recipes_df.sort_values(by='Popularity Score', ascending=False)
	sorted_candidate_recipes2 = candidate_recipes_df2.sort_values(by='Popularity Score', ascending=False)

	# Top 6 highest scoring recipes
	listOfRecipeIDs = sorted_candidate_recipes.head(3)['Recipe_ID'].to_list()    # From highest to lowest popularity score
	listOfRecipeIDs2 = sorted_candidate_recipes2.head(3)['Recipe_ID'].to_list()    # From highest to lowest popularity score

	print('The top 6 highest scoring recipes')
	display(sorted_candidate_recipes.head(3))
	display(sorted_candidate_recipes2.head(3))

	finalListOfRecipeIDs = listOfRecipeIDs + listOfRecipeIDs2
	joblib.dump(finalListOfRecipeIDs, 'listOfTrendingRecipeIDs.obj')
	print('Successfly created Trending ID list')

	end = time.time()
	print(f"Runtime: {end - start_time} seconds")