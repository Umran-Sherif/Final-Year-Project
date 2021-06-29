import requests
from bs4 import BeautifulSoup
import os
from recipe.models import Recipe

import time
import concurrent.futures

t1 = time.perf_counter()


os.chdir(os.getcwd() + '\\media\\recipe_pics')

all_recipes = Recipe.objects.all()


# TO EXECUTE IN DJANGO SHELL:
# exec(open('webscrape_recipe_images.py').read())


def download_image(r):
	name = r.recipe_name.lower().replace(' ', '-')
	urlname = ( name + ' ' + str(r.recipe_id) ).replace(' ', '-')

	# urlname = 'caras-sweet-and-sour-crock-pot-chicken-109750'	# test

	url = 'https://www.food.com/recipe/' + urlname

	r = requests.get(url)

	soup = BeautifulSoup(r.text, 'html.parser')

	images = soup.find_all('link')

	for image in images:
		if image['rel'][0] == 'image_src':
			link = image['href']  # get recipe image link

	with open(name + '.jpg', 'wb') as f:
	    im = requests.get(link)
	    f.write(im.content)
	    print('Writing: ', name.replace('-', ' '))
	    print('Done!')
    


with concurrent.futures.ThreadPoolExecutor() as executor:
    executor.map(download_image, all_recipes)


t2 = time.perf_counter()

print(f'Finished in {t2-t1} seconds')