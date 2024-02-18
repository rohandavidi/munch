import requests
from dotenv import load_dotenv
import os
import json
from openai import OpenAI

# Get API Keys
load_dotenv()
google_API_KEY = os.getenv("google_API_KEY")
openai_API_KEY = os.getenv("openai_API_KEY")

# Define the request payload
payload = {
    'textQuery': 'Bucca di Beppo in Bay Area'
}

# Define the request headers
headers_text = {
    'Content-Type': 'application/json',
    'X-Goog-Api-Key': google_API_KEY,
    'X-Goog-FieldMask': 'places.id,places.displayName'
}

# Define the API endpoint
url_text = 'https://places.googleapis.com/v1/places:searchText'

# Make the POST request
response = requests.post(url_text, json=payload, headers=headers_text)

# Check if the request was successful
if response.status_code != 200:
    print('Error:', response.status_code)

# Get IDs from textSearch results
restaurant_dict = response.json()
idList = []
topk = 1
for i in range(topk):
    idList.append(restaurant_dict['places'][i]['id'])

# Get Reviews from details search results
headers_details = {
    'Content-Type': 'application/json',
    'X-Goog-Api-Key': google_API_KEY,
    'X-Goog-FieldMask': 'id,displayName,price_level,rating,reviews'
}

restaurant_data = {}
for id in idList:
    url_cur = 'https://places.googleapis.com/v1/places/' + id
    response = requests.get(url_cur, headers = headers_details)
    if response.status_code != 200:
        print('Error:', response.status_code)
    restaurant_data[id] = response.json()

# Add GPT Summary of reviews for embedding
client = OpenAI(
   api_key=openai_API_KEY
)

for id in restaurant_data:
    cur_reviews = restaurant_data[id]['reviews']
    prompt_string = "Using these reviews write a summary of the restaurant incorporating the majority of the details listed and shared amongst the reviews:"
    for i in range(len(cur_reviews)):
        prompt_string += "\n " + str(i) + ") "
        cur_reviews[i]
    completion = client.chat.completions.create(
      model="gpt-3.5-turbo",
      messages=[
        {"role": "system", "content": "You are a summarizer who takes reviews of restaurants and determines a holistic review using these texts to represent the restaurant as a whole."},
        {"role": "user", "content": prompt_string}
      ]
    )
    review_summary = completion.choices[0].message
    print(review_summary)
    restaurant_data[id]['review_summary'] = review_summary
