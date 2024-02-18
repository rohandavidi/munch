
"""Welcome to Reflex! This file outlines the steps to create a basic app."""

from rxconfig import config
import requests
import reflex as rx
import json
from treehacks_2024 import styles

import numpy as np
import firebase_admin
from firebase_admin import db, credentials
from dotenv import load_dotenv
import os
import pickle
from scipy.spatial.distance import cosine

# Open the pickle file in binary mode for reading
with open('/Users/rohandavidi/Desktop/treehacks_2024/treehacks_2024/embs_dict_correct.pickle', 'rb') as f:
    # Load the pickle file as a dictionary
    embs_dict = pickle.load(f)

cred = credentials.Certificate('/Users/rohandavidi/Desktop/treehacks_2024/treehacks_2024/secret_key.json')
firebase_admin.initialize_app(cred, {
    'databaseURL': "https://treehacks2024-1c2ab-default-rtdb.firebaseio.com"
})
ref = firebase_admin.db.reference("/")


USER_NAME = "rdange"
load_dotenv()
google_API_KEY = os.getenv("google_API_KEY")

embedding_size = 768
#embeddings_normalization = restaurant_embeddings['normalization']

class State(rx.State):
    """The app state."""
    new_rating_data: dict = {}
    finding_data: dict = {}
    match_found: str = "..."
    def finding_submit(self, form_data: dict):
        self.finding_data = form_data
        three_users = len(self.finding_data['user_3']) > 1
        data = ref.get()
        user_a_ratings = data[self.finding_data['user_1']]['ratings']
        user_b_ratings = data[self.finding_data['user_2']]['ratings']
        user_c_ratings = None
        if three_users:
            user_c_ratings = data[self.finding_data['user_3']]['ratings']
        print(user_a_ratings)
        print(user_b_ratings)


        def find_highest_cosine_similarity(restaurant_embeddings, target_embedding, excluded_indices):
            similarities = [1 - cosine(target_embedding, restaurant_embeddings[i]) if i not in excluded_indices else -np.inf for i in range(len(restaurant_embeddings))]
            return np.argmax(similarities)

        def recommend_restaurant(user_top_k_restaurants, restaurant_embeddings, visited_restaurants, t, ind_to_rest, max_iters=10):
            R, d = restaurant_embeddings.shape
            N = len(user_top_k_restaurants)
            k = len(user_top_k_restaurants[0])

            # Convert visited_restaurants to recent_visits for the first t items
            recent_visits = [user_visits[:t] for user_visits in visited_restaurants]

            # Initialize neighboring lists for each user
            neighboring_lists = [[] for _ in range(N)]
            common_restaurants_found = []

            for iteration in range(1, max_iters + 1):
                print("--------")
                print("ITERATION " + str(iteration))
                for user_idx in range(N):
                    # Start with recent visits for the user
                    excluded_indices = set(recent_visits[user_idx])
                    # Add already found neighbors for the user
                    excluded_indices.update(neighboring_lists[user_idx])
                    print("EXCLUDED INDICES: " + str(excluded_indices))

                    for r_idx in user_top_k_restaurants[user_idx]:
                        # Ensure we don't recommend the same restaurant by adding it to excluded indices
                        # excluded_indices.add(r_idx)

                        # Find the restaurant with the highest cosine similarity
                        highest_similarity_idx = find_highest_cosine_similarity(restaurant_embeddings, restaurant_embeddings[r_idx], excluded_indices)
                        if highest_similarity_idx not in neighboring_lists[user_idx]:
                            neighboring_lists[user_idx].append(highest_similarity_idx)
                    print("NEIGHBORING LIST FOR USER " + str(user_idx) + ": " + str(neighboring_lists[user_idx]))

                # Check if there's a common restaurant in all neighboring lists
                common_restaurants = set(neighboring_lists[0])
                for neighbor_list in neighboring_lists[1:]:
                    common_restaurants.intersection_update(neighbor_list)

                # Process common restaurants found in this iteration
                common_restaurants_temp = []
                num_left = 3 - len(common_restaurants_found)
                for r_common in common_restaurants:
                    if r_common not in common_restaurants_found:
                        common_restaurants_temp.append(r_common)
                # hit exactly 3 total
                if len(common_restaurants_temp) == num_left:
                    sorted_temp = sorted(common_restaurants_temp, key=lambda r: sum(visited_restaurants[user_idx].index(r) if r in visited_restaurants[user_idx] else float('inf') for user_idx in range(N)), reverse=True)
                    common_restaurants_found.extend(sorted_temp)
                    # Return the top 3 common restaurants if we have found 3
                    return common_restaurants_found
                # hit >3, so only take 3
                if len(common_restaurants_temp) > num_left:
                    sorted_temp = sorted(common_restaurants_temp, key=lambda r: sum(visited_restaurants[user_idx].index(r) if r in visited_restaurants[user_idx] else float('inf') for user_idx in range(N)), reverse=True)
                    common_restaurants_found.extend(sorted_temp[:num_left])
                    return common_restaurants_found
                # hit <3
                if common_restaurants_temp:
                    # print(list(common_restaurants_temp))
                    # Return the common restaurant maximizing the sum of indices in visited_restaurants
                    sorted_temp = sorted(common_restaurants_temp, key=lambda r: sum(visited_restaurants[user_idx].index(r) if r in visited_restaurants[user_idx] else float('inf') for user_idx in range(N)), reverse=True)
                    # best_common_restaurant = max(common_restaurants, key=lambda r: sum(visited_restaurants[user_idx].index(r) if r in visited_restaurants[user_idx] else float('inf') for user_idx in range(N)))
                    common_restaurants_found.extend(sorted_temp)
                    # DO NOT RETURN, we have not hit 3 yet

                print('Common restaurants found: ')
                print(common_restaurants_found)
                # Continue to the next iteration

            num_left = 3 - len(common_restaurants_found)
            if num_left > 0:
                all_neighbors = sum(neighboring_lists, [])
                remaining_restaurants = [r for r in set(all_neighbors) if r not in common_restaurants_found]
                sorted_remaining = sorted(remaining_restaurants, key=all_neighbors.count, reverse=True)
                toExtend = min(num_left, len(sorted_remaining))
                common_restaurants_found.extend(sorted_remaining[:toExtend])

            toReturn = min(3, len(common_restaurants_found))
            commList = common_restaurants_found[:toReturn]
            return [ind_to_rest[x] for x in commList]

        def json_parser(user_a_ratings, user_b_ratings, user_c_ratings):
          k = 3

          copy_a = user_a_ratings.copy()
          copy_a.reverse()
          visit_history_a = [triple['id'] for triple in copy_a]
          sorted_a = sorted(copy_a, key = lambda x : int(x['rating']), reverse=True)
          topk_a = ([triple['id'] for triple in sorted_a])[:(min(k, len(sorted_a)))]

          copy_b = user_b_ratings.copy()
          copy_b.reverse()
          visit_history_b = [triple['id'] for triple in copy_b]
          sorted_b = sorted(copy_b, key = lambda x : int(x['rating']), reverse=True)
          topk_b = ([triple['id'] for triple in sorted_b])[:(min(k, len(sorted_b)))]

          if user_c_ratings is None:
            visited_restaurants_lst = [visit_history_a, visit_history_b]
            user_top_k_restaurants_lst = [topk_a, topk_b]
          else:
            copy_c = user_c_ratings.copy()
            copy_c.reverse()
            visit_history_c = [triple['id'] for triple in copy_c]
            sorted_c = sorted(copy_c, key = lambda x : int(x['rating']), reverse=True)
            topk_c = ([triple['id'] for triple in sorted_c])[:(min(k, len(sorted_c)))]
            visited_restaurants_lst = [visit_history_a, visit_history_b, visit_history_c]
            user_top_k_restaurants_lst = [topk_a, topk_b, topk_c]

          return user_top_k_restaurants_lst, visited_restaurants_lst


        def use_embeddings(user_top_k_restaurants_lst, visited_restaurants_lst, embs_dict):
          rest_ids = embs_dict.keys()
          rest_to_ind = {rest_ids[i] : i for i in range(len(rest_ids))}
          ind_to_rest = {i : rest_ids[i] for i in range(len(rest_ids))}

          restaurant_embeddings = np.array([embs_dict[ind_to_rest[i]] for i in range(len(rest_ids))])
          user_top_k_restaurants = np.array([[rest_to_ind[x] for x in sublist] for sublist in user_top_k_restaurants_lst])
          visited_restaurants = [[rest_to_ind[x] for x in sublist] for sublist in visited_restaurants_lst]
          t = 3

          return rest_to_ind, ind_to_rest, restaurant_embeddings, user_top_k_restaurants, visited_restaurants, t

        user_top_k_restaurants_lst, visited_restaurants_lst = json_parser(user_a_ratings, user_b_ratings, user_c_ratings)
        rest_to_ind, ind_to_rest, restaurant_embeddings, user_top_k_restaurants, visited_restaurants, t = use_embeddings(user_top_k_restaurants_lst, visited_restaurants_lst, embs_dict)

        recommendations = recommend_restaurant(user_top_k_restaurants, restaurant_embeddings, visited_restaurants, t, ind_to_rest)# [""]

        cur_match_found = recommendations[0]
        url_2 = 'https://places.googleapis.com/v1/places/' + cur_match_found

        # Make the POST request
        #response = requests.post(url_1, json=payload, headers=headers)

        headers_2 = {
            'Content-Type': 'application/json',
            'X-Goog-Api-Key': google_API_KEY,
            'X-Goog-FieldMask': 'id,displayName,price_level'
        }

        response = requests.get(url_2, headers = headers_2)
        self.match_found = response.json()['displayName']['text']


    def rating_submit(self, form_data: dict):
        self.new_rating_data = form_data
        # Define the request payload
        payload = {
            'textQuery': self.new_rating_data["restaurant"] + " in " + self.new_rating_data["location"]
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
        print(self.new_rating_data["restaurant"] + " Restaurant in " + self.new_rating_data["location"])
        print(response.json())
        new_rating_id = response.json()['places'][0]['id']
        rating_name = response.json()['places'][0]['displayName']['text']
        rating_dict = {"id": new_rating_id, "name": rating_name, "rating": self.new_rating_data['rating']}
        #if new_rating_id not in restaurant_embeddings:


        ref = db.reference("/")
        data = ref.get()
        if USER_NAME in data:
            print("FOUND")
            data[USER_NAME]['ratings'].append(rating_dict)
            ref.update(data)
        else:
            data[USER_NAME] = {"ratings": [rating_dict]}
            print(data)
            ref.update(data)

def found_munch():
    return rx.chakra.popover(
    rx.chakra.popover_trigger(
        rx.chakra.button("Get Munch Match", color="#9514ff")
    ),
    rx.chakra.popover_content(
        rx.chakra.popover_header("Here is your Munch Match!"),
        rx.chakra.popover_body(
            rx.chakra.text(State.match_found.to_string())
        ),
        rx.chakra.popover_close_button(),
    ),
)

def rating_form():
    return rx.chakra.vstack(
        rx.chakra.heading("Add Your New Rating"),
        rx.chakra.form(
            rx.chakra.vstack(
                rx.chakra.input(
                    placeholder="Restaurant Name",
                    name="restaurant",
                ),
                rx.chakra.input(
                    placeholder="City/Location",
                    name="location",
                ),
                rx.chakra.input(
                    placeholder="Rating Out of 5",
                    name="rating",
                ),
                rx.chakra.button("Submit", type_="submit"),
            ),
            on_submit=State.rating_submit,
            reset_on_submit=True,
        ),
        rx.chakra.divider(),
        rx.chakra.heading("Results"),
        rx.chakra.text(State.new_rating_data.to_string()),
    )

def finding_form():
    return rx.chakra.vstack(
        rx.chakra.heading("Time to Munch!"),
        rx.chakra.form(
            rx.chakra.vstack(
                rx.chakra.input(
                    placeholder="Username 1",
                    name="user_1",
                ),
                rx.chakra.input(
                    placeholder="Username 2",
                    name="user_2",
                ),
                rx.chakra.input(
                    placeholder="Username 3 (Optional)",
                    name="user_3",
                ),
                rx.chakra.button("Submit", type_="submit"),
            ),
            on_submit=State.finding_submit,
            reset_on_submit=True,
        ),
        rx.chakra.divider(),
        rx.chakra.heading("Munching the Numbers"),
        rx.chakra.text(State.finding_data.to_string()),
    )


def navbar():
    return rx.chakra.box(
        rx.chakra.hstack(
            rx.chakra.hstack(
                rx.chakra.link(
                    rx.chakra.box(
                        rx.chakra.image(src="favicon.ico", width=30, height="auto"),
                        p="1",
                        border_radius="6",
                        bg="#F0F0F0",
                        mr="2",
                    ),
                    href="/",
                ),
                rx.chakra.breadcrumb(
                    rx.chakra.breadcrumb_item(
                        rx.chakra.heading("Munch", size="sm", color="white"),
                    ),
                    # rx.chakra.breadcrumb_item(
                    #     rx.chakra.text(State.current_chat, size="sm", font_weight="normal"),
                    # ),
                ),
            ),
            rx.chakra.hstack(
                rx.chakra.link(rx.chakra.button("Find Food!"),
                    bg="white",
                    px="4",
                    py="2",
                    h="auto",
                    href="/find", button=True),
                rx.chakra.link(rx.chakra.button("+ New Rating"),
                    bg="white",
                    px="4",
                    py="2",
                    h="auto",
                    href="/rate", button=True),
                rx.chakra.menu(
                    rx.chakra.menu_button(
                        rx.chakra.avatar(name="User", size="md", bg="white"),
                        rx.chakra.box(),
                    ),
                    rx.chakra.menu_list(
                        rx.chakra.menu_item("Profile"),
                    ),
                ),
                spacing="8",
            ),
            justify="space-between",
        ),
        bg= "linear-gradient(to bottom, #9514ff, #b86af7)",
        backdrop_filter="auto",
        backdrop_blur="lg",
        p="4",
        border_bottom=f"1px solid white",
        position="sticky",
        top="0",
        width="100%",
        z_index="5",
    )

@rx.page(route='/rate', title="Add Rating")
def rate():
    return rx.chakra.vstack(navbar(), rating_form())

@rx.page(route='/find', title="Find Food")
def find():
    return rx.chakra.vstack(navbar(), finding_form(), found_munch())

@rx.page(route='/', title="Profile")
def profile():
    return rx.chakra.vstack(navbar())


app = rx.App(state=State)
app.add_page(rate)
app.add_page(find)
app.add_page(profile)
