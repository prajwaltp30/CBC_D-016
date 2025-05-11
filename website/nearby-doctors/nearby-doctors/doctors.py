from flask import Flask, render_template, request, jsonify
import requests
import os
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

@app.route('/')
def index():
    return render_template('index.html', google_api_key=GOOGLE_API_KEY)


@app.route('/nearby-doctors', methods=['POST'])
def get_nearby_doctors():
    data = request.get_json()
    # user_lat = 12.879376
    # user_lng = 77.544295
    user_lat = data.get("lat")
    user_lng = data.get("lng")


    places_url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    places_params = {
        "location": f"{user_lat},{user_lng}",
        "radius": 3000,  # 10km
        "type": "doctor|hospital",
        "key": GOOGLE_API_KEY
    }

    places_response = requests.get(places_url, params=places_params)
    places_data = places_response.json()

    results = places_data.get("results", [])
    if not results:
        return jsonify([])

    destinations = [f"{place['geometry']['location']['lat']},{place['geometry']['location']['lng']}" for place in results]

    # Distance Matrix
    matrix_url = "https://maps.googleapis.com/maps/api/distancematrix/json"
    matrix_params = {
        "origins": f"{user_lat},{user_lng}",
        "destinations": "|".join(destinations),
        "key": GOOGLE_API_KEY
    }

    matrix_response = requests.get(matrix_url, params=matrix_params)
    distance_data = matrix_response.json()

    # Step 4: Attach distances and sort
    for i, place in enumerate(results):
        try:
            place["distance_text"] = distance_data["rows"][0]["elements"][i]["distance"]["text"]
            place["distance_value"] = distance_data["rows"][0]["elements"][i]["distance"]["value"]
        except KeyError:
            place["distance_text"] = "Unknown"
            place["distance_value"] = float('inf')

    sorted_places = sorted(results, key=lambda x: x["distance_value"])

    doctors = []
    for place in sorted_places:
        loc = place["geometry"]["location"]

        # 🆕 Call Place Details API here
        place_details_url = "https://maps.googleapis.com/maps/api/place/details/json"
        place_details_params = {
            "place_id": place["place_id"],
            "fields": "name,formatted_phone_number,international_phone_number",
            "key": GOOGLE_API_KEY
        }
        details_response = requests.get(place_details_url, params=place_details_params)
        details_data = details_response.json()

        phone_number = None
        if details_data.get("result"):
            phone_number = details_data["result"].get("international_phone_number")

        doctors.append({
            "name": place["name"],
            "lat": loc["lat"],
            "lng": loc["lng"],
            "address": place.get("vicinity", ""),
            "distance": place.get("distance_text", "Unknown"),
            "rating": place.get("rating", "N/A"),
            "phone": phone_number
        })

    return jsonify(doctors)

if __name__ == '__main__':
    app.run(debug=True, port=8080)

