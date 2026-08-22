from geopy.geocoders import Nominatim

geolocator = Nominatim(user_agent="water-segmentation")

def get_location(place_name):
    location = geolocator.geocode(place_name)

    if not location:
        return None

    return {
        "name": place_name,
        "lat": location.latitude,
        "lon": location.longitude
    }