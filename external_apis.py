import requests

# to make sure the code works can I not just ask each code
# set to either print the dataset it extracted or the text to speech that it picked up?
def get_user_location(ipinfo_api_key):
    url = f"https://ipinfo.io?token={ipinfo_api_key}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        # The location is returned as a string 'latitude,longitude'
        coordinates = data['loc'].split(',')
        return {
            'latitude': float(coordinates[0]),
            'longitude': float(coordinates[1])
        }
    else:
        print("Error fetching GPS coordinates:", response.status_code)
        return None


def get_weather_data(latitude, longitude, api_key):
    url = f"https://api.weatherapi.com/v1/current.json?key={api_key}&q={latitude},{longitude}"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        temperature_c = data['current']['temp_c']  # Temperature (degrees C)
        pressure_mb = data['current']['pressure_mb']  # Air pressure (hPa)
        humidity = data['current']['humidity']  # % Humidity
        wind_speed_mph = data['current']['wind_mph']  # Wind speed in mph
        wind_direction_degrees = data['current']['wind_degree']  # Wind direction in degrees

        R_specific = 287.05
        temperature_kelvin = temperature_c + 273.15
        air_density = pressure_mb * 100 / (R_specific * temperature_kelvin) * (1 - 0.378 * (humidity / 100))

        return {
            'temperature_c': temperature_c,
            'pressure_mb': pressure_mb,
            'humidity': humidity,
            'wind_speed_mph': wind_speed_mph,
            'wind_direction_degrees': wind_direction_degrees,
            'air_density': air_density
        }
    else:
        print("Error fetching weather data:", response.status_code)
        return None

if __name__ == "__main__":
    ipinfo_api_key = "7f28228c490688"  # IPinfo API key
    weather_api_key = "dfd272a5b1ef4f66a14201359241509"  # WeatherAPI key

    # Fetch user's location (latitude, longitude) from IPinfo
    location = get_user_location(ipinfo_api_key)

    if location:
        print(f"User Location: Latitude = {location['latitude']}, Longitude = {location['longitude']}")
        # Fetch weather data based on user's GPS coordinates
        weather = get_weather_data(location['latitude'], location['longitude'], weather_api_key)

        if weather:
            print(f"Weather Conditions:\n"
                  f"Temperature: {weather['temperature_c']}°C\n"
                  f"Air Pressure: {weather['pressure_mb']} mb\n"
                  f"Humidity: {weather['humidity']}%\n"
                  f"Wind Speed: {weather['wind_speed_mph']} mph\n"
                  f"Wind Direction: {weather['wind_direction_degrees']}°\n"
                  f"Air Density: {weather['air_density']} kg/m³")
        else:
            print("Failed to retrieve weather data.")
    else:
        print("Failed to retrieve location data.")

