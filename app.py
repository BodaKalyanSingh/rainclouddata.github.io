import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
import numpy as np
from tensorflow.keras.models import load_model
import logging
from logging.handlers import RotatingFileHandler

app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'img'
model_path = 'cloud_classification_model_correct.h5'

model = load_model(model_path)

cloud_details = {
    "Ci": {
        "Name of the Cloud":"Cirrus",
        "filename": "Ci/Ci-N004.jpg",
        "cloud_type": "Cirrus",
        "Probability of Precipitation (PoP)": "< 5%",
        "Expected Rainfall": "0 mm",
        "Relative Humidity": "20-40%",
        "Wind Speed": "20-50 km/h",
        "Temperature": "No significant change",
        "Altitude": "High (6,000-12,000 meters)",
        "Indicates": "Approaching warm front or upper-level moisture",
        "Sky Coverage": "Thin, wispy",
        "Rainfall Onset Time": "Generally 24-48 hours before a change in weather",
        "Duration of Rainfall": "nan",
        "Rainfall Intensity": "nan",
        "Future Weather Trends": "Fair weather expected",
        "Cloud Movement Analysis": "Clouds moving eastward",
        "Description": "Cirrus clouds are thin and wispy, often resembling delicate strands of hair. They are composed of ice crystals and typically indicate fair weather, although they can signal a change in the weather within the next 24-48 hours.",
        "Formation": "Formed at high altitudes, cirrus clouds develop when water vapor undergoes deposition at temperatures below freezing.",
        "Activity Suggestions": "Great for outdoor activities. Enjoy a clear day!",
        "Type of Rain Droplets": "nan"
    },
    # Add details for other cloud types here...
    'Cs': {
        "Name of the Cloud":"Cirrostratus",
        'cloud_type': 'Cirrostratus',
        'Probability of Precipitation (PoP)': '10-30% within 24 hours',
        'Expected Rainfall': '0 mm initially, 2-5 mm as front approaches',
        'Relative Humidity': '30-50%',
        'Wind Speed': '10-30 km/h',
        'Temperature': 'Can precede a warming trend (15-25°C or 59-77°F)',
        'Altitude': 'High (6,000-12,000 meters)',
        'Indicates': 'Incoming frontal system',
        'Sky Coverage': 'Transparent, covers the sky',
        'Rainfall Onset Time': '12-24 hours before rain',
        'Duration of Rainfall': 'Light to moderate',
        'Rainfall Intensity': 'Light to moderate',
        'Future Weather Trends': 'Weather change likely, increasing cloudiness',
        'Cloud Movement Analysis': 'Clouds moving northward',
        'Description': 'Cirrostratus clouds form a thin, white veil that covers the sky. They often produce halos around the sun or moon and are associated with approaching weather fronts that may bring precipitation.',
        'Formation': 'Cirrostratus clouds form when a layer of moist air is lifted to high altitudes, causing the moisture to cool and condense into ice crystals.',
        'Activity Suggestions': 'Good day for indoor activities or light outdoor tasks.',
        'Type of Rain Droplets': 'Ice crystals'
    },
    'Cc': {
        "Name of the Cloud":"Cirrocumulus",
        'cloud_type': 'Cirrocumulus',
        'Probability of Precipitation (PoP)': '< 5%',
        'Expected Rainfall': '0 mm',
        'Relative Humidity': '30-50%',
        'Wind Speed': '20-50 km/h',
        'Temperature': 'No significant change',
        'Altitude': 'High (6,000-12,000 meters)',
        'Indicates': 'High-altitude turbulence',
        'Sky Coverage': 'Small, white patches',
        'Rainfall Onset Time': 'Not typically associated with rain',
        'Duration of Rainfall': 'None',
        'Rainfall Intensity': 'None',
        'Future Weather Trends': 'Stable weather',
        'Cloud Movement Analysis': 'Clouds moving westward',
        'Description': 'Cirrocumulus clouds appear as small, white patches high in the sky, often in rows. They are composed of ice crystals and typically indicate stable weather conditions.',
        'Formation': 'Cirrocumulus clouds form at high altitudes when moist air undergoes significant cooling and condensation.',
        'Activity Suggestions': 'Perfect for stargazing at night. Enjoy the clear skies!',
        'Type of Rain Droplets': 'None'
    },
    'Ac': {
        "Name of the Cloud":"Altocumulus",
        'cloud_type': 'Altocumulus',
        'Probability of Precipitation (PoP)': '10-20%',
        'Expected Rainfall': '0-2 mm',
        'Relative Humidity': '50-70%',
        'Wind Speed': '10-25 km/h',
        'Temperature': 'Mild (15-25°C or 59-77°F)',
        'Altitude': 'Middle (2,000-6,000 meters)',
        'Indicates': 'Weather variability',
        'Sky Coverage': 'White or gray patches',
        'Rainfall Onset Time': 'Occasional light rain',
        'Duration of Rainfall': 'Short bursts',
        'Rainfall Intensity': 'Light',
        'Future Weather Trends': 'Possible changes in weather',
        'Cloud Movement Analysis': 'Clouds moving southward',
        'Description': 'Altocumulus clouds are mid-level clouds that appear as white or gray patches, often in a wavy pattern. They can indicate changing weather conditions.',
        'Formation': 'Altocumulus clouds form when moist air at middle altitudes undergoes cooling and condensation.',
        'Activity Suggestions': 'Good for morning walks or moderate outdoor activities.',
        'Type of Rain Droplets': 'Small liquid droplets'
    },
    'As': {
        "Name of the Cloud":"Altostratus",
        'cloud_type': 'Altostratus',
        'Probability of Precipitation (PoP)': '30-50%',
        'Expected Rainfall': '2-10 mm',
        'Relative Humidity': '60-80%',
        'Wind Speed': '10-30 km/h',
        'Temperature': 'Cool to mild (10-20°C or 50-68°F)',
        'Altitude': 'Middle (2,000-6,000 meters)',
        'Indicates': 'Approaching warm front',
        'Sky Coverage': 'Gray or blue-gray',
        'Rainfall Onset Time': '6-12 hours before rain',
        'Duration of Rainfall': 'Several hours',
        'Rainfall Intensity': 'Light to moderate',
        'Future Weather Trends': 'Continued cloudiness and rain',
        'Cloud Movement Analysis': 'Clouds moving eastward',
        'Description': 'Altostratus clouds are mid-level clouds that cover the sky with a uniform gray or blue-gray sheet, often preceding a warm front.',
        'Formation': 'Altostratus clouds form when a large mass of moist air is lifted to middle altitudes, causing widespread cooling and condensation.',
        'Activity Suggestions': 'Ideal for indoor activities. Prepare for rain.',
        'Type of Rain Droplets': 'Liquid droplets'
    },
    'Sc': {
        "Name of the Cloud":"Stratocumulus",
        'cloud_type': 'Stratocumulus',
        'Probability of Precipitation (PoP)': '10-20%',
        'Expected Rainfall': '0-2 mm',
        'Relative Humidity': '60-80%',
        'Wind Speed': '5-20 km/h',
        'Temperature': 'Mild (15-25°C or 59-77°F)',
        'Altitude': 'Low (0-2,000 meters)',
        'Indicates': 'Fair weather with some clouds',
        'Sky Coverage': 'Lumpy, grayish-white',
        'Rainfall Onset Time': 'Intermittent drizzle',
        'Duration of Rainfall': 'Short periods',
        'Rainfall Intensity': 'Light',
        'Future Weather Trends': 'Partly cloudy with possible clearing',
        'Cloud Movement Analysis': 'Clouds moving northwestward',
        'Description': 'Stratocumulus clouds are low-level clouds that form in lumpy, grayish-white patches. They are often associated with fair weather but can bring light rain or drizzle.',
        'Formation': 'Stratocumulus clouds form when a layer of moist air near the ground undergoes partial cooling and condensation.',
        'Activity Suggestions': 'Good for short outdoor activities. Keep an eye on the sky for any changes.',
        'Type of Rain Droplets': 'Drizzle'
    },
    'St': {
        "Name of the Cloud":"Stratus",
        'cloud_type': 'Stratus',
        'Probability of Precipitation (PoP)': '10-30%',
        'Expected Rainfall': '0-2 mm',
        'Relative Humidity': '70-90%',
        'Wind Speed': '5-15 km/h',
        'Temperature': 'Cool (10-20°C or 50-68°F)',
        'Altitude': 'Low (0-2,000 meters)',
        'Indicates': 'Overcast skies',
        'Sky Coverage': 'Uniform, gray layer',
        'Rainfall Onset Time': 'Imminent light rain or drizzle',
        'Duration of Rainfall': 'Long periods',
        'Rainfall Intensity': 'Light',
        'Future Weather Trends': 'Persistent overcast conditions',
        'Cloud Movement Analysis': 'Clouds moving northward',
        'Description': 'Stratus clouds are low-level clouds that form a uniform, gray layer covering the sky. They often bring light rain or drizzle and can persist for long periods.',
        'Formation': 'Stratus clouds form when a large mass of moist air is lifted gently to low altitudes, causing extensive cooling and condensation.',
        'Activity Suggestions': 'Plan for indoor activities. Expect overcast conditions.',
        'Type of Rain Droplets': 'Drizzle'
    },
    'Cu': {
        "Name of the Cloud":"Cumulus",
        'cloud_type': 'Cumulus',
        'Probability of Precipitation (PoP)': '< 20%',
        'Expected Rainfall': '0 mm',
        'Relative Humidity': '30-50%',
        'Wind Speed': '10-30 km/h',
        'Temperature': 'Mild (15-25°C or 59-77°F)',
        'Altitude': 'Low to middle (0-6,000 meters)',
        'Indicates': 'Fair weather',
        'Sky Coverage': 'White, fluffy clouds with flat bases',
        'Rainfall Onset Time': 'None',
        'Duration of Rainfall': 'None',
        'Rainfall Intensity': 'None',
        'Future Weather Trends': 'Stable weather',
        'Cloud Movement Analysis': 'Clouds moving southward',
        'Description': 'Cumulus clouds are fluffy, white clouds with flat bases. They typically indicate fair weather with little or no precipitation.',
        'Formation': 'Cumulus clouds form when warm air rises and cools, causing water vapor to condense into droplets or ice crystals.',
        'Activity Suggestions': 'Enjoy outdoor activities under sunny skies!',
        'Type of Rain Droplets': 'None'
    },
    'Cb': {
        "Name of the Cloud":"Cumulonimbus",
        'cloud_type': 'Cumulonimbus',
        'Probability of Precipitation (PoP)': '60-100%',
        'Expected Rainfall': '10-50 mm',
        'Relative Humidity': '60-90%',
        'Wind Speed': '30-50 km/h (gusts up to 100 km/h during storms)',
        'Temperature': 'Can drop significantly',
        'Altitude': 'Low to high (0-12,000 meters)',
        'Indicates': 'Thunderstorms',
        'Sky Coverage': 'Towering, dense clouds with anvil shape',
        'Rainfall Onset Time': 'Imminent heavy rain, hail, or thunderstorms',
        'Duration of Rainfall': 'Variable, from brief showers to prolonged storms',
        'Rainfall Intensity': 'Heavy',
        'Future Weather Trends': 'Severe weather expected',
        'Cloud Movement Analysis': 'Clouds moving rapidly upward',
        'Description': 'Cumulonimbus clouds are massive, towering clouds with an anvil shape at the top. They produce thunderstorms with heavy rain, lightning, hail, and sometimes tornadoes.',
        'Formation': 'Cumulonimbus clouds form when warm, moist air rises rapidly and reaches high altitudes, condensing into large, dense clouds.',
        'Activity Suggestions': 'Seek shelter indoors during thunderstorms. Be cautious of severe weather.',
        'Type of Rain Droplets': 'Heavy raindrops, hailstones'
    }
}


# Add logging
handler = RotatingFileHandler('error.log', maxBytes=10000, backupCount=1)
handler.setLevel(logging.ERROR)
app.logger.addHandler(handler)

def process_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0
        predictions = model.predict(img_array)
        predicted_index = np.argmax(predictions)
        predicted_cloud_type = list(cloud_details.keys())[predicted_index]
        cloud_info = cloud_details[predicted_cloud_type]
        cloud_info['cloud_type'] = predicted_cloud_type
        cloud_info['filename'] = os.path.basename(img_path)
        return cloud_info
    except Exception as e:
        app.logger.error('Error in process_image', exc_info=e)
        return {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        if 'cloudImage' not in request.files:
            return jsonify({"error": "No file part"})
        file = request.files['cloudImage']
        if file.filename == '':
            return jsonify({"error": "No selected file"})
        if file:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            predicted_cloud_info = process_image(file_path)
            return jsonify(predicted_cloud_info)
    except Exception as e:
        app.logger.error('Error processing upload', exc_info=e)
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
