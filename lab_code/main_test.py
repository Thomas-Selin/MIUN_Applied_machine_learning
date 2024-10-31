import tensorflow as tf
import numpy as np
import os

# Load our pre-trained model
model = tf.keras.models.load_model('saved_model.keras')

# Directory containing test images
test_dir = 'images/test'

# Loop through all images in the test directory
for filename in os.listdir(test_dir):
    if filename.endswith(".jpg") or filename.endswith(".png"):  # Add other image formats if needed
        # Load single image
        test_image_path = os.path.join(test_dir, filename)
        test_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(150, 150))
        test_array = tf.keras.preprocessing.image.img_to_array(test_image)
        test_array = np.expand_dims(test_array, axis=0)  # Add batch dimension

        # Predict using model
        prediction = model.predict(test_array)

        # Interpret the results
        print(f"Test image {filename} evaluation: {prediction}")
        if prediction[0][0] > prediction[0][1]:
            print(f"Test image {filename} is of category: cat")
        else:
            print(f"Test image {filename} is of category: dog")
