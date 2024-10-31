import tensorflow as tf
import numpy as np

# Load our pre trained model
model = tf.keras.models.load_model('saved_model.keras')

# Load single image
test_image = tf.keras.preprocessing.image.load_img('images/mypet.jpg', target_size=(150, 150))
test_array = tf.keras.preprocessing.image.img_to_array(test_image)
test_array = np.array([test_array])

# Predict using model
prediction = model.predict(test_array)

# Interpret the results
print ("Test image evaluation: ", prediction)
if prediction[0][0] > prediction[0][1]:
    print ("Test image is of category: cat")
else:
    print ("Test image is of category: dog")
