import numpy as np
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

# 1. SETUP
MODEL_PATH = 'neurodetect_best.keras'  # Your saved model
IMG_SIZE = (224, 224)

# IMPORTANT: These must match the ALPHABETICAL order of your folders exactly.
# This is how Keras assigns indices (0, 1, 2...)
CLASS_NAMES = [
    'Glioma', 
    'Meningioma', 
    'Mild_Demented', 
    'Moderate_Demented', 
    'No_Disease', 
    'Pituitary', 
    'Very_Mild_Demented'
]

def predict_image(image_path):
    print(f"[INFO] Loading image: {image_path}")
    
    # 2. LOAD MODEL
    # We use compile=False because we don't need to train, just predict.
    # It makes loading faster.
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)

    # 3. PREPROCESS IMAGE
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found at that path.")
        return

    # Convert BGR (OpenCV standard) to RGB (Model standard)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize to 224x224
    img_resized = cv2.resize(img_rgb, IMG_SIZE)
    
    # Scale pixel values to [0, 1] (Normalization)
    img_array = img_resized.astype('float32') / 255.0
    
    # Add the batch dimension (1, 224, 224, 3)
    # The model expects a batch of images, even if it's just one.
    img_batch = np.expand_dims(img_array, axis=0)

    # 4. PREDICT
    predictions = model.predict(img_batch)
    
    # Get the index of the highest probability
    predicted_index = np.argmax(predictions)
    confidence_score = predictions[0][predicted_index]
    predicted_label = CLASS_NAMES[predicted_index]

    # 5. DISPLAY RESULTS
    print(f"\n✅ Prediction: {predicted_label}")
    print(f"📊 Confidence: {confidence_score * 100:.2f}%")
    
    # Show the image with the label
    plt.figure(figsize=(5, 5))
    plt.imshow(img_rgb)
    plt.title(f"{predicted_label} ({confidence_score*100:.1f}%)")
    plt.axis('off')
    plt.show()

# HOW TO RUN IT
test_image_path = '/mnt/e/D-Temp/5thSem/Dataset/Glioma/Te-gl_0144.jpg'                      #Glioma
test_image_path_1 = '/mnt/e/D-Temp/5thSem/Dataset/Glioma/Te-glTr_0004.jpg'                  #Glioma
test_image_path_2 = '/mnt/e/D-Temp/5thSem/Dataset/Glioma/Tr-gl_0820.jpg'                    #Glioma
test_image_path_3 = '/mnt/e/D-Temp/5thSem/Dataset/Meningioma/Te-me_0278.jpg'                #Meningioma
test_image_path_4 = '/mnt/e/D-Temp/5thSem/Dataset/Meningioma/Tr-me_0621.jpg'                #Meningioma
test_image_path_5 = '/mnt/e/D-Temp/5thSem/Dataset/Meningioma/Tr-me_1267.jpg'                #Meningioma
test_image_path_6 = '/mnt/e/D-Temp/5thSem/Dataset/Mild_Demented/0daa365f-113a-4142-884f-84de88d1d209.jpg'           #Mild_D
test_image_path_7 = '/mnt/e/D-Temp/5thSem/Dataset/Mild_Demented/48446ec0-0cde-423e-83f9-eb50883adc42.jpg'           #Mild_D
test_image_path_8 = '/mnt/e/D-Temp/5thSem/Dataset/Mild_Demented/ef0004b6-82e5-44ac-b098-9984df7657e6.jpg'           #Mild_D
test_image_path_9 = '/mnt/e/D-Temp/5thSem/Dataset/Moderate_Demented/0c02ad6a-1661-43aa-8a11-cd7bff140343.jpg'       #Moderate_D
test_image_path_10 = '/mnt/e/D-Temp/5thSem/Dataset/Moderate_Demented/8204f024-0ef3-4c3a-ba44-75d29a229e40.jpg'      #Moderate_D
test_image_path_11 = '/mnt/e/D-Temp/5thSem/Dataset/Moderate_Demented/aug_8571_d9e4bfbc-2425-45e3-8539-153250c2a279.jpg' #Moderate_D
test_image_path_12 = '/mnt/e/D-Temp/5thSem/Dataset/No_Disease/50c42848-08f2-4131-82c9-067a95bf825b.jpg'             #No_Disease
test_image_path_13 = '/mnt/e/D-Temp/5thSem/Dataset/No_Disease/cb5197a6-6433-4e70-8b02-23b97cb8f326.jpg'             #No_Disease
test_image_path_14 = '/mnt/e/D-Temp/5thSem/Dataset/No_Disease/nonDem2323.jpg'                                       #No_Disease
test_image_path_15 = '/mnt/e/D-Temp/5thSem/Dataset/Pituitary/Te-pi_0295.jpg'                #Pituitary
test_image_path_16 = '/mnt/e/D-Temp/5thSem/Dataset/Pituitary/Tr-pi_0531.jpg'                #Pituitary
test_image_path_17 = '/mnt/e/D-Temp/5thSem/Dataset/Pituitary/Tr-pi_1044.jpg'                #Pituitary
test_image_path_18 = '/mnt/e/D-Temp/5thSem/Dataset/Very_Mild_Demented/99012802-0415-443e-b833-ae1385aa25e4.jpg'     #VeryMild-D
test_image_path_19 = '/mnt/e/D-Temp/5thSem/Dataset/Very_Mild_Demented/be6f47fe-a01e-4d29-a1ba-8c3150e279fb.jpg'     #VeryMild_D
test_image_path_20 = '/mnt/e/D-Temp/5thSem/Dataset/Very_Mild_Demented/verymildDem1202.jpg'                          #VeryMild_D

predict_image(test_image_path)
predict_image(test_image_path_1)
predict_image(test_image_path_2)
predict_image(test_image_path_3)
predict_image(test_image_path_4)
predict_image(test_image_path_5)
predict_image(test_image_path_6)
predict_image(test_image_path_7)
predict_image(test_image_path_8)
predict_image(test_image_path_9)
predict_image(test_image_path_10)
predict_image(test_image_path_11)
predict_image(test_image_path_12)
predict_image(test_image_path_13)
predict_image(test_image_path_14)
predict_image(test_image_path_15)
predict_image(test_image_path_16)
predict_image(test_image_path_17)
predict_image(test_image_path_18)
predict_image(test_image_path_19)
predict_image(test_image_path_20)