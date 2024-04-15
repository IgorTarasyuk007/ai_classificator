from skimage.transform import resize 
import tensorflow as tf
import keras.utils as ku
import numpy as np
from PIL import Image 
model = tf.keras.models.load_model("my_model88.h5", compile=False)
number_to_class = ['containers', 'cups', 'bottles']
test_img_p = 'processed_images/'
def checkImg(test_img, number_to_class):
    img = ku.load_img(test_img, target_size = (32,32))
    img = ku.img_to_array(img, dtype=np.uint8)
    img = np.array(img)/255.0
    prediction = model.predict(img[np.newaxis, ...])

    #print("Predicted shape",p.shape)
    print("Probability:",np.max(prediction[0], axis=-1), prediction)
    predicted_class = number_to_class[np.argmax(prediction[0], axis=-1)]
    #return predicted_class == test_img.split('/')[2].split('_')[0]
    return predicted_class
att = []
for i in range(10):
    for j in number_to_class:
        try:
            #img = Image.open('tf_files/'+j+'/'+j+'_'+str(i)+'.jpg')
            #img.show()
            f = checkImg(test_img_p+j+'/'+j+'_'+str(i)+'.jpg', number_to_class)
            print(f"Predict: {f}\nPhoto's class: {j}")
            #t = input()
        except:
            pass

#        att.append(f)
#print(f"True/False = {att.count(True)}/{att.count(False)}")