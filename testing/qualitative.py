from keras import applications
from keras import backend as K
import cv2
import random

def getIndices(granularity):
    if (granularity == "2clusters"):
        num_class = 2
        dataset = "figures_2clusters.h5"
        exp_weights_uni = "figures_2clusters_weights.h5"
        exp_weights_mix = "figures_2clusters_weights_cross.h5"

    if (granularity == "5clusters"):
        num_class = 5
        dataset = "figures_5class.h5"
        exp_weights_uni = "figures_5class_weights.h5"
        exp_weights_mix = "figures_5class_weights_cross.h5"

    n_images = 8192
    test = range (0,n_images)
    batchSize = 64

    modelUni = models.generateVisualModel(num_class)
    modelUni.load_weights(exp_weights_uni)

    pred1 = modelUni.predict_generator(gen_images(dataset, test, batchSize=batchSize,shuffle=False),steps = len(test)//batchSize)
    maximos1 = np.argmax(pred1,axis=1)
    predNew1 = np.zeros(np.shape(pred1))
    for i in range(len(predNew1)):
        predNew1[i,maximos1[i]]=1
        
    modelMix = models.generateVisualModel(num_class)
    modelMix.load_weights(exp_weights_mix)

    pred2 = modelMix.predict_generator(gen_images(dataset, test, batchSize=batchSize,shuffle=False),steps = len(test)//batchSize)
    maximos2 = np.argmax(pred2,axis=1)
    predNew2 = np.zeros(np.shape(pred2))
    for i in range(len(predNew2)):
        predNew2[i,maximos2[i]]=1

    db = h5py.File(dataset, "r")
    labels_test = db["labels"][test,:]
    db.close()

    indices_res = []
    for j in range(0,n_images):
        if (np.array_equal(predNew2[j],labels_test[j]) and (np.array_equal(predNew1[j],labels_test[j]))== False):
            indices_res.append(j)
    correct_class = []
    for k in indices_res:
        for l in range(0,num_class):
            if (predNew2[k,l]==1):
                correct_class.append(l)
    return indices_res, correct_class

def getCAM(granularity):
    if (granularity == "2clusters"):
        num_class = 2
        dataset = "figures_2clusters.h5"
        exp_weights_uni = "figures_2clusters_weights.h5"
        exp_weights_mix = "figures_2clusters_weights_cross.h5"

    if (granularity == "5clusters"):
        num_class = 5
        dataset = "figures_5class.h5"
        exp_weights_uni = "figures_5class_weights.h5"
        exp_weights_mix = "figures_5class_weights_cross.h5"
        
    indices_res, correct_class = getIndices(granularity)
    i0 = random.choice(indices_res)
    i1 = random.choice(indices_res)
    i2 = random.choice(indices_res)
    i3 = random.choice(indices_res)
    indices = [i0,i1,i2,i3]
    for indice in indices:
        predicted_class = correct_class[indice]
        predMix = pred2[indice,predicted_class]
        predUni = pred1[indice,predicted_class]
        diff = predMix - predUni
        print ("Diferencia de "+str(indice)+": "+str(diff*100)+"(Uni: "+str(predUni)+"; Mix: "+str(predMix)+")")
        list_img = []
        db = h5py.File(dataset, "r")
        original_img = db["images"][indice,:,:,:]
        img_fr_a = Image.fromarray(original_img, 'RGB')
        img_fr_a.save(str(indice)+".png")
        db.close()
        list_img.append(original_img)
        img = np.array(list_img)
            
        cam, heatmap = grad_cam(exp_weights_uni, img, predicted_class)
        cv2.imwrite("uni-"+str(indice)+".png", cam)
        cv2.imwrite("uni-heatmap"+str(indice)+".png", heatmap)
        cam, heatmap = grad_cam(exp_weights_mix, img, predicted_class)
        cv2.imwrite("mix-"+str(indice)+".png", cam)
        cv2.imwrite("mix-heatmap"+str(indice)+".png", heatmap)
    return indices
  
def target_category_loss(x, category_index, nb_classes):
    return tf.multiply(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def register_gradient():
    if "GuidedBackProp" not in ops._gradient_registry._registry:
        @ops.RegisterGradient("GuidedBackProp")
        def _GuidedBackProp(op, grad):
            dtype = op.inputs[0].dtype
            return grad * tf.cast(grad > 0., dtype) * \
                tf.cast(op.inputs[0] > 0., dtype)
            
def modify_backprop(model, name):
    g = tf.get_default_graph()
    with g.gradient_override_map({'Relu': name}):

        # get layers that have an activation
        layer_dict = [layer for layer in model.layers[1:]
                      if hasattr(layer, 'activation')]

        # replace relu activation
        for layer in layer_dict:
            if layer.activation == keras.activations.relu:
                layer.activation = tf.nn.relu

        # re-instanciate a new model
        new_model = Sequential()
        new_model.add(InputLayer(input_shape=(224,224,3)))
        new_model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
        new_model.add(BatchNormalization())
        new_model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
        new_model.add(BatchNormalization())
        new_model.add(MaxPooling2D(2))
        new_model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
        new_model.add(BatchNormalization())
        new_model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
        new_model.add(BatchNormalization())
        new_model.add(MaxPooling2D(2))
        new_model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
        new_model.add(BatchNormalization())
        new_model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
        new_model.add(BatchNormalization())
        new_model.add(MaxPooling2D(2))
        new_model.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
        new_model.add(BatchNormalization())
        new_model.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
        new_model.add(BatchNormalization())
        new_model.add(MaxPooling2D((28,28),2))
        new_model.add(Flatten())
        new_model.add(Dense(128, activation='relu'))
        new_model.add(Dense(num_class, activation='softmax'))
        #new_model.load_weights('./models/modelMixHvsT1.h5')
    return new_model

def compile_saliency_function(model):
    input_img = model.input
    layer_output = model.layers[-6].output
    max_output = K.max(layer_output, axis=3)
    saliency = K.gradients(K.sum(max_output), input_img)[0]
    return K.function([input_img, K.learning_phase()], [saliency])

def deprocess_image(x):
    '''
    Same normalization as in:
    https://github.com/fchollet/keras/blob/master/examples/conv_filter_visualization.py
    '''
    if np.ndim(x) > 3:
        x = np.squeeze(x)
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    if K.image_dim_ordering() == 'th':
        x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x

def grad_cam(weights, image, category_index):
    input_model = Sequential()
    input_model.add(InputLayer(input_shape=(224,224,3)))
    input_model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    input_model.add(BatchNormalization())
    input_model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
    input_model.add(BatchNormalization())
    input_model.add(MaxPooling2D(2))
    input_model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    input_model.add(BatchNormalization())
    input_model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
    input_model.add(BatchNormalization())
    input_model.add(MaxPooling2D(2))
    input_model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    input_model.add(BatchNormalization())
    input_model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
    input_model.add(BatchNormalization())
    input_model.add(MaxPooling2D(2))
    input_model.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
    input_model.add(BatchNormalization())
    input_model.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
    input_model.add(BatchNormalization())
    input_model.add(MaxPooling2D((28,28),2))
    input_model.add(Flatten())
    input_model.add(Dense(128, activation='relu'))
    input_model.add(Dense(num_class, activation='softmax'))
    input_model.load_weights(weights)
    nb_classes = num_class
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    input_model.add(Lambda(target_layer,
                     output_shape = target_category_loss_output_shape))
    
    loss = K.sum(input_model.layers[-1].output)
    conv_output =  input_model.layers[-7].output
    grads = normalize(K.gradients(loss, conv_output)[0])
    gradient_function = K.function([input_model.layers[0].input], [conv_output, grads])

    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]

    weights = np.mean(grads_val, axis = (0, 1))
    cam = np.ones(output.shape[0 : 2], dtype = np.float32)

    for i, w in enumerate(weights):
        cam += w * output[:, :, i]

    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    visible_heatmap = cv2.applyColorMap(np.uint8(255*heatmap), cv2.COLORMAP_JET)
    return np.uint8(cam), visible_heatmap
