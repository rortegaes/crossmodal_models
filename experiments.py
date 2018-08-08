from data_loading import gen_text, gen_images, gen_cross

def exp_title_abs(granularity):
    if (granularity == "2clusters"):
        num_class = 2
        dataset = "title_abstract_2clusters.h5"
        exp_weights = "title_abstract_2clusters_weights.h5"
    if (granularity == "5clusters"):
        num_class = 5
        dataset = "title_abstract_5class.h5"
        exp_weights = "title_abstract_5class_weights.h5"
    else:
        print("Error calling method. Options: '2clusters' or '5class'")
        break

    batchSize= 64
    n_papers = #NUMERO DE PAPERS
    print ("Numero de imagenes: "+str(n_papers))
    print ("Numero de clases: "+str(num_class))

    kfold = KFold(n_splits=10, shuffle=True)
    precisions = []
    recalls = []
    f1s = []
    cont = 1

    for train, test in kfold.split([None] * n_papers):
      model = Sequential()
      model.add(Embedding(len(word_index)+1, 300, embeddings_initializer="uniform", input_length=max_sequence_length, trainable=True))
      model.add(Conv1D(512, 5, activation="relu"))
      model.add(MaxPooling1D(5))
      model.add(Conv1D(512, 5, activation="relu"))
      model.add(MaxPooling1D(5))
      model.add(Conv1D(512, 5, activation="relu"))
      model.add(MaxPooling1D(35))
      model.add(Reshape((1,1,512)))
      model.add(Flatten())
      model.add(Dense(128, activation='relu'))
      model.add(Dense(num_class, activation='softmax'))
      model.load_weights(exp_weights)

      db = h5py.File(dataset, "r")
      labels_test = db["labels"][test,:]
      db.close()
      
      pred = model.predict_generator(gen_text(dataset, test, batchSize=batchSize,shuffle=False), steps = len(test)//batchSize) 
      maximos = np.argmax(pred,axis=1)
      predNew = np.zeros(np.shape(pred))
      for i in range(len(predNew)):
        predNew[i,maximos[i]]=1
      print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew))
      precisions.append(precision_score(labels_test[0:batchSize*(len(test)//batchSize)], predNew, average="weighted"))
      recalls.append(recall_score(labels_test[0:batchSize*(len(test)//batchSize)], predNew, average="weighted"))
      f1s.append(f1_score(labels_test[0:batchSize*(len(test)//batchSize)], predNew, average="weighted"))
      cont = cont+1
    print("Precision: %.2f (+/- %.2f)" % (np.mean(precisions), np.std(precisions)))
    print("Recall: %.2f (+/- %.2f)" % (np.mean(recalls), np.std(recalls)))
    print("F1 Score: %.2f (+/- %.2f)" % (np.mean(f1s), np.std(f1s)))

def exp_captions(granularity, modality):
    if (granularity == "2clusters"):
        num_class = 2
        dataset = "captions_2clusters.h5"
        if (modality == "unimodal"):
            exp_weights = "captions_2clusters_weights.h5"
        if (modality == "crossmodal"):
            exp_weights = "captions_2clusters_cross_weights.h5"
        else:
            print("Error calling method. Options: 'unimodal' or 'crossmodal'")
            break
    if (granularity == "5clusters"):
        num_class = 5
        dataset = "captions_5class.h5"
        if (modality == "unimodal"):
            exp_weights = "captions_5class_weights.h5"
        if (modality == "crossmodal"):
            exp_weights = "captions_5class_cross_weights.h5"
        else:
            print("Error calling method. Options: 'unimodal' or 'crossmodal'")
            break
    else:
        print("Error calling method. Options: '2clusters' or '5class'")
        break

    batchSize= 64
    n_papers = #NUMERO DE PAPERS
    print ("Numero de imagenes: "+str(n_papers))
    print ("Numero de clases: "+str(num_class))

    kfold = KFold(n_splits=10, shuffle=True)
    precisions = []
    recalls = []
    f1s = []
    cont = 1

    for train, test in kfold.split([None] * n_papers):
      model = Sequential()
      model.add(Embedding(len(word_index)+1, 300, embeddings_initializer="uniform", input_length=max_sequence_length, trainable=True))
      model.add(Conv1D(512, 5, activation="relu"))
      model.add(MaxPooling1D(5))
      model.add(Conv1D(512, 5, activation="relu"))
      model.add(MaxPooling1D(5))
      model.add(Conv1D(512, 5, activation="relu"))
      model.add(MaxPooling1D(35))
      model.add(Reshape((1,1,512)))
      model.add(Flatten())
      model.add(Dense(128, activation='relu'))
      model.add(Dense(num_class, activation='softmax'))
      model.load_weights(exp_weights)

      db = h5py.File(dataset, "r")
      labels_test = db["labels"][test,:]
      db.close()
      
      pred = model.predict_generator(gen_text(dataset, test, batchSize=batchSize,shuffle=False), steps = len(test)//batchSize) 
      maximos = np.argmax(pred,axis=1)
      predNew = np.zeros(np.shape(pred))
      for i in range(len(predNew)):
        predNew[i,maximos[i]]=1
      print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew))
      precisions.append(precision_score(labels_test[0:batchSize*(len(test)//batchSize)], predNew, average="weighted"))
      recalls.append(recall_score(labels_test[0:batchSize*(len(test)//batchSize)], predNew, average="weighted"))
      f1s.append(f1_score(labels_test[0:batchSize*(len(test)//batchSize)], predNew, average="weighted"))
      cont = cont+1
    print("Precision: %.2f (+/- %.2f)" % (np.mean(precisions), np.std(precisions)))
    print("Recall: %.2f (+/- %.2f)" % (np.mean(recalls), np.std(recalls)))
    print("F1 Score: %.2f (+/- %.2f)" % (np.mean(f1s), np.std(f1s)))

def exp_figures(granularity, modality):
    if (granularity == "2clusters"):
        num_class = 2
        dataset = "figures_2clusters.h5"
        if (modality == "unimodal"):
            exp_weights = "figures_2clusters_weights.h5"
        if (modality == "crossmodal"):
            exp_weights = "figures_2clusters_cross_weights.h5"
        else:
            print("Error calling method. Options: 'unimodal' or 'crossmodal'")
            break
    if (granularity == "5clusters"):
        num_class = 5
        dataset = "figures_5class.h5"
        if (modality == "unimodal"):
            exp_weights = "figures_5class_weights.h5"
        if (modality == "crossmodal"):
            exp_weights = "figures_5class_cross_weights.h5"
        else:
            print("Error calling method. Options: 'unimodal' or 'crossmodal'")
            break
    else:
        print("Error calling method. Options: '2clusters' or '5class'")
        break

    kfold = KFold(n_splits=10, shuffle=True)
    precisions = []
    recalls = []
    f1s = []
    cont = 1

    for train, test in kfold.split([None] * n_images):
      model = Sequential()
      model.add(InputLayer(input_shape=(224,224,3)))
      model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(Conv2D(64, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(2))
      model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(Conv2D(128, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(2))
      model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(Conv2D(256, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(MaxPooling2D(2))
      model.add(Conv2D(512, (3,3), padding = "same", activation="relu"))
      model.add(BatchNormalization())
      model.add(Conv2D(512, (3,3), padding = "same", activation="relu")) 
      model.add(BatchNormalization())
      model.add(MaxPooling2D((28,28),2))
      model.add(Flatten())
      model.add(Dense(128, activation='relu'))
      model.add(Dense(num_class, activation='softmax'))
      model.load_weights(exp_weights)
      
      db = h5py.File(dataset, "r")
      labels_test = db["labels"][test,:]
      db.close()
      
      pred = model.predict_generator(gen_images(dataset, test, batchSize=batchSize, shuffle=False), steps = len(test)//batchSize) 
      maximos = np.argmax(pred,axis=1)
      predNew = np.zeros(np.shape(pred))
      for i in range(len(predNew)):
        predNew[i,maximos[i]]=1
      print(classification_report(labels_test[0:batchSize*(len(test)//batchSize)], predNew))
      precisions.append(precision_score(labels_test[0:batchSize*(len(test)//batchSize)], predNew, average="weighted"))
      recalls.append(recall_score(labels_test[0:batchSize*(len(test)//batchSize)], predNew, average="weighted"))
      f1s.append(f1_score(labels_test[0:batchSize*(len(test)//batchSize)], predNew, average="weighted"))
      cont = cont+1
    print("Precision: %.2f (+/- %.2f)" % (np.mean(precisions), np.std(precisions)))
    print("Recall: %.2f (+/- %.2f)" % (np.mean(recalls), np.std(recalls)))
    print("F1 Score: %.2f (+/- %.2f)" % (np.mean(f1s), np.std(f1s)))

