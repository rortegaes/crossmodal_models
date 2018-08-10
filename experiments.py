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
    n_papers = 46953
    print ("Numero de imagenes: "+str(n_papers))
    print ("Numero de clases: "+str(num_class))

    kfold = KFold(n_splits=10, shuffle=True)
    precisions = []
    recalls = []
    f1s = []
    cont = 1

    for train, test in kfold.split([None] * n_papers):
      model = models.generateTextualModel()
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
    n_captions = 82396
    print ("Numero de imagenes: "+str(n_captions))
    print ("Numero de clases: "+str(num_class))

    kfold = KFold(n_splits=10, shuffle=True)
    precisions = []
    recalls = []
    f1s = []
    cont = 1

    for train, test in kfold.split([None] * n_captions):
      model = models.generateTextualModel()
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
    
    n_images = 82396
    kfold = KFold(n_splits=10, shuffle=True)
    precisions = []
    recalls = []
    f1s = []
    cont = 1

    for train, test in kfold.split([None] * n_images):
      model = models.generateVisualModel()
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

def exp_cross():
    num_class = 2
    n_images = 87402
    dataset = "cross.h5"
    exp_weights = "cross_weights.h5"

    kfold = KFold(n_splits=10, shuffle=True)
    precisions = []
    recalls = []
    f1s = []
    cont = 1

    for train, test in kfold.split([None] * n_images):
      model = models.generateCrossModel(num_class)
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
