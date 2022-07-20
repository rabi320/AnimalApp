import streamlit as st
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from PIL import Image

#title
st.markdown("![](https://j.gifs.com/W72OxE.gif)")
st.markdown("""
# What is this creature?! 🐱‍👓

please upload a Picture of the animal you want to identify:

""")

#the classes
my_classes = ['antelope',
 'badger',
 'bat',
 'bear',
 'bee',
 'beetle',
 'bison',
 'boar',
 'butterfly',
 'cat',
 'caterpillar',
 'chimpanzee',
 'cockroach',
 'cow',
 'coyote',
 'crab',
 'crow',
 'deer',
 'dog',
 'dolphin',
 'donkey',
 'dragonfly',
 'duck',
 'eagle',
 'elephant',
 'flamingo',
 'fly',
 'fox',
 'goat',
 'goldfish',
 'goose',
 'gorilla',
 'grasshopper',
 'hamster',
 'hare',
 'hedgehog',
 'hippopotamus',
 'hornbill',
 'horse',
 'hummingbird',
 'hyena',
 'jellyfish',
 'kangaroo',
 'koala',
 'ladybugs',
 'leopard',
 'lion',
 'lizard',
 'lobster',
 'mosquito',
 'moth',
 'mouse',
 'octopus',
 'okapi',
 'orangutan',
 'otter',
 'owl',
 'ox',
 'oyster',
 'panda',
 'parrot',
 'pelecaniformes',
 'penguin',
 'pig',
 'pigeon',
 'porcupine',
 'possum',
 'raccoon',
 'rat',
 'reindeer',
 'rhinoceros',
 'sandpiper',
 'seahorse',
 'seal',
 'shark',
 'sheep',
 'snake',
 'sparrow',
 'squid',
 'squirrel',
 'starfish',
 'swan',
 'tiger',
 'turkey',
 'turtle',
 'whale',
 'wolf',
 'wombat',
 'woodpecker',
 'zebra']

@st.cache()
def model_loader(model_path = None, label = None):
    """
    Args:
    >model_path (str)- the path to current model.
    >label (str)-the label to predict. 
    """
    model = models.resnet18()
    num_features = model.fc.in_features
    #new layer
    model.fc = nn.Sequential(
                          nn.Linear(model.fc.in_features, 512),
                          nn.ReLU(),
                          nn.Dropout(0.5),
                          nn.Linear(512, 256),
                          nn.ReLU(),
                          nn.Dropout(0.5),
                          nn.Linear(256, len(my_classes)),
                          nn.LogSoftmax(dim=1))


    model.load_state_dict(torch.load(model_path,map_location=torch.device('cpu')))
    return model.eval()

model = model_loader(model_path = "Data/animal_model.pt", label = my_classes)

transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(p = 0.5)
])

uploaded_file = st.file_uploader('File uploader')

if st.button("Identify"):
    try:
        st.image(uploaded_file, width=100)

        @st.cache()
        def predictor(img, n=5):
            """
            Args: 
                >img - the image to predict.
                >n - number of top probabilities.
            
            Outputs:
                >pred - the top prediction.
                > top preds - top n predictions.
            """
            #transform the image
            img = transforms(img)
            # get the class predicted 
            pred = int(np.squeeze(model(img.unsqueeze(0)).data.max(1, keepdim=True)[1].numpy()))
            # the number is also the index for the class label
            pred = my_classes[pred]
            # get model log probabilities
            preds = torch.from_numpy(np.squeeze(model(img.unsqueeze(0)).data.numpy()))
            # convert to prediction probabilities of the top n predictions
            prob_func = nn.Softmax(dim=0)
            preds = prob_func(preds)
            top_preds = torch.topk(preds,n)
            #display at an orgenized fasion
            top_preds = dict(zip([my_classes[i]for i in top_preds.indices],[f"{round(float(i)*100,2)}%" for i in top_preds.values]))
            return pred, top_preds

        pred, preds = predictor(Image.open(uploaded_file), n=5)
        st.balloons()
        st.write("This animal is a",pred)

        df = pd.DataFrame({"Animal":list(preds.keys()),"Probability":list(preds.values())}, index = list(range(1,6)))
        st.write("Top 5 most likely animals:")
        st.dataframe(df)
    except:
        st.write("no picture uploaded, please upload a picture")