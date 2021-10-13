import torch
from fastbook import *
from astropy.io import fits
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import urllib.request
import os
import time
from sklearn.metrics import classification_report

class Main_Classifier:

    def __init__(self, classifier_name, img_size:int = 512):
        self.classifier_name = classifier_name
        self.width = img_size
        self.height = img_size
        self.classes = classifier_name.split('_')
        self.classifier = 0

    def image_coords(self, address):
        dat_hud=fits.open(f'{address}/{self.classifier_name}_Catalogue.fits')
        dat=dat_hud[1].data

        Class_one=dat[self.class1]
        Class_Two=dat[self.class2]

        Class_one_dat = dat[Class_one==1]
        Class_two_dat = dat[Class_Two==1]

        Class_one_ras = Class_one_dat['RA']
        Class_one_decs = Class_one_dat['DEC']
        Class_two_ras = Class_two_dat['RA']
        Class_two_decs = Class_two_dat['DEC']

        Class_one_urls = self.get_image_urls(ras=Class_one_ras, decs=Class_one_decs)
        Class_two_urls = self.get_image_urls(ras=Class_two_ras, decs=Class_two_decs)
        
        assert len(Class_one_urls) == len(Class_one_ras) 
        assert len(Class_two_urls) == len(Class_two_ras)

        return Class_one_urls, Class_two_urls

    def get_image_urls(self, ras, decs):
        urls=[]
        for ra,dec in zip(ras,decs):
            urls.append(f'http://skyserver.sdss.org/dr16/SkyServerWS/ImgCutout/getjpeg?ra={ra}&dec={dec}&width={self.width}&height={self.height}&scale=0.1&.jpg')
        return urls

    def download_imgs(self, num_lim, address, dataset_add):
        
        if os.path.exists(f'{address}/Data_{num_lim[1]-num_lim[0]}') == False:
            if os.path.exists(f'{address}') == False:
              os.mkdir(f'{address}')
            os.mkdir(f'{address}/Data_{num_lim[1]-num_lim[0]}')
            os.mkdir(f'{address}/Data_{num_lim[1]-num_lim[0]}/{self.classes[0]}')
            os.mkdir(f'{address}/Data_{num_lim[1]-num_lim[0]}/{self.classes[1]}')
            
        Class_one_urls, Class_two_urls = self.image_coords(dataset_add)

        time1=time.perf_counter()
        for i in range(num_lim[0], num_lim[1]+1):
            urllib.request.urlretrieve(Class_one_urls[i], f'{address}/Data_{num_lim[1]-num_lim[0]}/{self.classes[0]}/{i}.jpg')
            urllib.request.urlretrieve(Class_two_urls[i], f'{address}/Data_{num_lim[1]-num_lim[0]}/{self.classes[1]}/{i}.jpg')
        time2=time.perf_counter()
        print('time taken : ', time2-time1)
        
    def test_imgs(self, nums, address, dataset_add):
        if os.path.exists(f'{address}') == False:
            os.mkdir(f'{address}')
            
        Class_one_urls, Class_two_urls = self.image_coords(dataset_add)
        
        time1=time.perf_counter()
        length = (len(Class_two_urls)*(len(Class_one_urls) > len(Class_two_urls)) +
                 len(Class_one_urls)*(len(Class_one_urls) < len(Class_two_urls)))
        for i, j in enumerate(np.random.randint(0, length, nums)):
            urllib.request.urlretrieve(Class_one_urls[j], f'{address}/{self.classes[0]}{i}.jpg')
            urllib.request.urlretrieve(Class_two_urls[j], f'{address}/{self.classes[1]}{i}.jpg')
        time2=time.perf_counter()
        print('time taken : ', time2-time1)

    def learning(self, address, architecture, tune:int = 5, test_split = 0.2, lrfind = True):
        datblk = DataBlock(
                    blocks=(ImageBlock, CategoryBlock), 
                    get_items=get_image_files, 
                    splitter=TrainTestSplitter(test_size=test_split, random_state=42, shuffle=True),
                    get_y=parent_label,
                    item_tfms=Resize(self.width),
                    batch_tfms=aug_transforms())

        datlds=datblk.dataloaders(address)
        Classifier = cnn_learner(datlds, architecture, metrics=accuracy)
        
        print("Finding the suitable learning rate....")
        if lrfind == True:
            try:
                lr = Classifier.lr_find()
            except:
                lr = [5e-4,0]
        else:
            lr = [5e-4,0]
        print(f"lr : {lr[0]}")
        
        Classifier.recorder.train_metrics=True
        callbacks = [EarlyStoppingCallback(patience=7), SaveModelCallback(monitor='error_rate', fname = "model")]
        Classifier.callbacks = callbacks
        print("Training....")
        Classifier.fine_tune(tune, base_lr = lr[0])
        self.classifier = Classifier

        return Classifier

    def predict(self, image_address):
        return self.classifier.predict(image_address)

    def export_model(self, address):
        self.classifier.export(f'{address}/{self.classifier_name}_Classifier_34.pkl')
        
    def import_model(self, address):
        return load_learner(address)
    
    def test_model(self, test_x_address, test_y, Classifier):
        test_x = os.listdir(test_x_address)
        preds=[]
        with Classifier.no_bar():
            for i in test_x:
                if Classifier.predict(f"{test_x_address}/{i}")[2][0] >=0.5:
                    preds.append(0)
                else:
                    preds.append(1)
        print(classification_report(np.array(test_y), np.array(preds),
                                    target_names=self.classes))

if __name__ == '__main__':
    dat_add = '../input/starspiralelliptical'
    SG = Main_Classifier('Star_Galaxy')
    SE = Main_Classifier('Spiral_Elliptical')

    SG.download_imgs([0,10000], 'SG', dat_add)
    SE.download_imgs([0,10000], 'SE', dat_add)
    SG.test_imgs(500, 'test', dat_add)
    SE.test_imgs(500, 'test', dat_add)

    SGC = SG.learning('SG/Data_100', resnet34, tune = 5, lrfind = True)
    SEC = SE.learning('SE/Data_100', resnet34, tune = 5, lrfind = True)

    img = './test/Elliptical3.jpg'
    SGR = SG.predict(img)
    SER = SE.predict(img)

    Star = SGR[1].numpy()*1*'Star'
    Spiral = (not SGR[1].numpy())*SER[1].numpy()*'Spiral Galaxy'
    Elliptical = (not SGR[1].numpy())*(not SER[1].numpy())*'Elliptical Galaxy'
    Result = Star + Spiral + Elliptical
    
    print(f'The Image is a {Result}')
	