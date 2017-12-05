import os
import inspect
import random
import numpy as np
from sklearn.model_selection import train_test_split
from skimage.io import imread

class CharacterManager:
    def __init__(self, size=32):
        
        class_path = os.path.abspath(os.path.dirname(inspect.getfile(self.__class__)))
        self.folder = os.path.join(os.path.join(class_path, '..'),"data_" + str(size))
        image_names = self._get_file_names()
        random.shuffle(image_names)
        stroke_values = list(map(self.get_file_stroke_count, image_names))

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                image_names
                , stroke_values
                , test_size=0.1)
        
        self.training_indexes = np.random.permutation(len(self.X_train)).tolist()
        self.training_point = 0;

    def _get_file_names(self):
        return os.listdir(self.folder) 

    def training_batch(self, num=50):
        index = []
        while num > 0 :
            if self.training_point + num > len(self.training_indexes):
                index += self.training_indexes[self.training_point:]
                num -= len(self.training_indexes) - self.training_point
                self.training_indexes = np.random.permutation(len(self.X_train)).tolist() 
                self.training_point = 0
            else:
                index += self.training_indexes[self.training_point:self.training_point + num]
                self.training_point += num
                break;

        data = [self.import_photo(self.X_train[i]) for i in index]
        y_values = np.array([self.y_train[i] for i in index]).reshape(-1,1).tolist()
        return[data, y_values]

    def import_photo(self,file_name):
        image = imread(os.path.join(self.folder,file_name))
        return image.flatten()/np.max(image)

    def testing_data(self):
        data = list(map(self.import_photo, self.X_test))
        y_values = np.array(self.y_test).reshape(-1,1).tolist()
        return[data, y_values]

    def get_file_stroke_count(self, file_name):
        '''
        Returns the stroke count of the character depicted in the file.
        Used so that if the name format chages, don't have to change every function.

        Parameters
        ----------
        file_name : string
            A filename using standard file formating for this project.

        Returns
        -------
        int 
            The file's character's stroke count.
        '''
        return int(file_name.split(".")[0])
