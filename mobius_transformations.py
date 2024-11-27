import pandas as pd
import numpy as np
from numpy import *
import random
from random import random
import re

class MobiusTransform:
    def __init__(self, df):
        self.df = df

    def extract_key(self, col_name):

        match = re.match(r'(.+)_([XY])$', col_name)
        if match:
            landmark_name = match.group(1)
            coord = match.group(2)
            coord_order = 0 if coord == 'X' else 1  
            return (landmark_name, coord_order)
        else:

            return (float('inf'), col_name)


    def testM(self, a, b, c, d):     
        M = 1.5 # M must be > 1, the higher it is the more it distorts the image

        v1 = np.absolute(a) ** 2 / np.absolute(a*d - b*c)
        if not (v1 < M and v1 > 1/M):
            return False

        v2 = np.absolute(a-1*c) ** 2 / (np.absolute(a*d -b*c))
        if not (v2 < M and v2 > 1/M):
            return False

        v3 = np.absolute(complex(a,-1*c)) ** 2 / np.absolute(a*d-b*c)
        if not (v3 < M and v3 > 1/M):
            return False

        v4 = np.absolute(complex(a-1*c,-1*c)) ** 2 / np.absolute(a*d-b*c)
        if not (v4 < M and v4 > 1/M):
            return False

        v5 = np.absolute(complex(a-1*c,-1*c)) ** 2 / (np.absolute(a*d-b*c))
        if not (v5 < M and v5 > 1/M):
            return False
            
        v6 = real(complex(1-b,1*d)/complex(a-1*c,-1*c))
        if not( v6 > 0 and v6 < 1):
            return False


        return True
            
    def getabcd(self, height, width):
        test = False

        while not test:
            zp=[complex(height*random(),width*random()), complex(height*random(),width*random()),complex(height*random(),width*random())] 
            wa=[complex(height*random(),width*random()), complex(height*random(),width*random()),complex(height*random(),width*random())]
            # transformation parameters
            a = linalg.det([[zp[0]*wa[0], wa[0], 1], 
                            [zp[1]*wa[1], wa[1], 1], 
                            [zp[2]*wa[2], wa[2], 1]]);

            b = linalg.det([[zp[0]*wa[0], zp[0], wa[0]], 
                            [zp[1]*wa[1], zp[1], wa[1]], 
                            [zp[2]*wa[2], zp[2], wa[2]]]);         


            c = linalg.det([[zp[0], wa[0], 1], 
                            [zp[1], wa[1], 1], 
                            [zp[2], wa[2], 1]]);

            d = linalg.det([[zp[0]*wa[0], zp[0], 1], 
                            [zp[1]*wa[1], zp[1], 1], 
                            [zp[2]*wa[2], zp[2], 1]]);
            test = self.testM(a, c, b, d)

        return a, b, c, d
        
    def mobius_transformation(self, x, y):
        test = False

        while not test:
            a, b, c, d = self.getabcd(1, 1)
            z = x + 1j * y
            w = (a * z + b) / (c * z + d)

            test = self.check_minmax(real(w), imag(w))

        return real(w), imag(w)

    def check_minmax(self, x, y):
        print('out of range, trying again')
        if max(x) > 1 or min(x) < 0 or max(y) > 1 or min(y) < 0: # check if any values are not in the range (0, 1)
            return False
        else:
            return True

    def transform(self):
        columns = self.df.columns.tolist() # copy column names
        sorted_columns = sorted(columns, key=self.extract_key) 
        self.df = self.df.reindex(columns=sorted_columns)
        x_columns = self.df.iloc[:, ::2]
        y_columns = self.df.iloc[:, 1::2]
        mask = (x_columns.values != 0) | (y_columns.values != 0)
        augmented_x = x_columns.values.copy()
        augmented_y = y_columns.values.copy()
        augmented_x[mask], augmented_y[mask] = self.mobius_transformation(x_columns.values[mask], y_columns.values[mask])
        new_df = np.empty_like(self.df)
        new_df[:, ::2] = augmented_x
        new_df[:, 1::2] = augmented_y
        new_df = np.around(new_df, decimals=6)
        transformed_df = pd.DataFrame(new_df, columns=self.df.columns)
        return transformed_df




































