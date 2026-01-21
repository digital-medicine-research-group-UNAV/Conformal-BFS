


import numpy as np
import pandas as pd
import multiprocessing
import sys

from ._utils_MrmrMS import beta_measures

from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.special import digamma
from sklearn.svm import  LinearSVC, SVC
from sklearn.multiclass import OneVsRestClassifier



def min_max_normalize_numpy(arr, a: int = 1, b: int = 2):
    if a >= b:
        raise ValueError("The 'a' parameter must be less than 'b' for min-max normalization.")
    
    min_val = arr.min()
    max_val = arr.max()
    
    if min_val == max_val:
        # Handle the case where all values in the array are the same
        return np.full_like(arr, a)
    
    normalized_arr = a + (b - a) * (arr - min_val) / (max_val - min_val)
    return normalized_arr


class numpy_InformationTheoryFS:

    def __init__(self, dataset: np.ndarray, parallel: bool = False):
        self.dataset = dataset
        self.n, self.p = dataset.shape
        self.class_column_index = self.p - 1
        self.columns = list(range(dataset.shape[1]))  # Assuming columns are numerical indices
        self.dataset_selected_features = [] 
        self.dataset_non_selected_columns = [] 
        self.parallel = parallel
         

    def getPairwiseDistArray(self, coords = [], discrete_dist = 1):

        '''
        Input: 
        data: pandas data frame
        coords: list of indices for variables to be used
        discrete_dist: distance to be used for non-numeric differences

        Output:
        p x n x n array with pairwise distances for each variable
        '''
    
    
        #n, p = self.dataset.shape # p: columns, n:rows
        if not coords:
            coords = range(self.p)

        distArray = np.empty((len(coords), self.n, self.n)) # np.full((len(coords), self.n, self.n), np.nan);  p -  n x n arrays initialized with NaNs
     
        for idx,coord in enumerate(coords):
 
            col_data = self.dataset[:, coord]  

            if np.issubdtype(self.dataset[:, coord].dtype, np.number):
                distArray[idx, :, :] = np.abs(col_data[:, None] - col_data)
            
            else:
                distArray[idx,:,:] = (col_data[:, None] != col_data).astype(float) * discrete_dist
    
        return distArray




    def countNeighbors(self, coord_dists, rho, coords = list()):
        '''
        input: list of coordinate distances (output of coordDistList), 
        coordinates we want (coords), distance (rho)

        output: scalar integer of number of points within ell infinity radius
        '''
    
        if not coords:
            coords = range(coord_dists.shape[1])
        dists = np.max(coord_dists[:,coords], axis = 1)
        count = np.count_nonzero(dists <= rho) - 1
        return count



    def cmiPoint(self, point_i, k, distArray):

        '''
        input:
        point_i: current observation row index
        x, y, z: list of indices
        k: positive integer scalar for k in knn
        distArray: output of getPairwiseDistArray

        output:
        cmi point estimate
        '''
        #n = distArray.shape[1] # num of rows
    
        coord_dists = np.transpose(distArray[[0,1,2], :, point_i])
    
        dists = np.max(coord_dists, axis=1)
        rho = np.partition(dists, k)[k]
        k_tilde = np.count_nonzero(dists <= rho) - 1
    
        xz_coords = [0, 2]
        yz_coords = [1, 2]
        z_coords = [2]
    
        nxz = self.countNeighbors(coord_dists, rho, xz_coords)
        nyz = self.countNeighbors(coord_dists, rho, yz_coords)
        nz = self.countNeighbors(coord_dists, rho, z_coords)
        xi = digamma(k_tilde) - digamma(nxz) - digamma(nyz) + digamma(nz)

        return xi


    ## NO BORRAR x,y AUNQUE NO SE USEN!
    def miPoint(self, point_i,x, y, k, distArray):
        '''
        input:
        point_i: current observation row index
        x, y: list of indices
        k: positive integer scalar for k in knn
        distArray: output of getPairwiseDistArray

        output:
        mi point estimate
        '''
        n = distArray.shape[1]
    
        coord_dists = np.transpose(distArray[[0,1], :, point_i])
    
        dists = np.max(coord_dists, axis=1)
        rho = np.partition(dists, k)[k]
        k_tilde = np.count_nonzero(dists <= rho) - 1

    
        x_coords = [0]
        y_coords = [1]

        nx = self.countNeighbors(coord_dists, rho, x_coords)
        ny = self.countNeighbors(coord_dists, rho, y_coords)
        xi = digamma(k_tilde) + digamma(n) - digamma(nx) - digamma(ny)
        return xi



    
    def cmi(self, x, y, z, k, discrete_dist = 1, minzero = 1):
        '''
        computes conditional mutual information, I(x,y|z)
        input:
        x: list of indices for x
        y: list of indices for y
        z: list of indices for z
        k: hyper parameter for kNN
        self.dataset: pandas dataframe

        output:
        scalar value of I(x,y|z)
        '''
        # compute CMI for I(x,y|z) using k-NN
        #n, p = self.dataset.shape

        # convert variable to index if not already
        vrbls = [x,y,z] # variable columns in a list
       
    
    
        for i, lst in enumerate(vrbls):
            if all(isinstance(elem, str) for elem in lst) and len(lst) > 0:
                vrbls[i] = [self.columns.index(col) for col in lst]
           
        x,y,z = vrbls

        

        distArray = self.getPairwiseDistArray( x + y + z, discrete_dist)
    

    
        if len(z) > 0:
            ptEsts = map(lambda obs: self.cmiPoint(obs, k, distArray), range(self.n))         
                
        else:
            ptEsts = map(lambda obs: self.miPoint(obs, x, y, k, distArray), range(self.n))

        if minzero == 1:
            return(max(sum(ptEsts)/self.n,0))
        elif minzero == 0:
            return(sum(ptEsts)/self.n)




    def relevance_func(self, column):
        return self.cmi([column], [self.class_column_index], [], 3)
                    
             
    def redundancy_func( self, column):
        return np.mean([self.cmi([ele], [column], [], 3) for ele in self.dataset_selected_features])

    
    def cond_redundancy_func(self, column): 
        return np.mean([self.cmi([ele], [column], [self.class_column_index], 3) for ele in self.dataset_selected_features])


    def cond_redundancy_3order_func(self, column):

        selected_columns = self.dataset_selected_features
        if len(selected_columns) == 1:
            return 0   
        
        else:
            suma = 0
            for ele in selected_columns:
                suma += np.mean([ self.cmi([column], [ele2], [ele], 3) for ele2 in selected_columns if ele2 != ele ])
                   
            return suma/len(selected_columns)
    

    def process_column_mRMR(self, column):
        relevance = self.relevance_func(column)
        redundancy = self.redundancy_func(column)
              
        return relevance - redundancy 

    def process_column_JMI(self, column):
        relevance = self.relevance_func(column)
        if not self.dataset_selected_features:
            return relevance
        
        redundancy = self.redundancy_func(column)
        cond_redundancy = self.cond_redundancy_func(column)

        return relevance - redundancy + cond_redundancy


        # The classic approach:
        #cond_sum = sum(self.cmi([column], [s], [self.class_column_index], 3)
        #           for s in self.dataset_selected_features)
        #return relevance + cond_sum

    def process_column_relax_mRMR(self, column):
        relevance = self.relevance_func(column)
        redundancy = self.redundancy_func(column)
        cond_redundancy = self.cond_redundancy_func(column)
        cond_redundancy_3o = self.cond_redundancy_3order_func(column)
        return relevance - redundancy + cond_redundancy - cond_redundancy_3o
    
    def divide_into_batches(self, batch_size):
        for i in range(0, len(self.dataset_non_selected_columns), batch_size):
            yield self.dataset_non_selected_columns[i:i + batch_size]
    

    def process_columns(self, method:str):
        """
        Process columns using the specified method. Allows parallel or sequential execution.
    
        :param method: The method to use for processing columns ('MIM', 'mRMR', 'JMI', 'relax_mRMR').
        :param parallel: Whether to process columns in parallel. Default is True.
        :return: A list of results from processing the columns.
        """

        self.dataset_non_selected_columns = [col for col in self.columns if col not in self.dataset_selected_features and col != self.class_column_index]
        
        method_map = {
            "MIM": self.relevance_func,
            "mRMR": self.process_column_mRMR,
            "JMI": self.process_column_JMI,
            "relax_mRMR": self.process_column_relax_mRMR
        }

        if method not in method_map:
            raise ValueError("No valid method selected")

        process_method = method_map[method]
        
        results = []  
        if self.parallel:
          
            batch_size = max(100, len(self.dataset_non_selected_columns) // (multiprocessing.cpu_count() * 2))
        
            batches_non_selected_features = self.divide_into_batches(batch_size)
        
            with ProcessPoolExecutor() as executor:
                for batch in batches_non_selected_features:
                    results.extend(list(executor.map(process_method, batch)))

        else:
            
            results = [process_method(column) for column in self.dataset_non_selected_columns]

        return results
    


class FeatureSelector:

    def __init__(self, classes_:list , lambda_: float = 0.5, max_features=-1, parallel: bool = False,  verbose: bool = False):
        self.max_features = max_features
        self.lambda_ = lambda_
        self.classes_ = classes_
        self.all_selected_features = []
        self.parallel = parallel
        self.verbose = verbose
        self.kernel: str 


    def mRMR_MS(self, X_tr ,Y_tr , kernel: str = "linear", split_size: float = 0.5, selected_features:list = []) -> None: 
        
        self.kernel = kernel

        dataset: np.ndarray = np.hstack((X_tr, Y_tr.reshape(-1,1)))  # The class column must be the last one always.

        objeto_numpy = numpy_InformationTheoryFS(dataset.copy(), self.parallel)

        objeto_numpy.dataset_selected_features = selected_features
        

        if self.max_features == -1:
            self.max_features = len(X_tr[0])

        while len(objeto_numpy.dataset_selected_features) < self.max_features:

            if self.verbose:
                #sys.stdout.write(f"\r{len(objeto_numpy.dataset_selected_features)} out of {self.max_features} features were selected")
                sys.stdout.flush()
            
            if len(objeto_numpy.dataset_selected_features) == 0:
                PHI = objeto_numpy.process_columns( method = "MIM")

            else:
                PHI = objeto_numpy.process_columns(method = "mRMR")
            
            Phi = min_max_normalize_numpy(np.array(PHI))  
                                                                                                
            Beta = beta_measures(X_tr[:, objeto_numpy.dataset_non_selected_columns],
                                Y_tr,
                                classes_ = self.classes_,
                                split_size =split_size,
                                kernel = self.kernel)
            
            Beta = min_max_normalize_numpy(np.array(Beta))
            
                    
            tg_alpha =  []
            for (phi,beta) in zip(Phi, Beta): 
                tg_alpha.append(phi/beta)
            
            alpha = list(np.arctan(tg_alpha))
            accepted_index = alpha.index(max(alpha)) # Indice en w del minimo. la posición del indice coincide con la del head
            

            selected_feature = objeto_numpy.dataset_non_selected_columns[accepted_index]

            objeto_numpy.dataset_selected_features.append(selected_feature)
        
            self.all_selected_features.append(objeto_numpy.dataset_selected_features[:])

        
    

    def mRMR(self, X_tr,Y_tr, selected_features:list = [])-> None:

        dataset: np.ndarray = np.hstack((X_tr, Y_tr.reshape(-1,1)))  # The class column must be the last one always.

        objeto_numpy = numpy_InformationTheoryFS(dataset.copy(), self.parallel)

        objeto_numpy.dataset_selected_features = selected_features


        if self.max_features == -1:
            self.max_features = len(X_tr[0])

        while len(objeto_numpy.dataset_selected_features) < self.max_features:

            if self.verbose:
                sys.stdout.write(f"\r{len(objeto_numpy.dataset_selected_features)} out of {self.max_features} features were selected")
                sys.stdout.flush()
            
            if len(objeto_numpy.dataset_selected_features) == 0:
                PHI = objeto_numpy.process_columns( method = "MIM")
            else:
                PHI = objeto_numpy.process_columns(method = "mRMR")

            accepted_index = PHI.index(max(PHI))
            selected_feature = objeto_numpy.dataset_non_selected_columns[accepted_index]

            objeto_numpy.dataset_selected_features.append(selected_feature)
        
            self.all_selected_features.append(objeto_numpy.dataset_selected_features[:])

            #print(f"PHI: {PHI}")
            #print(f"seleceted features: {objeto_numpy.dataset_selected_features}")
            #print(f"non selected features (NOT UPDATED): {objeto_numpy.dataset_non_selected_columns}")

        
    

    def JMI(self, X_tr,Y_tr, selected_features:list = [])-> None:

        dataset: np.ndarray = np.hstack((X_tr, Y_tr.reshape(-1,1)))  # The class column must be the last one always.

        objeto_numpy = numpy_InformationTheoryFS(dataset.copy(), self.parallel)

        objeto_numpy.dataset_selected_features = selected_features

        if self.max_features == -1:
            self.max_features = len(X_tr[0])

        while len(objeto_numpy.dataset_selected_features) < self.max_features:

            if self.verbose:
                sys.stdout.write(f"\r{len(objeto_numpy.dataset_selected_features)} out of {self.max_features} features were selected")
                sys.stdout.flush()
            
            if len(objeto_numpy.dataset_selected_features) == 0:
                PHI = objeto_numpy.process_columns( method = "MIM")
            else:
                PHI = objeto_numpy.process_columns(method = "JMI")

            accepted_index = PHI.index(max(PHI))
            selected_feature = objeto_numpy.dataset_non_selected_columns[accepted_index]

            objeto_numpy.dataset_selected_features.append(selected_feature)
        
            self.all_selected_features.append(objeto_numpy.dataset_selected_features[:])

            #print(f"PHI: {PHI}")
            #print(f"seleceted features: {objeto_numpy.dataset_selected_features}")
            #print(f"non selected features (NOT UPDATED): {objeto_numpy.dataset_non_selected_columns}")

        

    def relax_mRMR(self, X_tr,Y_tr, selected_features:list = [])-> None:

        dataset: np.ndarray  = np.hstack((X_tr, Y_tr.reshape(-1,1)))  # The class column must be the last one always.

        objeto_numpy = numpy_InformationTheoryFS(dataset.copy(), self.parallel)

        objeto_numpy.dataset_selected_features = selected_features
        

        if self.max_features == -1:
            self.max_features = len(X_tr[0])

        while len(objeto_numpy.dataset_selected_features) < self.max_features:

            if self.verbose:
                sys.stdout.write(f"\r{len(objeto_numpy.dataset_selected_features)} out of {self.max_features} features were selected")
                sys.stdout.flush()
            
            if len(objeto_numpy.dataset_selected_features) == 0:
                PHI = objeto_numpy.process_columns( method = "MIM")
            
            elif len(objeto_numpy.dataset_selected_features) == 1:
                PHI = objeto_numpy.process_columns( method = "JMI")

            else:
                PHI = objeto_numpy.process_columns( method = "relax_mRMR")


            accepted_index = PHI.index(max(PHI))
            selected_feature = objeto_numpy.dataset_non_selected_columns[accepted_index]

            objeto_numpy.dataset_selected_features.append(selected_feature)
        
            self.all_selected_features.append(objeto_numpy.dataset_selected_features[:])

            #print(f"PHI: {PHI}")
            #print(f"seleceted features: {objeto_numpy.dataset_selected_features}")
            #print(f"non selected features (NOT UPDATED): {objeto_numpy.dataset_non_selected_columns}")


        
