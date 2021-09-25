import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import math
class Lab4(object):
    
    def expectation_maximization(self,read_mapping,tr_lengths,n_iterations) :
        #start code here
        abundance = np.empty((len(tr_lengths), n_iterations + 1))
        z = np.empty((len(read_mapping), len(tr_lengths)))
        for i in range(abundance.shape[0]):
            abundance[i][0] = 1/len(tr_lengths)
        for itr in range(n_iterations):
            for a in range (len(read_mapping)):
                val = 0
                for transcripts in read_mapping[a]:
                    val = val + abundance[transcripts][itr]
                for i in range(len(tr_lengths)):
                    if i in read_mapping[a]:
                        z[a][i] = abundance[i][itr] / val
                    else:
                        z[a][i] = 0
            result = z.sum(axis = 0)
            result = result / len(read_mapping)
            val = 0
            for i in range(len(result)):
                val = val + (result[i]/ tr_lengths[i])
            for i in range(len(result)):
                abundance[i][itr+1] = (result[i]/tr_lengths[i])/(val)

        return abundance

        pass
        #end code here

    def prepare_data(self,lines_genes) :
        '''
        Input - list of strings where each string corresponds to expression levels of a gene across 3005 cells
        Output - gene expression dataframe
        '''
        #start code here
        data = {}
        for i in range (len(lines_genes)):
            arr = lines_genes[i].split()
            arr1 = []
            name = "Gene_" + str(i)
            for j in range(len(arr)):
                arr1.append(round(math.log(1 + int(arr[j])),5))
            data[name] = arr1
        df = pd.DataFrame(data)
        return df
        #end code here
    
    def identify_less_expressive_genes(self,df) :
        '''
        Input - gene expression dataframe
        Output - list of column names which are expressed in less than 25 cells
        '''
        #start code here
        output = []
        for column in df:
            ser = pd.Series(df[column])
            result = ser.to_numpy().nonzero()
            if len(result[0]) < 25:
                output.append(column)
        return output
        #end code here
    
    
    def perform_pca(self,df) :
        '''
        Input - df_new
        Output - numpy array containing the top 50 principal components of the data.
        '''
        #start code here
        data = df.values
        pca = PCA(n_components = 50, random_state=365)
        pca_result = pca.fit_transform(data)
        pca_result = np.round(pca_result, decimals = 5)
        return pca_result


        #end code here

    
    def perform_tsne(self,pca_data) :
        '''
        Input - pca_data
        Output - numpy array containing the top 2 tsne components of the data.
        '''
        #start code here
        
        #end code here