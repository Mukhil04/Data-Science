import pandas as pd
import statsmodels.api as sm 
import numpy as np
import statsmodels

class Lab3(object):
    
    def create_data(self,snp_lines) :
        '''
        Input - the snp_lines parsed at the beginning of the notebook
        Output - You should return the 53 x 3902 dataframe
        '''
        #start code here
        dic = {}
        for line in snp_lines:
            i = 0
            str1 = ''
            while line[i] != '\t':
                str1 = str1 + line[i]
                i = i + 1
            str1 = str1 + ':'
            i = i + 1
            while line[i] != '\t':
                str1 = str1 + line[i]
                i = i + 1
            i = i + 1
            count = 7
            while count != 0:
                i = i + 1
                if line[i] == '\t':
                    count = count - 1
            i = i + 1
            val = 0
            values = []
            while line[i-1] != '\n':
                val = 0
                if line[i] == '.':
                    values.append(np.nan)
                else:
                    val = float(line[i]) + float(line[i + 2])
                    values.append(val)
                i = i + 4
            dic[str1] = values
        df = pd.DataFrame(dic)
        return df
        #end code here

    def create_target(self,header_line) :
        '''
        Input - the header_line parsed at the beginning of the notebook
        Output - a list of values(either 0 or 1)
        '''
        #start code here
        index = []
        flag = 0
        str1 = ''
        for i in range(len(header_line)):
            if (header_line[i] == '\t' or header_line[i] == '\n') and len(str1) != 0:
                if str1[0] == 'd':
                    index.append(0)
                elif str1[0] == 'y':
                    index.append(1)
                flag = 0
                str1 = ''
            if flag == 1:
                str1 = str1 + header_line[i]
            if flag == 0 and header_line[i] == 'd':
                str1 = str1 + 'd'
                flag = 1
            elif flag == 0 and header_line[i] == 'y':
                str1 = str1 + 'y'
                flag = 1
        return index
        #end code here
    
    def logistic_reg_per_snp(self,df) :
        '''
        Input - snp_data dataframe
        Output - list of pvalues and list of betavalues
        '''
        #start code here
        output = df["target"].tolist()
        pvalues = []
        beta = []
        for (columnName, columnData) in df.iteritems():
            if columnName == 'target':
                continue
            result = sm.Logit(output, sm.add_constant(columnData.values), missing = "drop").fit(method = 'bfgs', disp = False)
            pvalues.append(round(result.pvalues[1], 9))
            beta.append(round(result.params[1], 5))
        return (pvalues, beta)
        #end code here
    
    
    def get_top_snps(self,snp_data,p_values) :
        '''
        Input - snp dataframe with target column and p_values calculated previously
        Output - list of 5 tuples, each with chromosome and position
        '''
        #start code here
        output = []
        for a in range(5):
            val = min(p_values)
            index = p_values.index(val)
            str1 = snp_data.columns[index]
            p_values[index] = 100
            chromosome = ''
            position = ''
            i = 0
            while str1[i] != ':':
                chromosome = chromosome + str1[i]
                i = i + 1
            i = i + 1
            while i != len(str1):
                position = position + str1[i]
                i = i + 1
            output.append((chromosome, position))

        return output
        #end code here