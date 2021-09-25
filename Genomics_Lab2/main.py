import numpy as np
from collections import OrderedDict

class Lab2(object):
    
    def smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - an integer value which is the maximum smith waterman alignment score
        '''
        #start code here
        match = penalties['match']
        mismatch = penalties ['mismatch']
        gap = penalties ['gap']
        '''
        print(len(s1))
        print(len(s2))
        print(arr.shape)
        '''
        
        arr = np.zeros((len(s1) + 1, len(s2) + 1))
        for i in range (1, len(s1) + 1):
            for j in range (1, len(s2) + 1):
                if s2[j-1] == s1[i-1]:
                    val1 = arr[i-1][j-1] + match
                else:
                    val1 = arr[i-1][j-1] + mismatch
                node2 = arr[i-1][j] + gap
                node3 = arr[i][j-1] + gap
                arr[i][j] = max(val1, node2, node3, 0)

        maximum = 0
        for i in range (1, len(s1) + 1):
            for j in range (1, len(s2) + 1):
                if arr[i][j] > maximum:
                    maximum = arr[i][j]
        return int(maximum)
        #end code here

    def print_smith_waterman_alignment(self,s1,s2,penalties) :
        '''
        Input - two sequences and a dictionary with penalities for match, mismatch and gap
        Output - a tuple with two strings showing the two sequences with '-' representing the gaps
        '''
        #start code here
        match = penalties['match']
        mismatch = penalties ['mismatch']
        gap = penalties ['gap']

        arr = np.zeros((len(s1) + 1, len(s2) + 1))
        for i in range (1, len(s1) + 1):
            for j in range (1, len(s2) + 1):
                if s2[j-1] == s1[i-1]:
                    val1 = arr[i-1][j-1] + match
                else:
                    val1 = arr[i-1][j-1] + mismatch
                node2 = arr[i-1][j] + gap
                node3 = arr[i][j-1] + gap
                arr[i][j] = max(val1, node2, node3, 0)
        maximum = 0
        for i in range (1, len(s1) + 1):
            for j in range (1, len(s2) + 1):
                if arr[i][j] > maximum:
                    maximum = arr[i][j]
                    m = i
                    n = j
        itr = arr[m][n]
        str1 = ""
        str2 = ""
        while itr != 0:
            if s1[m-1] == s2[n-1]:
                str1 = str1 + s1[m-1]
                str2 = str2 + s2[n-1]
                m = m - 1
                n = n - 1
                itr = arr[m][n]
            else:
                if arr[m-1][n] > arr[m][n-1]:
                    str1 = str1 + s1[m-1]
                    str2 = str2 + "-"
                    m = m - 1
                    itr = arr[m][n]
                else:
                    str1 = str1 + "-"
                    str2 = str2 + s2[n-1]
                    n = n - 1
                    itr = arr[m][n]

        return (str1[::-1], str2[::-1])
        #end code here

    def find_exact_matches(self,list_of_reads,genome):
        
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output - a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome where the ith read appears. The starting positions should be specified using the "chr2:120000" format
        '''
        
        #start code here
        d = {}
        
        flag = 0
        for i in range(len(genome)):
            if i == len(genome) - 1:
                continue
            elif genome[i] == '>':
                v = genome[i+4]
            elif genome[i] ==  '\n' and genome[i+1] != '>':
                b = i + 1
                if b+len(list_of_reads[0]) - 1 >= len(genome) - 1:
                    break
                while genome[b+len(list_of_reads[0]) - 1] != '>':
                    str1 = ""
                    for a in range(len(list_of_reads[0])):
                        if genome[a + b] == '\n':
                            str1 = str1 + genome[a + b]
                        str1 = str1 + genome[b + a]
                    if str1 in d:
                        d[str1].append('chr:' + v + ':' + str(b - i ))
                    else:
                        d[str1] = ['chr:' + v + ':' + str(b - i)]
                    b = b + 1
 
            

        
        '''
        for line in genome.splitlines():
            if line == ">chr1":
                v = '1'
            elif line == ">chr2":
                v = '2'
            elif line == ">chr3":
                v = '3'
            else:
                for i in range(len(line)-len(list_of_reads)):
                    str1 = ""
                    for j in range (len(list_of_reads[0])):
                        str1 = str1 + line[i + j]
                    if str1 in d:
                        d[str1].append('chr:' + v + ':' + str(i+1))
                    else:
                        d[str1] = ['chr:' + v + ':' + str(i+1)]
        '''
        a = []
        for i in range(len(list_of_reads)):
            if list_of_reads[i] in d:
                a.append(d[list_of_reads[i]])
            else:
                a.append([])
        return a
        #end code here
       
    
    def find_approximate_matches(self,list_of_reads,genome):
        '''
        Input - list of reads of the same length and a genome fasta file (converted into a single string)
        Output -  a list with the same length as list_of_reads, where the ith element is a list of all locations (starting positions) in the genome which have the highest smith waterman alignment score with ith read in list_of_reads
        '''
        
        #start code here
        #end code here
        