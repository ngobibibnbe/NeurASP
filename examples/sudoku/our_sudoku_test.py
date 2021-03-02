import sys
sys.path.append('../../')

import torch

from dataGen import dataListTest, obsListTest, test_loader
from neurasp import NeurASP
from network import Sudoku_Net

######################################
# The NeurASP program can be written in the scope of ''' Rules '''
######################################

dprogramnn = '''
% neural rule
nn(identify(81, img), [empty,1,2,3,4,5,6,7,8,9]).
'''

########
# Define nnMapping and initialze NeurASP object
########

m = Sudoku_Net()
nnMapping = {'identify': m}

########
# Load pre-trained model and start testing
########
folder="data"
numOfData = [15, 17, 19, 21, 23,25]


#testing ACC_identify of a neural network with sudoku\r 


dprogram1 = '''
    % we assign one number at each position (R,C)
    a(R,C,N) :- identify(Pos, img, N), R=Pos/9, C=Pos\9, N!=empty.
{a(R,C,N): N=1..9}=1 :- identify(Pos, img, empty), R=Pos/9, C=Pos\9.

% it's a mistake if the same number shows 2 times in a row
:- a(R,C1,N), a(R,C2,N), C1!=C2.

% it's a mistake if the same number shows 2 times in a column
:- a(R1,C,N), a(R2,C,N), R1!=R2.

% it's a mistake if the same number shows 2 times in a 3*3 grid
:- a(R,C,N), a(R1,C1,N), R!=R1, C!=C1, ((R/3)*3 + C/3) = ((R1/3)*3 + C1/3).

    '''

dprogram2='''a(R,C,N) :- identify(Pos, img, N), R=Pos/9, C=Pos\9, N!=empty.

% it's a mistake if the same number shows 2 times in a row
:- a(R,C1,N), a(R,C2,N), C1!=C2.

% it's a mistake if the same number shows 2 times in a column
:- a(R1,C,N), a(R2,C,N), R1!=R2.

% it's a mistake if the same number shows 2 times in a 3*3 grid
:- a(R,C,N), a(R1,C1,N), R!=R1, C!=C1, ((R/3)*3 + C/3) = ((R1/3)*3 + C1/3).'''
 
dprogram3 = '''
    a(R,C,N) :- identify(Pos, img, N), R=Pos/9, C=Pos\9.
    '''
all_dprogram = {"NN + NeurASP":dprogram1,"NN +NeurASP without r":dprogram2,"NN":dprogram3}

numOfData = [15, 17, 19, 21, 23,25]

def testAccidentity ():
   
    for num in numOfData:
        print('\nLoad the model trained with {} data'.format(num))
        m.load_state_dict(torch.load(folder+'/model_data{}.pt'.format(num), map_location='cpu'))
        NeurASPobj = NeurASP(dprogramnn, nnMapping, optimizers=None,gpu=False)
        for key, dprogram in all_dprogram.items() :
            print("############# Method "+key+" on "+str(num)+" training data")
            prediction={}
            correct=0
            total=0
            for dataIdx, data in enumerate(dataListTest):
                model,_ = NeurASPobj.infer(dataDic=data, mvpp=dprogram)
                real={}
                prediction={}
                for atom in model[0]:
                    if 'identify(' in atom :
                        pos=int(atom.split("(")[1].split(",")[0])#*9 + int(atom.split("(")[1].split(",")[1])
                        pred=atom.split("(")[1].split(",")[2].split(")")[0]
                        prediction[str(pos)]=str(pred)
                flag =True 
                for identify in obsListTest[dataIdx].split("\n")[:-1]:
                    real[str(identify.split("(")[1].split(",")[0])] =(identify.split("(")[1].split(",")[2].split(")")[0])
                total+=1
                for id in prediction:
                    if prediction[id]!=real[id].replace(' ',''):
                        flag= False
                        break
                if flag==True :
                    #print ("super : ",str(correct) , "raté :",str(total-correct))
                    correct+=1
            print("acc of Method "+key+" on "+str(num)+" training data",'{:0.2f}%'.format(100*(correct/total)))
        
print(testAccidentity())