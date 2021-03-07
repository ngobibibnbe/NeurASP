# **********************methode 1 entity and relation
# etape:2 this part consist of extracting fact and needed tensors **** we are using neurASP here 
# ajouter les methodes avec le no qui enlève les relations redondantes

import numpy as np   
import cv2
import matplotlib.pyplot as plt
import os 
from matplotlib import *
import torchvision
import sys
sys.path.append('../../')
import time
import torch
from network import Net
from neurasp import NeurASP
import pandas as pd
from torchvision import datasets, transforms
import helper
import cv2
import glob
import json

    
dprogram = r'''
nn(nerjoinre(1,R,IdR), [ work_For, kill, orgBased_In, live_In, located_In]) :- relation(R,IdR,IdE1,IdE2).
nn(nerjoinre(1,E,IdE), [loc, org, peop, other]) :- entity(E,IdE,Value).

'''
aspProgram = r'''

% an entity can't have two name assignement 
 test(Value, Named) :- nerjoinre(0,E,IdE,Named) , entity(E,IdE,Value).
 test_1(Value) :- entity(E,IdE,Value).
 count_value(Value, S) :- S=#count{E : test(Value, E)} ,test_1(Value).
 :- count_value(Value, S), S>1.

% Each entity pair can be assign a relation once. 

% entity_pair_relation(Value1, Value2,R1, Named_Relation):-nerjoinre(0,R,R1,Named_Relation), relation(R,R1,E1,E2),entity(E,E1,Value1), entity(E,E2,Value2).
% entity_pair(Value1, Value2, Named_Relation) :- nerjoinre(0,R,R1,Named_Relation), relation(R,R1,E1,E2), entity(E,E1,Value1), entity(E,E2,Value2).
% test_pair(Value1, Value2, Named_Relation, S):- S=#count{R1 : entity_pair_relation(Value1, Value2,R1, Named_Relation)}, entity_pair(Value1, Value2, Named_Relation).
% :- test_pair(Value1, Value2, Named_Relation, S), S>1.

% the type assignment to each entity should be consistent with it's neighbour relation

group(IdR,R1,E1,E2):- relation(R,IdR,IdE1,IdE2),nerjoinre(0,E,IdE1,E1), nerjoinre(0,E,IdE2,E2),nerjoinre(0,R,IdR,R1).
:- relation(R,IdR,_,_), not group(IdR,work_For,peop,org), not group(IdR,kill,peop,peop), not group(IdR,orgBased_In,org,loc), not group(IdR,live_In,peop,loc), not group(IdR,located_In,loc,loc) .

% :- relation(R,IdR,_,_),entity(e,IdE1,_,E1), entity(e,IdE2,_,E2), not group(IdR,no,E1,E2), not group(IdR,work_For,peop,org), not group(IdR,kill,peop,peop), not group(IdR,orgBased_In,org,loc), not group(IdR,live_In,peop,loc), not group(IdR,located_In,loc,loc) .
'''
m = Net()    
nnMapping = {'nerjoinre': m}  
def smooth_Neur (data_JSON,file_result):
    smoothE=0.
    smooth =0.
    for doc_id , document in enumerate(data_JSON):
        #document =data_JSON[1]
        Factlist=""
        DataList={}
        entities=[]
        entities_values={}
        replacement={}
        entities_with_all=document["entities"]

        for idx, entity_brut in enumerate(entities_with_all):
            entity_words = ('_'.join(document["tokens"][entity_brut["start"]:entity_brut["end"]]))
            entity_probs = entity_brut["probs"].split("[")[1].split("]")[0]
            entity_type = str(entity_brut["type"])
            #print("***",str(doc_id)+"-"+str(idx),entity_words,entity_probs)
            Factlist=Factlist+"entity(e,e"+str(idx)+",'b_"+(''.join(e for e in entity_words if e.isalnum())).lower()+"').\n "
            #print(type(entity_probs))
            probs =np.fromstring(entity_probs, dtype=float, sep=' ')[1:]
            DataList['e,e'+str(idx)]=torch.tensor([probs], dtype=torch.float64)
            if (smoothE!=0.):
                probs[probs == 0.0] = smoothE   
            
        relations_with_all=document["relations"]
        for idx ,relation_brut in enumerate(relations_with_all):
            relation_probs = relation_brut["probs"].split("[")[1].split("]")[0]
            Factlist=Factlist+"relation(r,r"+str(idx)+",e"+str(relation_brut["head"])+",e"+str(relation_brut["tail"])+").\n "
            probs= np.fromstring(relation_probs, dtype=float, sep=' ')
            #probs =np.append(probs, [0.1])
            if (smooth!=0.):
                probs[probs == 0.] = smooth
            DataList['r,r'+str(idx)] = torch.tensor([probs], dtype=torch.float64)
        Factlist=[Factlist]
        DataList=[DataList]

        for idx, facts in enumerate(Factlist):
            NeurASPobj = NeurASP(dprogram + facts, nnMapping, optimizers=None)
            try :
                models = NeurASPobj.infer(dataDic=DataList[idx], obs='', mvpp=aspProgram + facts)
                #print(models[0])
                bool = True
            except Exception as e:
                bool=False
                print (str(doc_id),"error*****************************",e)
        if bool :
            for model in models[0]:
                for atom in model:
                    if 'nerjoinre(0,e,e' in atom:
                        entity_id= atom.split("nerjoinre(0,e,e")[1].split(",")[0]
                        #print("ll",len(data_JSON[doc_id]["entities"]),len(entities_with_all))
                        data_JSON[doc_id]["entities"][int(entity_id)]["type"]= atom.split(",")[3].split(")")[0][:1].upper() + atom.split(",")[3].split(")")[0][1:]
                    if 'no)' in atom and 'nerjoinre(0,r,r' in atom:
                        print("****************************** we are removing a no relation; it's :",int(relation_id))
                        relation_id= atom.split("nerjoinre(0,r,r")[1].split(",")[0]
                        #del (data_JSON[doc_id]["relations"])[int(relation_id)]
                    
                    elif 'nerjoinre(0,r,r' in atom:
                        relation_id= atom.split("nerjoinre(0,r,r")[1].split(",")[0]
                        data_JSON[doc_id]["relations"][int(relation_id)]["type"]= atom.split(",")[3].split(")")[0].upper()[:1] + atom.split(",")[3].split(")")[0][1:]
    with open(file_result, 'w') as f:
        json.dump(data_JSON, f)
        print("*********** You can actually test your model with evaluation functions")
  
    

    


FEdprogram = r'''
nn(nerjoinre(1,R,IdR), [ work_For, kill, orgBased_In, live_In, located_In, no]) :- relation(R,IdR,IdE1,IdE2).
% because entities are fixed we can have entities fixed as entities can be find badly by the NN, in this case 
% because we don't want to penalize other relation identified in case on relation does'nt satisfy a constraint 

'''
FEaspProgram = r'''

% an entity pair can't have the same relation twice
entity_pair_relation(Value1, Value2,R1, Named_Relation):-nerjoinre(0,R,R1,Named_Relation), relation(R,R1,E1,E2),entity(E,E1,Value1,T_E1), entity(E,E2,Value2,T_E2).
entity_pair(Value1, Value2, Named_Relation) :- nerjoinre(0,R,R1,Named_Relation), relation(R,R1,E1,E2),entity(E,E1,Value1,T_E1), entity(E,E2,Value2,T_E2).
test_pair(Value1, Value2, Named_Relation, S):- S=#count{R1 : entity_pair_relation(Value1, Value2,R1, Named_Relation)}, entity_pair(Value1, Value2, Named_Relation).
:- test_pair(Value1, Value2, Named_Relation, S), S>1, not test_pair(Value1, Value2,no,S) .


% the type assignment to each entity should be consistent with it's neighbour relation
group(IdR,R1,E1,E2):- relation(R,IdR,IdE1,IdE2),entity(e,IdE1,_,E1), entity(e,IdE2,_,E2),nerjoinre(0,R,IdR,R1).
:- relation(R,IdR,IdE1,IdE2), entity(e,IdE1,_,E1), entity(e,IdE2,_,E2), not group(IdR,no,E1,E2), not group(IdR,work_For,peop,org), not group(IdR,kill,peop,peop), not group(IdR,orgBased_In,org,loc), not group(IdR,live_In,peop,loc), not group(IdR,located_In,loc,loc) .
'''

m = Net()    
nnMapping = {'nerjoinre': m}  

def NeurFE (data_JSON,file_result,sup=False):
    
    for doc_id , document in enumerate(data_JSON):
        #document =data_JSON[1]
        Factlist=""
        DataList={}
        entities=[]
        entities_with_all=document["entities"]

        for idx, entity_brut in enumerate(entities_with_all):
            entity_words = ('_'.join(document["tokens"][entity_brut["start"]:entity_brut["end"]]))
            entity_probs = entity_brut["probs"].split("[")[1].split("]")[0]
            entity_type = str(entity_brut["type"])
            #print("***",str(doc_id)+"-"+str(idx),entity_words,entity_probs)
            Factlist=Factlist+"entity(e,e"+str(idx)+",'b_"+(''.join(e for e in entity_words if e.isalnum())).lower()+"',"+entity_type.lower()+").\n "
            #print(type(entity_probs))
            probs =np.fromstring(entity_probs, dtype=float, sep=' ')[1:]
            
            #**********to be back *********DataList['e,e'+str(idx)]=torch.tensor([probs], dtype=torch.float64)

        # relation(r,r1,e1,e2)ZZ
        relations_with_all=document["relations"]
        for idx ,relation_brut in enumerate(relations_with_all):
            relation_probs = relation_brut["probs"].split("[")[1].split("]")[0]
            Factlist=Factlist+"relation(r,r"+str(idx)+",e"+str(relation_brut["head"])+",e"+str(relation_brut["tail"])+").\n "
            probs= np.fromstring(relation_probs, dtype=float, sep=' ')
            probs =np.append(probs, [probs.min()])
            
            DataList['r,r'+str(idx)] = torch.tensor([probs], dtype=torch.float64)
        Factlist=[Factlist]
        DataList=[DataList]
        #print("datalist***",DataList)

        for idx, facts in enumerate(Factlist):
            # Initialize NeurASP object
            NeurASPobj = NeurASP(FEdprogram + facts, nnMapping, optimizers=None)
            # Find the most probable stable model
            models = NeurASPobj.infer(dataDic=DataList[idx], obs='', mvpp=FEaspProgram + facts)
        bool=True
        if bool :
            for model in models[0]:
                for atom in model:
                    if 'nerjoinre(0,r,r' in atom and 'no' not in atom:
                        relation_id= atom.split("nerjoinre(0,r,r")[1].split(",")[0]
                        data_JSON[doc_id]["relations"][int(relation_id)]["type"]= atom.split(",")[3].split(")")[0].upper()[:1] + atom.split(",")[3].split(")")[0][1:]
                    if sup :
                        if 'no)' in atom and 'nerjoinre(0,r,r' in atom:
                            print("****************************** we are removing a no relation; it's :",int(relation_id))
                            relation_id= atom.split("nerjoinre(0,r,r")[1].split(",")[0]
                            del (data_JSON[doc_id]["relations"])[int(relation_id)]
                            #data_JSON[doc_id]["relations"].remove(int(relation_id))
                    
    #print(data_JSON[:2])
    with open(file_result, 'w') as f:
        json.dump(data_JSON, f)
        print("*********** You can actually test your model with evaluation functions")

thresholds=[0.1,0.2,0.3,0.4,0.5]
print("logs\spert_treshold_"+str(threshold)+"/*.json")
for threshold in thresholds:
    files =glob.glob("logs\spert_treshold_"+str(threshold)+"\predictions*.json")
    for file_ in files :
        with open(file_) as file :
            data_JSON = json.load(file)
        file_result=file_.split("\\")[0]
        file_result+="\\"+file_.split("\\")[1]
        file_name=file_.split('\\')[-1]
        file_result=file_result+"\\neurASP_"+file_name
        smooth_Neur(data_JSON,file_result)


thresholds=[0.1,0.2,0.3,0.4,0.5]
#print("logs\spert_treshold_"+str(threshold)+"/*.json")
for threshold in thresholds:
    files =glob.glob("logs\spert_treshold_"+str(threshold)+"\predictions*.json")
    for file_ in files :
        with open(file_) as file :
            data_JSON = json.load(file)
        file_result=file_.split("\\")[0]
        file_result+="\\"+file_.split("\\")[1]
        file_name=file_.split('\\')[-1]
        file_result1=file_result+"\\neurASP-FE_"+file_name
        NeurFE(data_JSON,file_result1)
        file_result2=file_result+"\\neurASP-FE-SUP_"+file_name
        NeurFE(data_JSON,file_result2,sup=True)
        
