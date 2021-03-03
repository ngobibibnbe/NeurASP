
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
import unicodedata

#### method 3: we don't fix anything but we can have entity moving and we add a confidence to entities since it has a higher confidence in prediction

print("***********************************************METHOD3")

m = Net()    
nnMapping = {'nerjoinre': m}  

def NeurSUP (data_JSON,file_result,sup=False, sup_Entity=False):

    if sup_Entity :
        FEdprogram = r'''
        nn(nerjoinre(1,R,IdR), [ work_For, kill, orgBased_In, live_In, located_In,no]) :- relation(R,IdR,IdE1,IdE2).
        nn(nerjoinre(1,E,IdE), [no, loc, org, peop, other]) :- entity(E,IdE,Value).

        '''
        FEaspProgram = r'''
        % an entity can't have two name assignement 
         test(Value, Named) :- nerjoinre(0,E,IdE,Named) , entity(E,IdE,Value).
         test_1(Value) :- entity(E,IdE,Value).
         count_value(Value, S) :- S=#count{E : test(Value, E)} ,test_1(Value).
         :- count_value(Value, S), S>1 .

        % Each entity pair can be assign a relation once. 

         entity_pair_relation(Value1, Value2,R1, Named_Relation):-nerjoinre(0,R,R1,Named_Relation), relation(R,R1,E1,E2),entity(E,E1,Value1), entity(E,E2,Value2).
         entity_pair(Value1, Value2, Named_Relation) :- nerjoinre(0,R,R1,Named_Relation), relation(R,R1,E1,E2), entity(E,E1,Value1), entity(E,E2,Value2).
         test_pair(Value1, Value2, Named_Relation, S):- S=#count{R1 : entity_pair_relation(Value1, Value2,R1, Named_Relation)}, entity_pair(Value1, Value2, Named_Relation).
         :- test_pair(Value1, Value2, Named_Relation, S), S>1.
        
        % if a relation has an entity no, it should be assign no directly. 
        nerjoinre(0,R,IdR,no):-relation(R,IdR,IdE1,IdE2), entity(E,IdE1,_), nerjoinre(0,E,IdE1,no).
        nerjoinre(0,R,IdR,no):-relation(R,IdR,IdE1,IdE2), entity(E,IdE2,_), nerjoinre(0,E,IdE2,no).

        % the type assignment to each entity should be consistent with it's neighbour relation

        group(IdR,R1,E1,E2):- relation(R,IdR,IdE1,IdE2),nerjoinre(0,E,IdE1,E1), nerjoinre(0,E,IdE2,E2),nerjoinre(0,R,IdR,R1).
        :- relation(R,IdR,IdE1,IdE2),nerjoinre(0,E,IdE2,E2),nerjoinre(0,E,IdE1,E1), entity(e,IdE1,_), entity(e,IdE2,_), not group(IdR,no,E1,E2), 
        not group(IdR,work_For,peop,org), not group(IdR,kill,peop,peop), not group(IdR,orgBased_In,org,loc),
        not group(IdR,live_In,peop,loc), not group(IdR,located_In,loc,loc) .
        '''

    #i add relation(R,IdR,IdE1,IdE2), entity(e,IdE1,_,E1), entity(e,IdE2,_,E2), not group(IdR,no,E1,E2) from the previous test
 
    
    for doc_id , document in enumerate(data_JSON):
        #document =data_JSON[1]
        #doc_id=28
        Factlist=""
        DataList={}
        entities=[]
        entities_with_all=document["entities"]
        for idx, entity_brut in enumerate(entities_with_all):
            entity_words = ('_'.join(document["tokens"][entity_brut["start"]:entity_brut["end"]])).replace("ó","o")
            text = unicodedata.normalize('NFD', entity_words)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")
            entity_words = str(text)
            entity_probs = entity_brut["probs"].split("[")[1].split("]")[0]
            entity_type = str(entity_brut["type"])
            Factlist=Factlist+"entity(e,e"+str(idx)+",'b_"+(''.join(e for e in entity_words if e.isalnum())).lower()+"').\n "
            probs =1.2*np.fromstring(entity_probs, dtype=float, sep=' ')
            probs[0]=0.2
            print("********",probs)
            DataList['e,e'+str(idx)]=torch.tensor([probs], dtype=torch.float64)

            
            #**********to be back *********DataList['e,e'+str(idx)]=torch.tensor([probs], dtype=torch.float64)

        # relation(r,r1,e1,e2)ZZ
        relations_with_all=document["relations"]
        for idx ,relation_brut in enumerate(relations_with_all):
            relation_probs = relation_brut["probs"].split("[")[1].split("]")[0]
            Factlist=Factlist+"relation(r,r"+str(idx)+",e"+str(relation_brut["head"])+",e"+str(relation_brut["tail"])+").\n "
            probs= np.fromstring(relation_probs, dtype=float, sep=' ')
            probs =np.append(probs, [0.5])
            DataList['r,r'+str(idx)] = torch.tensor([probs], dtype=torch.float64)
        Factlist=[Factlist]
        DataList=[DataList]
        #print("datalist***",DataList)
        #print("factlist***",Factlist)

        for idx, facts in enumerate(Factlist):
            # Initialize NeurASP object
            
            NeurASPobj = NeurASP(FEdprogram + facts, nnMapping, optimizers=None)
            # Find the most probable stable model
            models = NeurASPobj.infer(dataDic=DataList[idx], obs='', mvpp=FEaspProgram + facts)
        bool=True
        if bool :
            for model in models[0]:
                for atom in model:
                    if 'nerjoinre(0,r,r' in atom:
                        relation_id= atom.split("nerjoinre(0,r,r")[1].split(",")[0]
                        data_JSON[doc_id]["relations"][int(relation_id)]["type"]= atom.split(",")[3].split(")")[0].upper()[:1] + atom.split(",")[3].split(")")[0][1:]
                    if 'nerjoinre(0,e,e' in atom :
                        entity_id= atom.split("nerjoinre(0,e,e")[1].split(",")[0]
                        print(doc_id,"**",entity_id, "**", data_JSON[doc_id]["entities"])
                        data_JSON[doc_id]["entities"][int(entity_id)]["type"]= atom.split(",")[3].split(")")[0][:1].upper() + atom.split(",")[3].split(")")[0][1:]
                    
                if sup :
                    relation_with_all =[]
                    for relation in data_JSON[doc_id]["relations"]:
                        if relation["type"]!="No":
                            relation_with_all.append(relation)

                    data_JSON[doc_id]["relations"]=relation_with_all

                        #data_JSON[doc_id]["relations"].remove(int(relation_id))
                if sup_Entity:
                    entity_with_all =[]
                    for entity in data_JSON[doc_id]["entities"]:
                        if entity["type"]!="No":
                            entity_with_all.append(entity)

                    data_JSON[doc_id]["entities"]=entity_with_all

    #print(data_JSON[:2])
    with open(file_result, 'w') as f:
        json.dump(data_JSON, f)
        print("*********** You can actually test your model with evaluation functions")

print("ok")

thresholds=[0.4]
#print("logs\spert_treshold_"+str(threshold)+"/*.json")
for threshold in thresholds:
    files =glob.glob("CONLL04\logs\spert_treshold_"+str(threshold)+"\predictions*.json")
    for file_ in files :
        with open(file_) as file :
            data_JSON = json.load(file)
        file_result=file_.split("\\")[0]
        file_result+="\\"+file_.split("\\")[1]
        file_name=file_.split('\\')[-1]
        file_result3=file_result+"\\neurASP-SUP_"+file_name
        NeurSUP(data_JSON,file_result3,sup=True,sup_Entity=True)
