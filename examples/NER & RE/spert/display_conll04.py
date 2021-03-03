def get_macro_and_micro(dict, file,nam):
    type= nam.split("_")[0]
    name = (nam.split("_")[-1].split(".json")[0])
    #print(name)
    for line in file :
        
        if "micro" in line :
                               
            dict["micro"]["precision"][name][type]=line.split("        ")[2]
            dict["micro"]["recall"][name][type]=line.split("        ")[3]
            dict["micro"]["f1-score"][name][type]=line.split("        ")[4]
        if "macro" in line :
            
            dict["macro"]["precision"][name][type]=line.split("        ")[2] 
            dict["macro"]["recall"][name][type]=line.split("        ")[3]
            dict["macro"]["f1-score"][name][type]=line.split("        ")[4]
            #print("**macro",dict)
            return dict 
        
def get_result_by_threshold(threshold):
    result = "conll04/result_"+str(threshold)+".txt"
    
    f = open(result, "r")   
    all=[]
    dict={}
    dict["NER"]={}
    dict["RE"]={}
    dict["NER-RE"]={}
    for dicte in [ dict["NER-RE"],  dict["NER"], dict["RE"]] :
        dicte["micro"]={}
        dicte["micro"]["precision"]={}
        dicte["micro"]["recall"]={}
        dicte["micro"]["f1-score"]={}
        dicte["macro"]={}
        dicte["macro"]["precision"]={}
        dicte["macro"]["recall"]={}
        dicte["macro"]["f1-score"]={}
        for dicte in [dicte["macro"]["f1-score"],dicte["macro"]["precision"],dicte["macro"]["recall"],
                      dicte["micro"]["precision"],dicte["micro"]["recall"],dicte["micro"]["f1-score"]] : 
            for id in range(1,21,1):
                dicte[str(id)]={}
        
    for line in f :
        if "###"  in str(line):
            name =(line.split("#")[-1]).replace(" ","").replace("\n","")   
            
        if "Entities (named entity recognition (NER))" in str(line):
            
            dict["NER"]=get_macro_and_micro(dict["NER"],f,name)
        if "Without named entity classification (NEC)" in str(line):
            
            dict["RE"] =get_macro_and_micro(dict["RE"],f,name)
        if "With named entity classification (NEC)" in str(line):
            
            dict["NER-RE"] =get_macro_and_micro(dict["NER-RE"],f,name)
            
    #print(all) 
    f.close()
    return dict

thresholds=[0.4]
result={}
for threshold in thresholds :
    result[str(threshold)]=get_result_by_threshold(threshold)

print(result)



# display it with tensorboard
from torch.utils.tensorboard import SummaryWriter
import numpy as np
database="CONLL04"
metrics=["precision","recall","f1-score"]
neurasp={}
spert ={}
writer = SummaryWriter(database+"/Threshold")

for threshold,values_parts in result.items():
    #0.1
        for key_type,type_evaluation in values_parts.items() :
            print(key_type)
            #NER
            for key_eval, evaluation in type_evaluation.items() :
                #micro
                
                for metric, values in evaluation.items() :
                        #precision
                    if key_eval =="micro" : #and (key_type=="NER-RE" or key_type=="RE" ): 
                        print(values)
                        
                        for epoch, value in values.items():
                            ####################après enlever pour qu'on ait aussi le micro 
                            print(epoch, " ", threshold)
                            writer.add_scalars("Test-"+database+" "+str("runs/"+threshold+"/"+key_type+"/"+key_eval+"/"+metric) ,
                                                {"SPERT": float(value["predictions"]), "SPERT-Neur-SUP": float(value["neurASP-SUP"])}, int(epoch))
                            """                    "SPERT-Neur": float(value["neurASP"]), "SPERT-Neur-FE": float(value["neurASP-FE"]),
                                                "SPERT-Neur-FE-SUP": float(value["neurASP-FE-SUP"]),"""
                                                
                            #ceci c'est en function du threshold"""
                            if database =="scierc":
                                writer.add_scalars("Test-epoch"+threshold+" "+database+" "+str(key_type+"/"+key_eval+"/"+metric) ,
                                                {"SPERT": float(value["predictions"]), 
                                                "SPERT-Neur": float(value["neurASP"]), "SPERT-Neur-FE": float(value["neurASP-FE"]),
                                                "SPERT-Neur-FE-SUP": float(value["neurASP-FE-SUP"]),
                                                "SPERT-Neur-SUP": float(value["neurASP-SUP"])}, float(epoch))
                            """ else :
                                writer.add_scalars("Test-epoch"+threshold+" "+database+" "+str(key_type+"/"+key_eval+"/"+metric) ,
                                                {"SPERT": float(value["predictions"]), 
                                                "SPERT-Neur": float(value["neurASP"]), "SPERT-Neur-FE": float(value["neurASP-FE"]),
                                                "SPERT-Neur-FE-SUP": float(value["neurASP-FE-SUP"])}, float(epoch)*100)"""
writer.close()