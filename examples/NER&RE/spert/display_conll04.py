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
    result = "CONLL04/result_"+str(threshold)+".txt"
    
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

#thresholds=[0.0,0.1,0.3,0.5,0.8,0.4]
thresholds=[0.0]

result={}
for threshold in thresholds :
    result[str(threshold)]=get_result_by_threshold(threshold)

print(result)


max_spert=[0,0]
max_neur=[0,0]
max_neur0=[0,0]
# display it with tensorboard
from torch.utils.tensorboard import SummaryWriter
import numpy as np
database="CONLL04"
metrics=["precision","recall","f1-score"]
neurasp={}
spert ={}
writer = SummaryWriter(database+"/Threshold")
max=[{},{},{} ]
a=0
R_NER={"spert":{"spert":[],"neur": [] ,'neur-sup':[]},"neur": {"spert":[],"neur": [] ,'neur-sup':[]},'neur-sup':{"spert":[],"neur": [] ,'neur-sup':[]}}
R_RE={"spert":{"spert":[],"neur": [] ,'neur-sup':[]},"neur": {"spert":[],"neur": [] ,'neur-sup':[]},'neur-sup':{"spert":[],"neur": [] ,'neur-sup':[]}}
R_NR={"spert":{"spert":[],"neur": [] ,'neur-sup':[]},"neur": {"spert":[],"neur": [] ,'neur-sup':[]},'neur-sup':{"spert":[],"neur": [] ,'neur-sup':[]}}
for threshold,values_parts in result.items():
    #0.1
    a=a+1
    for key_type,type_evaluation in values_parts.items() :
        #NER
        for key_eval, evaluation in type_evaluation.items() :
            #micro
            for metric, values in evaluation.items() :
                #precision
                if a==1:
                    max[0][key_type+" "+key_eval+" "+metric]=[0]
                    max[2][key_type+" "+key_eval+" "+metric]=[0]
                    max[1][key_type+" "+key_eval+" "+metric]=[0]
                i=0
                if i==0 :#key_eval =="micro" :# or key_type=="RE" ): 
                    #print(key_eval," ",key_type, " ",metric)
                    i=0
                    values=dict(sorted(values.items(), key=lambda item: int(item[0])))#k: v for k, v in sorted(values.items(), key=lambda item: int(item[0]))}
                    for epoch, value in values.items():
                        i+=1
                        if float(threshold)==0.0 and key_eval=="macro" and metric=='f1-score':
                            if key_type =="NER" and key_eval=="macro":
                                R_NER["neur-sup"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_NER["neur-sup"]["neur"].append((epoch, value["neurASP"]))
                                R_NER["neur-sup"]["spert"].append((epoch, value["predictions"]))
                            if key_type =="RE":
                                R_RE["neur-sup"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_RE["neur-sup"]["neur"].append((epoch, value["neurASP"]))
                                R_RE["neur-sup"]["spert"].append((epoch, value["predictions"]))
                            if key_type =="NER-RE":
                                R_NR["neur-sup"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_NR["neur-sup"]["neur"].append((epoch, value["neurASP"]))
                                R_NR["neur-sup"]["spert"].append((epoch, value["predictions"]))
                        
                        if float(threshold)==0.5 and key_eval=="macro" and metric=='f1-score':
                            if key_type =="NER" and key_eval=="macro":
                                R_NER["spert"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_NER["spert"]["neur"].append((epoch, value["neurASP"]))
                                R_NER["spert"]["spert"].append((epoch, value["predictions"]))
                            if key_type =="RE":
                                R_RE["spert"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_RE["spert"]["neur"].append((epoch, value["neurASP"]))
                                R_RE["spert"]["spert"].append((epoch, value["predictions"]))
                            if key_type =="NER-RE":
                                R_NR["spert"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_NR["spert"]["neur"].append((epoch, value["neurASP"]))
                                R_NR["spert"]["spert"].append((epoch, value["predictions"]))

                        if float(threshold)==0.4 and key_eval=="macro" and metric=='f1-score':
                            if key_type =="NER" :
                                R_NER["neur"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_NER["neur"]["neur"].append((epoch, value["neurASP"]))
                                R_NER["neur"]["spert"].append((epoch, value["predictions"]))
                            if key_type =="RE":
                                R_RE["neur"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_RE["neur"]["neur"].append((epoch, value["neurASP"]))
                                R_RE["neur"]["spert"].append((epoch, value["predictions"]))

                            if key_type =="NER-RE":
                                R_NR["neur"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_NR["neur"]["neur"].append((epoch, value["neurASP"]))
                                R_NR["neur"]["spert"].append((epoch, value["predictions"]))
                            
                        #if key_type=="RE" and key_eval =="micro"and metric=="precision":
                        if max[2][key_type+" "+key_eval+" "+metric][0]<float(value["neurASP-SUP"]) :
                            max[2][key_type+" "+key_eval+" "+metric]=[float(value["neurASP-SUP"]),epoch ,threshold]
                        if metric=="f1-score" and max[1][key_type+" "+key_eval+" "+metric][0]<float(value["neurASP"]) :
                            max[1][key_type+" "+key_eval+" "+metric]=[float(value["neurASP"]),epoch ,threshold]
                        if max[0][key_type+" "+key_eval+" "+metric][0]<float(value["predictions"]): 
                            max[0][key_type+" "+key_eval+" "+metric]=[float(value["predictions"]),epoch ,threshold]
                        ####################après enlever pour qu'on ait aussi le micro 
                        #print(epoch, " ", float(value["neurASP-SUP"]), " ",float(value["neurASP-SUP"]) )
                        writer.add_scalars("Test-"+database+" "+str("runs/"+threshold+"/"+key_eval+"/"+key_type+"/"+metric) ,
                        {"SPERT": float(value["predictions"]),"SPERT-Neur": float(value["neurASP"]), "SPERT-Neur-SUP": float(value["neurASP-SUP"])}, int(epoch))
                        #                    {"SPERT": float(value["predictions"]), "SPERT-Neur-SUP": float(value["neurASP-SUP"])}, int(epoch))
                        #                   "SPERT-Neur": float(value["neurASP"]), "SPERT-Neur-FE": float(value["neurASP-FE"]),
                        #                    "SPERT-Neur-FE-SUP": float(value["neurASP-FE-SUP"]),"""
                                            
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

print ("best values: ",max)
""""import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import  pandas as pd
for R in [R_NER, R_NR, R_RE]:
    for key,values in   R.items():
        mat=[[],[],[],[]]
        mat[3]=[i for i in range(1,21,1)]
        for key2,vals in values.items():
            tmp1=[]
            tmp2=[]
            for tuple in vals :
                tmp1.append(float(tuple[0]))
                tmp2.append(float(tuple[1]))
            if key2=="spert":
                mat[0]=tmp2
                #mat[3]=tmp1
                #print(mat[0])
            if key2=="neur":
                mat[1]=tmp2
            if key2=="neur-sup":
                mat[2]=tmp2
        sns.set(style='darkgrid')
        if R_RE ==R:
             op="Relation extraction"
        if R_NR==R:
            op="join named entity and relation extraction"
        if R_NER ==R:
            op="name entity recognition"
        print("+++",op)
        df =pd.DataFrame({'spert':mat[0], 'neurASP':mat[1],'neurASP-SUP':mat[2],
        "epoch":mat [3] })
        
        sns.lineplot(
            data=df,  x="epoch", y="spert" , color='red'
        )
        sns.lineplot(
            data=df,  x="epoch", y="neurASP", color='yellow'
        )
        sns.lineplot(
            data=df,  x="epoch", y="neurASP-SUP",color='green'
        )
        if R_RE ==R:
             op="RE"
        if R_NR==R:
            op="NER join RE"
        if R_NER ==R:
            op="NER "

        plt.title('result on '+op+" best performances of " +key)
        plt.legend()
        plt.show()

"""
