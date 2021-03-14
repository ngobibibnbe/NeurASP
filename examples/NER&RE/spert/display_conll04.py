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

print ("best values: ",max[2])
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
{0: 19.39, 500: 47.92, 1000: 87.76, 1500: 92.13, 2000: 92.65, 2500: 93.88, 3000: 95.19, 3500: 95.1, 4000: 94.44, 4500: 95.02, 5000: 95.6, 5500: 96.79, 6000: 96.83, 6500: 96.81, 7000: 95.79, 7500: 96.15, 8000: 96.41, 8500: 96.73, 9000: 97.4, 9500: 97.27, 10000: 96.14, 10500: 97.02, 11000: 96.52, 11500: 96.48, 12000: 96.96, 12500: 96.34, 13000: 97.82, 13500: 96.72, 14000: 97.02, 14500: 97.31, 15000: 97.11, 15500: 97.41, 16000: 97.55, 16500: 96.55, 17000: 97.51, 17500: 97.66, 18000: 96.78, 18500: 97.69, 19000: 97.51, 19500: 98.1, 20000: 97.86, 20500: 98.2, 21000: 97.61, 21500: 97.42, 22000: 96.93, 22500: 97.25, 23000: 97.88, 23500: 98.05, 24000: 97.91, 24500: 97.25, 25000: 96.25, 25500: 97.87, 26000: 98.14, 26500: 97.9, 27000: 97.67, 27500: 98.07, 28000: 96.24, 28500: 97.27, 29000: 98.29, 29500: 97.97, 30000:97.95}
{500: 9.34, 1000: 8.58, 1500: 8.7, 2000: 11.64, 2500: 14.6, 3000: 19.900000000000002, 3500: 23.18, 4000: 29.24, 4500: 26.5, 5000: 33.08, 5500: 39.5, 6000: 42.44, 6500: 44.379999999999995, 7000: 48.14, 7500: 52.32, 8000: 53.76, 8500: 54.08, 9000: 60.199999999999996, 9500: 64.68, 10000: 61.419999999999995, 10500: 66.82000000000001, 11000: 66.32000000000001, 11500: 69.39999999999999, 12000: 70.88, 12500: 71.32, 13000: 71.82, 13500: 76.72, 14000: 77.86, 14500: 72.92, 15000: 77.56, 15500: 76.78, 16000: 80.9, 16500: 80.7, 17000: 78.38000000000001, 17500: 79.66, 18000: 83.89999999999999, 18500: 82.12, 19000: 82.06, 19500: 81.42, 20000: 85.32, 20500: 83.16, 21000: 83.24000000000001, 21500: 83.06, 22000: 84.34, 22500: 85.92, 23000: 85.11999999999999, 23500: 83.22, 24000: 84.92, 24500: 86.98, 25000: 86.2, 25500: 86.18, 26000: 86.0, 26500: 86.8, 27000: 87.2, 27500: 88.16000000000001, 28000: 85.22, 28500: 87.78, 29000: 85.98, 29500: 86.83999999999999, 30000: 87.88}
{500: 68.7, 1000: 96.11, 1500: 98.13, 2000: 98.06, 2500: 98.27, 3000: 98.42, 3500: 96.43, 4000: 98.34, 4500: 98.27, 5000: 98.82, 5500: 98.27, 6000: 98.56, 6500: 99.08, 7000: 98.79, 7500: 98.59, 8000: 97.61999999999999, 8500: 98.91, 9000: 98.99, 9500: 98.92, 10000: 98.48, 10500: 99.29, 11000: 98.99, 11500: 99.32, 12000: 99.28, 12500: 98.9, 13000: 99.31, 13500: 99.38, 14000: 99.24, 14500: 99.11999999999999, 15000: 99.32, 15500: 99.32, 16000: 99.05000000000001, 16500: 99.21, 17000: 99.35000000000001, 17500: 99.31, 18000: 99.38, 18500: 99.35000000000001, 19000: 99.31, 19500: 99.06, 20000: 99.3, 20500: 99.24, 21000: 98.61999999999999, 21500: 99.24, 22000: 99.16, 22500: 99.27, 23000: 99.38, 23500: 99.33, 24000: 99.32, 24500: 99.5, 25000: 99.48, 25500: 99.26, 26000: 99.57000000000001, 26500: 99.18, 27000: 99.55000000000001, 27500: 99.37, 28000: 99.3, 28500: 99.56, 29000: 99.62, 29500: 99.42999999999999, 30000: 99.56}
