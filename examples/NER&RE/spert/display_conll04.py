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

thresholds=[0.0,0.1,0.3,0.5,0.8,0.4]
result={}
for threshold in thresholds :
    result[str(threshold)]=get_result_by_threshold(threshold)

#print(result)


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
                    #max[0][key_type+" "+key_eval+" "+metric]=[0]
                    max[2][key_type+" "+key_eval+" "+metric]=[0]
                    #max[1][key_type+" "+key_eval+" "+metric]=[0]
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
                                R_RE["neur"]["neur-sup"].append((epoch, value["neurASP-SUP"]))
                                R_RE["neur"]["neur"].append((epoch, value["neurASP"]))
                                R_RE["neur"]["spert"].append((epoch, value["predictions"]))
                            
                        #if key_type=="RE" and key_eval =="micro"and metric=="precision":
                        """if max[2][key_type+" "+key_eval+" "+metric][0]<float(value["neurASP-SUP"]) :
                            max[2][key_type+" "+key_eval+" "+metric]=[float(value["neurASP-SUP"]),epoch ,threshold]
                        if metric=="f1-score" and max[1][key_type+" "+key_eval+" "+metric][0]<float(value["neurASP"]) :
                            max[1][key_type+" "+key_eval+" "+metric]=[float(value["neurASP"]),epoch ,threshold]
                        if max[0][key_type+" "+key_eval+" "+metric][0]<float(value["predictions"]): 
                            max[0][key_type+" "+key_eval+" "+metric]=[float(value["predictions"]),epoch ,threshold]"""
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

#print ("best values: ",R_RE , R_NR, R_NER)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import  pandas as pd
print(R_NR)
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
        print(mat)
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
             op="Relation extraction"
        if R_NR==R:
            op="join named entity and relation extraction"
        if R_NER ==R:
            op="name entity recognition"

        plt.title('result on '+op+" for " +key)
        plt.legend()
        plt.show()


{'spert': {'spert': [('1', ' 2.76'), ('2', '29.71'), ('3', '49.64'), ('4', '62.71'), 
('5', '68.18'), ('6', '66.91'), ('7', '68.69'), ('8', '69.54'), ('9', '69.77'), ('10',
 '69.68'), ('11', '69.35'), ('12', '71.21'), ('13', '69.22'), ('14', '70.44'), 
 ('15', '70.44'), ('16', '71.67'), ('17', '70.67'), ('18', '70.84'), ('19', '71.24'), 
 ('20', '70.99')], 'neur': [('1', ' 2.76'), ('2', '29.64'), ('3', '49.53'), ('4', '63.03'),
  ('5', '68.32'), ('6', '66.60'), ('7', '68.22'), ('8', '68.58'), ('9', '69.58'),
   ('10', '69.32'), ('11', '68.98'), ('12', '71.03'), ('13', '68.75'), ('14', '70.17'), 
   ('15', '70.17'), ('16', '71.50'), ('17', '70.50'), ('18', '70.66'), ('19', '71.06'), 
   ('20', '70.81')], 'neur-sup': [('1', ' 2.17'), ('2', '27.51'), ('3', '49.55'), 
   ('4', '65.19'), ('5', '70.45'), ('6', '69.77'), ('7', '69.80'), ('8', '71.16'), 
   ('9', '71.52'), ('10', '70.77'), ('11', '70.67'), ('12', '72.27'), ('13', '70.97'),
    ('14', '72.13'), ('15', '71.88'), ('16', '73.38'), ('17', '72.35'), ('18', '72.31'), 
    ('19', '72.76'), ('20', '72.51')]}, 'neur': {'spert': [], 'neur': [], 'neur-sup': []},
     'neur-sup': {'spert': [('1', ' 2.23'), ('2', ' 7.69'), ('3', '10.51'), ('4', '13.16'), ('5', '14.03'), ('6', '12.92'), ('7', '15.53'), ('8', '14.35'), ('9', '15.55'), ('10', '15.18'), ('11', '14.87'), ('12', '15.61'), ('13', '15.24'), ('14', '15.61'), ('15', '15.85'), ('16', '16.13'), ('17', '15.95'), ('18', '16.24'), ('19', '15.98'), ('20', '16.11')], 'neur': [('1', ' 0.69'), ('2', ' 1.35'), ('3', ' 1.54'), ('4', ' 1.76'), ('5', ' 1.74'), ('6', ' 1.73'), ('7', ' 1.91'), ('8', ' 1.89'), ('9', ' 2.11'), ('10', ' 1.95'), ('11', ' 1.93'), ('12', ' 1.99'), ('13', ' 2.02'), ('14', ' 2.00'), ('15', ' 2.01'), ('16', ' 2.07'), ('17', ' 2.07'), ('18', ' 2.04'), ('19', ' 2.04'), ('20', ' 2.03')], 'neur-sup': [('1', ' 3.12'), ('2', '27.12'), ('3', '52.56'), ('4', '61.85'), ('5', '69.25'), ('6', '68.57'), ('7', '70.13'), ('8', '70.88'), ('9', '71.02'), ('10', '70.97'), ('11', '71.56'), ('12', '73.71'), ('13', '71.78'), ('14', '72.05'), ('15', '71.78'), 
('16', '72.91'), ('17', '72.50'), ('18', '72.47'), ('19', '73.12'), ('20', '72.76')]}}