
cd ..
python test_conll04.py 
cd spert
python automated_evaluation.py "C:\Users\sophie\Desktop\iCS\reseacrch project\NeurASP\NeurASP\examples\NER & RE\CONLL04\logs\spert_treshold_0.4" conll04 > "CONLL04\result_0.4.txt"
python display_conll04.py

tensorboard --logdir=conll04/threshold