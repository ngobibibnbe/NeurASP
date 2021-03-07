conda activate NeurASP2
cd ..
python test_conll04.py 
cd spert
python automated_evaluation.py "../CONLL04/logs/spert_treshold_0.4" conll04 > "CONLL04/result_0.4.txt"
python automated_evaluation.py "../CONLL04/logs/spert_treshold_0.1" conll04 > "CONLL04/result_0.1.txt"
python automated_evaluation.py "../CONLL04/logs/spert_treshold_0.8" conll04 > "CONLL04/result_0.8.txt"


conda deactivate 
rm -r CONLL04/Threshold/*
python display_conll04.py
tensorboard --logdir=CONLL04/Threshold