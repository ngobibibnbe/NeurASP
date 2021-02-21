# To test MNIST

Installation: You need the following :

pip install cython  
pip install torch==1.7.1+cu101 torchvision==0.8.2+cu101 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
pip install problog[sdd]
pip install problog 

 
For testing Deepproblog, enter the deepproblog folder and do as following :

python generate_data.py
python run.py  
In the same folder make python runb.py for evaluation of the single neural network model


For testing NeurASP do as following :
Launch the mnist.ipynb notebook file 


You will have a record of test results in the runs folder  that you can visualize with:
tensorboard --logdirs=runs
