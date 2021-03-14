import torch
import math
import signal
from logger import Logger
import time
from logic import term2list2
import random
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("../runs")

interrupt = False
zero_probability = False


def signal_handler(sig, frame):
        global interrupt
        print("Interrupted!")
        interrupt = True
        signal.signal(signal.SIGINT, signal.SIG_DFL)


def train(model, optimizer, query, eps=1e-8):
    loss = 0
    pos = True
    if type(query) is tuple:
        pos = query[1]
        query = query[0]
    ground = model.solve(query)
    for k in ground:
        if k == query:
            p,d = ground[k]
            break
    if pos:
        if p <= 0:
            print('zero probability query:', query, p)
        loss_grad = -1.0/(p+eps)
        loss += -math.log(p+eps)
    else:
        loss_grad = 1.0/(1.0-p+eps)
        loss += -math.log(1-p+eps)
    for k, v in d.items():
        if type(k[0]) is str:
            name = k[0]
            i = tuple(term2list2(k[1]))
            grad = torch.FloatTensor(loss_grad*v)
            optimizer.backward(name,i,grad)
        else:
            optimizer.add_param_grad(k,loss_grad*float(v))

    return loss


def train_model(model,queries,nr_epochs,optimizer, loss_function = train, test_iter=100,test=None,log_iter=100,snapshot_iter=None,snapshot_name='model',shuffle=True):
    signal.signal(signal.SIGINT, signal_handler)
    i = 1
    accumulated_loss = 0
    logger = Logger()
    start = time.time()
    print("Training for {} epochs ({} iterations).".format(nr_epochs,nr_epochs*len(queries)))
    if test is not None:
        #logger.log_list(i,test(model))
        print("ok")
    for epoch in range(nr_epochs):
        epoch_start = time.time()
        if interrupt:
            break
        print("Epoch",epoch+1)
        q_indices = list(range(len(queries)))
        if shuffle:
            random.shuffle(q_indices)
        for q in q_indices:
            q = queries[q]
            iter_time = time.time()
            if interrupt:
                break
            if i==1:
              accuracy_d = test(model)
              writer.add_scalar("NeurASP",100*accuracy_d,i)
            loss = loss_function(model, optimizer, q)
            accumulated_loss += loss
            optimizer.step()
            if snapshot_iter and i % snapshot_iter == 0:
                fname = '{}_iter_{}.mdl'.format(snapshot_name,i)
                print('Writing snapshot to '+fname)
                model.save_state(fname)
            if i % log_iter == 0:
                print('Iteration: ',i,'\tAverage Loss: ',accumulated_loss/log_iter)
                logger.log('time',i,iter_time - start)
                logger.log('loss',i,accumulated_loss/log_iter)
                for k in model.parameters:
                    logger.log(str(k),i,model.parameters[k])
                accumulated_loss = 0
                #and i % test_iter== 0) :
            """if i <=100: 
              #logger.log_list(i,test(model))
              print("ok",i)
              accuracy_d = test(model)
              writer.add_scalar("NeurASP",100*accuracy_d,i)"""

            if test is not None and i % 500 == 0: 
              #logger.log_list(i,test(model))
              print("ok",i)
              accuracy_d = test(model)
              writer.add_scalar("NeurASP",100*accuracy_d,i)

            i += 1
        optimizer.step_epoch()
        print('Epoch time: ',time.time()-epoch_start)
    return logger
