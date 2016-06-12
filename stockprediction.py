# -*- coding: utf-8 -*-
"""
Created on Fri May 13 12:31:00 2016

@author: Amruta Pawar
"""

import matplotlib.pyplot as pl
import numpy as np
import pylab as py



inputs = []
test_inputs =[]
input_weights = []
output_weights = []
hidden2_weights = []
change_inweights = []
change_hid2weights = []
change_outweights = []
training_output = []
closingprice = []

def initialize():
    
    for a in range(0,3):
        new_weight = []
        for c in range(0,2):
            weight = round(np.random.rand(),3)
            new_weight.append(weight)
        new_weight.append(0.567)
        input_weights.append(new_weight)
    
    
    for x in range(0,3):
       hweight2 = []
       for y in range(0,3):
          weight = round(np.random.rand(),3)
          hweight2.append(weight)
       hidden2_weights.append(hweight2)
    
    
    
    for b in range(0,3):
        weight1 = []
        for c in range(0,1):
            weight = round(np.random.rand(),3)
            weight1.append(weight)
        output_weights.append(weight1)

    
    for d in range(0,3):
        new_hidweight1 = []
        for e in range(0,3):
            weight = 0.0
            new_hidweight1.append(weight)
        change_hid2weights.append(new_hidweight1)
    
    
    for d in range(0,3):
        new_weight1 = []
        for e in range(0,3):
            weight = 0.0
            new_weight1.append(weight)
        change_inweights.append(new_weight1)
    

    for b in range(0,3):
        weight1 = []
        for c in range(0,1):
            weight = 0.0
            weight1.append(weight)
        change_outweights.append(weight1)
 
"""get historical data for training from the excel sheet"""
def getdata():
    """dateinput,openprice,closingprice= np.genfromtxt('F:\CUNY\Fourth Semester\Program 4\yahoostockdata.csv',delimiter=',',usecols=(0,16,2),unpack=True,dtype=float)"""
    dateinput,openprice,closingprice= np.genfromtxt('F:\yahoostockdata.csv',delimiter=',',usecols=(0,16,2),unpack=True,dtype=float)
    
    print dateinput,openprice,closingprice
    
    
    for i in range(1,32):
        traininput = []
        dinput =[]
        output = []
        bias = 1 
        dinput.append(round(dateinput[i],8))
        dinput.append(round(openprice[i],8))
        dinput.append(bias)
        output.append(round(closingprice[i],8))
               
        traininput.append(dinput)
        traininput.append(output)
        
        inputs.append(traininput)
    print inputs
    
    for j in range(32,37):
        testinput = []
        test_dinput =[]
        test_output = []
        bias = 1
        test_dinput.append(round(dateinput[j],8))
        test_dinput.append(round(openprice[j],8))
        test_dinput.append(bias)
        test_output.append(round(closingprice[j],8))
       
        testinput.append(test_dinput)
        testinput.append(test_output)
        test_inputs.append(testinput)

    print "test inputs are " + str(test_inputs)
              
        
        
        
def sigmoid(x):
    return (1/(1+np.exp(-x)))
    

def runNN(input1):
    hidden = 3
    output = 1
    hidden_output =[]
    hidden2_output2 = []
    finaloutput = []
    
    for i in range(hidden):
        sum = 0.0
        for j in range(len(input1)):
            
            sum += (input1[j] * input_weights[i][j])
       
        hidden_output.append(round(sigmoid(sum),8))
    
    print "output of first layer nodes is " + str(hidden_output)
    
    for m in range(hidden):
        sum = 0.0
        for n in range(len(hidden_output)):
            sum += (hidden_output[n] * hidden2_weights[n][m])
        
        hidden2_output2.append(round(sigmoid(sum),8))
    
    print "output of second layer nodes is " + str(hidden2_output2)
    
    for k in range(output):
        sum = 0.0
        for l in range(len(hidden2_output2)):
            sum += (hidden2_output2[l] * output_weights[l][k])
        
        finaloutput.append(round(sigmoid(sum),8))
   
    print "final output is " + str(finaloutput)
    
    return hidden_output,hidden2_output2,finaloutput

def backpropagation(input1,target,alpha,hidden_output,hidden2_output2,finaloutput):
    output_delta = []
    
    output= 1
    hidden = 3
    
    
    """output deltas"""
    for k in range(output):
        error = target[k] - finaloutput[k]
        print "error is  " + str(error)
        output_delta.append(round(error*(finaloutput[k]*(1-finaloutput[k])),8))
    
    
    """update output weights"""
    for j in range(len(hidden_output)):
         for k in range(output):
             change = output_delta[k] * hidden2_output2[j]
             
             output_weights[j][k] = round(output_weights[j][k] + (alpha*change ),8)
             
             change_outweights[j][k] = change
    
    
    """hidden2 layer deltas"""
    hidden2_delta2 = [0.0] * 3
    for p in range(len(hidden2_output2)):
        error=0.0
        for q in range(output):
            error += output_delta[q] * hidden2_weights[p][q]
        hidden2_delta2[p]= round((error*(hidden2_output2[p]*(1-hidden2_output2[p]))*input1[p]),8)
   
    
    """update hidden2 weights"""
    for r in range(len(input1)):
        for s in range(hidden):
            change = hidden2_delta2[s] * hidden_output[r]
            hidden2_weights[r][s] = round(hidden2_weights[r][s] + (alpha * change),8)
            
            change_hid2weights[r][s] = change
   
    
    """hidden deltas"""
    hidden_delta = [0.0] * 3
    for j in range(len(hidden_output)):
        error=0.0
        for k in range(output):
            error += output_delta[k] * input_weights[j][k]
        hidden_delta[j]= round((error*(hidden_output[j]*(1-hidden_output[j]))*input1[j]),8)
    
    
    """update input weights"""
    for x in range(len(input1)):
        for y in range(hidden):
            change = hidden_delta[y] * input1[x]
            input_weights[x][y] = round(input_weights[x][y] + (alpha * change),8)
            
            change_inweights[x][y] = change
   
    
    """combined error"""
    error = 0.0 
    for k in range(len(target)):
        error = 0.5 * ( (target[k] - finaloutput[k])*(target[k] - finaloutput[k]))
    print "combined error is " + str(error)
    return error


def Plot(training_output):
        closingprice= np.genfromtxt('F:\CUNY\Fourth Semester\Program 4\yahoostockdata.csv',delimiter=',',usecols=(2),unpack=True,dtype=float)
        actualprice=[]        
        for e in range(32,len(closingprice)):
            actualprice.append(closingprice[e]*100)
        actualprice = np.array(actualprice)
        print "Actual price is " + str(actualprice)
        
        plot_output = []
        xinput = []
        for x in range(0,5):
             xinput.append(x+1)
        print "xinput is "+ str(xinput),str(len(training_output)),str(training_output[0])
            
        for l in range(0,len(training_output)):
            for m in range(0,len(training_output[l])):
                var = (training_output[l][m] * 100)
                plot_output.append(var)
        plot_output = np.array(plot_output)
        print "plot out is " + str(plot_output)
        
        plot_out = []
        for z in range(31,len(training_output)):
            plot_out.append(float(plot_output[z]))
        print "Predicted price is " + str(plot_out)   
        pl.title('YAHOO STOCK')
        pl.xlim(xmin=0,xmax=7)
        pl.ylim(ymin=30.1,ymax=40.1)
        
        pl.plot(xinput, actualprice,color="blue",linewidth=1.5,linestyle='-',label="Actual")
        pl.plot(xinput, plot_out,color="red",linewidth=1.5,linestyle='-',label="Predicted")
        pl.legend(loc='lower right',frameon=False)
       
        py.show
        
def test(test_inputs):
    for p in test_inputs:
        inputz = p[0]
        runinput,runhidden,runoutput = runNN(inputz)
        training_output.append(runoutput)
        print "inputz is " + str(len(runoutput))
        print "final training & test output is " + str(training_output)
        print "Inputs: " + str(p[0]) + "--->>" + str(runoutput) + " Target : " + str(p[1])   
        print "training output is " + str(training_output),str(len(training_output))
    Plot(training_output)

def train(inputs,iterations=14,alpha=0.15):
    for t in range(iterations):
        for p in inputs:
            input1 = p[0]
            target = p[1]
            hidden_output,hidden2_output2, finaloutput = runNN(input1)
            print "iter is " + str(t)
            if t == iterations-1:
                
                training_output.append(finaloutput)
            
            error =  backpropagation(input1,target,alpha,hidden_output,hidden2_output2,finaloutput)
        
        if t % 50 == 0:
             print "Combined error is " + str(error)
    print "new input weights are " + str(input_weights)
    print "new hidden weights are " + str(hidden2_weights)
    print "new output weights are " + str(output_weights)
    
    test(test_inputs) 
    
    
if __name__ == "__main__":
    getdata()
    initialize()
    train(inputs)
    