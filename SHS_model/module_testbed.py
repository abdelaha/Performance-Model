
import csv
from SHS_model import SHS_model
import matplotlib.pyplot as pl
import time


# initiate the cloud model
# winSize is the size of data elements which will be used to initialize the model and learn the cluster
# gpWinSize is GP window size
# nu_cluster: number of clusters
# controlFeedback: initial number of DB CUP cores that the system will start with.
# alpha: is the upper latency threshold (in sec) where the system should increase number of cores if reached
# beta: is the lower latency threshold (in sec) where the system should decrease the number of cores if reached
# sigma: confidence level for the controller
# controllerDelay: determines how many steps the controller should wait from its last control signal to issue a new one.
myModel = SHS_model(winSize = 1000, gpWinSize = 250, gpin_winSize =200 ,nu_cluster =3,
                 controlFeedback = 1, alpha = 3, beta = 2, alpha2 = 4, beta2 = 1.5, sigma = 0.9, controllerDelay = 6 )
# set model verbose to equal one to allow the model  to print some messages (mainly for debugging )
myModel.verbose = 1
winSize = 1000

fileName = "data17.csv"
# input, output and model state indices in data
indxIn = 0        # Input indices 
indxOut = 18        # Output indices
indxS = [14,17]     # State indices  
indxQ = [2,3, 4, 5, 6, 8, 12, 13]  #clustering data indicies

# load data
inData = 0
outData = 0
statData = []
counter = 0
control = []
with open(fileName, 'rt') as dataFile:
    reader = csv.reader(dataFile, delimiter=',')
    for row in reader:
        inData = float(row[indxIn])
        outData = float(row[indxOut])
        statData = [float(row[i]) for i in indxS]
        clusData = [float(row[i]) for i in indxQ]
        
        counter += 1         # for debuging purpose
        
        # get the control signal
        # sysControl function do the following:
        # if the model is not initilized, it keeps buffering data to the size of winSize then init it
        # otherwise, it update the model with the new data (inData, statData, outData), forecast the workload, predict
        # the latency, and lastly calculate the control signal and return it
        # sysControl takes the following arg:
        # sysIn: average Input (scalar)
        # sysStat: current utilization vector (list of four elements)
        # sysOut: Average Latency (scalar)
        # controlFeedback: current # CPU cores for DB server
        
        print('passing data #:' + str(counter))
        start_time = time.time()   # to cal. the running time of each iteration
        control.append(myModel.sysControl(inData,statData,outData,clusData,4,8,4))
        print("Runing time: %s seconds" % (time.time() - start_time))

        #if counter > 500:
        #    break
        #if counter > 10:  # for debuging purpose
         #    break

    print('End of File')

    # evalmodel function: calculate error metrics that evaluate the model prediction and forecasting.
    myModel.evalModel()
    # plot_results function: plot the prediction with its confidence intervals
    myModel.plot_results()

    myModel.export_results()
    # for debugging
    pl.figure(2)
    pl.clf()
    pl.plot(control)
    pl.show()



