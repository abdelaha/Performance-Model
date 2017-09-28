

# libaries
import csv
import numpy as np
import matplotlib.pyplot as pl
import pyGPs as gp
from sklearn.cluster import KMeans
from math import log, pi, sqrt
import scipy.stats
import logging

class SHS_model:
    'SHS model learning for performance-aware control of cloud vertical elasticity '


    def __init__(self, winSize = 1000, gpWinSize = 250,gpin_winSize =200 ,nu_cluster = 3,
                 controlFeedback = 1, alpha = 3, beta = 2, sigma = 0.9, alpha2 = 4, beta2 = 1.5,controllerDelay = 6 ):
        #  online Learning parameters and variables
        self.__winSize = winSize
        self.__gpWinSize = gpWinSize
        self.__gpin_winSize = gpin_winSize ;
        self.__fileName = "data_init.csv"
        # system data
        self.__inData = []
        self.__outData = []
        self.__statData = []
        self.__clusData = []
        
        # unclontroled input forecast model
        self.__gpfX = []
        self.__gpfY = []


        self.__sysState = []
        # Number of clusters
        self.__Nc = nu_cluster
        self.__onlineCluster = 1

        # model status
        self.__modelInitilization  = 0  # 0: initial, 1: learn and predict
        self.__modelTimeIdx = 0
        self.__modelstate = 0
        # model control output
        self.__controlOut = []
        self.__controlFeedback = controlFeedback
        self.__alpha = alpha
        self.__beta = beta
        self.__alpha2 = alpha2
        self.__beta2 = beta2
        self.__sigma = sigma
        self.__controllerDelay = controllerDelay
        self.__controllerCounter = 0
        # model objects
        # state models
        self.__GPs = []

        # forecast model
        self.__gpFmdl = gp.GPR()

        # classifier model
        self.__myCluster = KMeans(n_clusters=self.__Nc ,init='random', random_state=0)



        # variables to store prediction data for evaluation and plotting:
        # forecasting workload variables
        self.__In_mean = []
        self.__In_si = []
        self.__In_pred_data = []

        # classification variables
        self.__sysModes = []

        # for output variables
        self.__out_mean =[]
        self.__out_si = []
        self.__out_pred_data = []
        # for i in self.__Nc:
        #     self.__out_mean.append([])
        #     self.__out_si.append([])
        #     self.__out_pred_data.append([])


        # other
        self.verbose = 1

        # pass the offline data to initialize the model
        # input, output and model state indices in data
        indxIn = 0        # Input indices 
        indxOut = 18        # Output indices
        indxS = [14,17]     # State indices  
        indxQ = [2,3, 4, 5, 6, 8, 12, 13]  #clustering data indicies
        
        inData = 0
        outData = 0
        statData = []
        clusData = [];
        with open(self.__fileName, 'rt') as dataFile:
            reader = csv.reader(dataFile, delimiter=',')
            for row in reader:
                inData = float(row[indxIn])
                outData = float(row[indxOut])
                statData = [float(row[i]) for i in indxS]
                clusData = [float(row[i]) for i in indxQ]
                self.__initModel(inData, statData, outData, clusData)

                if self.__modelInitilization:
                    break

#    def __featureExtraction(self, X):
#        m = len(X[0])  # number of measurements
#        n = len(X)  # Window Size
#        Y = []  # feature Vector
#
#        # Calculate mean feature
#        for i in range(m):
#            num = []
#            for j in range(n):
#                num.append(X[j][i])
#
#            Y.append(np.mean(num))
#
#
#            # Calculate Variance feature
#            # Y.append(np.var(num))
#
#        return Y

    def __initModel(self, sysIn, sysStat, sysOut, sysClus):
        
        # buffer data untill the window size is reached
        self.__inData.append(sysIn)
        self.__outData.append(sysOut)
        self.__statData.append(sysStat)
        self.__clusData.append(sysClus)


        # Lean forecast model and system indx
        self.__gpfX.append(self.__modelTimeIdx)
        self.__gpfY.append(sysIn)

        while len(self.__gpfX) > self.__gpin_winSize:
            # delete oldest data
            self.__gpfX.pop(0)
            self.__gpfY.pop(0)

        self.__modelTimeIdx += 1

        if len(self.__inData) >= self.__winSize:
            # Learn the models  (align the data to input output format  Y(k+1) = f(x(k),u(k+1)))
            self.__inData.pop(0)
            self.__outData.pop(0)
            # self.__statData.pop()
            # self.__clusterFeatures.pop()

            # classify the data
            clustersX = self.__myCluster.fit_predict(self.__clusData)
            if self.verbose:
                print ("training state-space models using  GPs")

            for i in range(self.__Nc):
                gprX = []
                gprY = []

                for j in range(len(clustersX) - 1):
                    if clustersX[j] == i:
                        gprX.append([self.__inData[j]] + self.__statData[j])
                        gprY.append(self.__outData[j])
                gprX = np.array(gprX)
                gprY = np.array(gprY)
                gpmdl = gp.GPR()
                m = gp.mean.Zero()
                RBF_hyp_init = [0.5] * (len(sysStat) + 2) 
                k = gp.cov.RBFard(D=None, log_ell_list=RBF_hyp_init[:-1], log_sigma=RBF_hyp_init[-1])
                gpmdl.setPrior(mean=m, kernel=k)
                if self.verbose:
                    print ("training GP of mode: " + str(i))
                # gpmdl.getPosterior(gprX,gprY)
                gpmdl.setNoise(log_sigma=np.log(0.8))

                gpmdl.setOptimizer('Minimize')  # ('Minimize');
                gpmdl.optimize(gprX, gprY)  # ,numIterations=100)

                self.__GPs.append(gpmdl)

            if self.verbose:
                print ("training forecast Model using GP")

            try:
                k_f = gp.cov.RBF(log_ell=1, log_sigma=1)
                self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
                self.__gpFmdl.setNoise(log_sigma=np.log(0.8))
                self.__gpFmdl.setOptimizer('BFGS')  # ('Minimize');
                self.__gpFmdl.optimize(np.array(self.__gpfX), np.array(self.__gpfY))  # ,numIterations=100)
            except:
                print('can quasi-newton it (forecast)')
                self.__gpFmdl = gp.GPR()
                k_f = gp.cov.RBF(log_ell=1, log_sigma=1)
                self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
                self.__gpFmdl.setNoise(log_sigma=np.log(0.8))
                self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
                self.__gpFmdl.setOptimizer('Minimize')  # ('Minimize');
                self.__gpFmdl.optimize(np.array(self.__gpfX), np.array(self.__gpfY))  # , numIterations=100)
            self.__sysState = self.__statData[-1]
            self.__modelInitilization = 1

    def __updateModel(self, sysIn, sysStat, sysOut,sysClus):

        # update the forecast model
        if self.verbose:
            print ("update forecast Model")
        # add the new data and delete the oldest  (slid the win)
        gpfX = np.append(self.__gpFmdl.x, np.array([self.__modelTimeIdx]).reshape(-1, 1), axis=0)
        gpfY = np.append(self.__gpFmdl.y, np.array([sysIn]).reshape(-1, 1), axis=0)

        while gpfY.size > self.__gpin_winSize:
            # delete oldest data
            gpfX = np.delete(gpfX, 0, 0)
            gpfY = np.delete(gpfY, 0, 0)
        self.__modelTimeIdx += 1
        # get the old hyp
        hyp_f = self.__gpFmdl.covfunc.hyp

        # relearn the model with the old hyp as a prior model
        try:
            self.__gpFmdl = gp.GPR()
            k_f = gp.cov.RBF(log_ell=hyp_f[0], log_sigma=hyp_f[1])
            self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
            self.__gpFmdl.setNoise(log_sigma=np.log(0.8))
            self.__gpFmdl.setOptimizer('BFGS')
            self.__gpFmdl.optimize(gpfX, gpfY)
        except:
            print('cannot BFGS it, forecast')
            self.__gpFmdl = gp.GPR()
            k_f = gp.cov.RBF(log_ell=hyp_f[0], log_sigma=hyp_f[1])
            self.__gpFmdl.setPrior(mean=gp.mean.Zero(), kernel=k_f)
            self.__gpFmdl.setNoise(log_sigma=np.log(0.8))
            self.__gpFmdl.setOptimizer('Minimize')
            self.__gpFmdl.optimize(gpfX, gpfY)

        # Update cluster model
        if self.__onlineCluster:
            if self.verbose:
                print ("update cluster Model ...")
            self.__myCluster = KMeans(n_clusters=self.__Nc,
                                      init=self.__myCluster.cluster_centers_,
                                      random_state=0,n_init=1)


        # update the cluster data  (slid the window)
        self.__clusData.append(sysClus)
        self.__clusData.pop(0)
        # update the clusterer
        self.__myCluster.fit(self.__clusData)

        # update system state GP model
        if self.verbose:
            print ("update system Models ...")
            
        # Estiamte discrete Mode

        predCluster = self.__myCluster.predict(np.array(sysClus).reshape(1, -1))
        self.__modelstate = np.asscalar(predCluster[0])
        self.__sysModes.append(self.__modelstate)
        
        # pull the model used for the last prediction
        gprMdl = self.__GPs[self.__modelstate]
        newgprX = np.array([sysIn] + self.__sysState).reshape(1, -1)
        gprX = np.append(gprMdl.x, newgprX, axis=0)
        gprY = np.append(gprMdl.y, np.array([sysOut]).reshape(1, -1), axis=0)
        # gprMdl.x = np.append(gprMdl.x, Xs, axis=0)
        # gprMdl.y = np.append(gprMdl.y, [outData[i]], axis=0)

        while gprY.size > self.__gpWinSize:
            gprX = np.delete(gprX, 0, 0)
            gprY = np.delete(gprY, 0, 0)

        hyp = gprMdl.covfunc.hyp

        gprMdl = gp.GPR()
        m = gp.mean.Zero()
        # k = gp.cov.SumOfKernel(gp.cov.RBFard(D=None, log_ell_list=hyp, log_sigma=1.),gp.cov.Noise(1))
        k = gp.cov.RBFard(D=None, log_ell_list=hyp[:-1], log_sigma=hyp[-1])
        gprMdl.setPrior(mean=m, kernel=k)
        # gprMdl.getPosterior(gprX,gprY)
        gprMdl.setNoise(log_sigma=np.log(0.81))
        try:
            gprMdl.setOptimizer('BFGS')
            gprMdl.optimize(gprX, gprY)
        except:
            print('cannot BFGS it ')
            gprMdl = gp.GPR()
            m = gp.mean.Zero()
            # k = gp.cov.SumOfKernel(gp.cov.RBFard(D=None, log_ell_list=hyp, log_sigma=1.),gp.cov.Noise(1))
            k = gp.cov.RBFard(D=None, log_ell_list=hyp[:-1], log_sigma=hyp[-1])
            gprMdl.setPrior(mean=m, kernel=k)
            # gprMdl.getPosterior(gprX,gprY)
            gprMdl.setNoise(log_sigma=np.log(0.81))
            gprMdl.setOptimizer('Minimize')
            gprMdl.optimize(gprX, gprY)
        self.__GPs[self.__modelstate] = gprMdl

        # Update system state
        self.__sysState = sysStat

        # save the data for Error calculation and prediction evaluation
        self.__In_pred_data.append(sysIn)
        self.__out_pred_data.append(sysOut)

    def __predictPerformance(self):

        if self.verbose:
            print ("forecasting and predicting ...")
        # forecast workLoad (predict workload input)
        INm, INsi, foom, foos2, foolp = self.__gpFmdl.predict(np.array([self.__modelTimeIdx + 1]).reshape(1, -1))

        if INm[0] < 0:  # bound the prediction
            INm[0] = 0
        self.__In_mean.append(np.asscalar(INm[0]))
        self.__In_si.append(np.asscalar(INsi[0]))

   

        # initialize the return
        Y_mean = 0
        Y_si = 0

        # calculate the input
        Xs = np.array([np.asscalar(INm[0])] + self.__sysState).reshape(1, -1)
        # pull the model
        gprMdl = self.__GPs[self.__modelstate]
        # predict the system performance (latency)
        ym, ysi, fm, fs2, lp = gprMdl.predict(Xs)

        if ym[0] < 0:  # bound the prediction
            ym[0] = 0

        Y_mean = np.asscalar(ym[0])
        Y_si = np.asscalar(ysi[0])

        self.__out_mean.append(Y_mean)
        self.__out_si.append(Y_si)

        return Y_mean, Y_si

    def sysControl(self, sysIn, sysStat, sysOut,sysClus, controlFeedback, MaxN, MinN):
        # sysIn: request rate (scalar)
        # sysStat: current utilization vector (list)
        # sysOut: Average Latency (scalar)
        # sysClus: Host measurement  (list)
        # controlFeedback: current number of CPU cores (scalar)
        self.__controlFeedback = controlFeedback
        if not(self.__modelInitilization):
            self.__initModel(sysIn,sysStat,sysOut, sysClus)
        else:
			try:
				self.__updateModel(sysIn,sysStat,sysOut, sysClus)
			except:
				print('Error during updating the model')
				logging.info('Error during updating the model')
        

        controlSignal = controlFeedback

        ym, ysi = self.__predictPerformance()
        p_lge_alpha = (1 - scipy.stats.norm(ym, ysi).cdf(self.__alpha))
        P_lle_beta = scipy.stats.norm(ym, ysi).cdf(self.__beta)
		
        if self.verbose:
			print('step: ' + str(len(self.__controlOut)))
            print('predicted latency: N(' +str(ym)  +','+str(ysi) +')')
            print('P(L > '+ str(self.__alpha) +')= ' + str(p_lge_alpha))
            print('P(L < ' + str(self.__beta) + ')= ' + str(P_lle_beta))

			logging.info('step: ' + str(len(self.__controlOut)))
			logging.info('predicted latency: N(' + str(ym) + ',' + str(ysi) + ')')
            logging.info('P(L > ' + str(self.__alpha) + ')= ' + str(p_lge_alpha))
            logging.info('P(L < ' + str(self.__beta) + ')= ' + str(P_lle_beta))
        if p_lge_alpha >= self.__sigma and self.__controllerCounter >= self.__controllerDelay and self.__controlFeedback < MaxN:
			#check the second upper limit probability
            p_lge_alpha2 = (1 - scipy.stats.norm(ym, ysi).cdf(self.__alpha2))
            if self.verbose:
	            print('P(L > '+ str(self.__alpha2) +')= ' + str(p_lge_alpha2))
	            logging.info('P(L > ' + str(self.__alpha2) + ')= ' + str(p_lge_alpha2))
	
            if p_lge_alpha2 >= self.__sigma and self.__controlFeedback < MaxN - 1:
                controlSignal = self.__controlFeedback + 2
            else:
                controlSignal = self.__controlFeedback + 1
            self.__controllerCounter = 0

        elif P_lle_beta >= self.__sigma and self.__controllerCounter >= self.__controllerDelay and self.__controlFeedback > MinN :
            #check the second upper limit probability
            P_lle_beta2 = scipy.stats.norm(ym, ysi).cdf(self.__beta2)
            if self.verbose:
                print('P(L > '+ str(self.__beta2) +')= ' + str(P_lle_beta2))
                logging.info('P(L > ' + str(self.__beta2) + ')= ' + str(P_lle_beta2))
            #update the control signal	
            if P_lle_beta2 >= self.__sigma and self.__controlFeedback > MinN + 1:
                controlSignal = self.__controlFeedback - 2
            else:
                controlSignal = self.__controlFeedback - 1
            self.__controllerCounter = 0
        else: # No control action has been taken
            self.__controllerCounter += 1

        # Save the control history
        self.__controlOut.append(controlSignal)
        return controlSignal

    def evalModel(self):
        # Model Evaluation

        print('Prediction evaluation metrics for each mode:')
        for i in range(self.__Nc):
            print('Mode ' + str(i) + ':')
			logging.info('Mode ' + str(i) + ':')
            ym = [self.__out_mean[j] for j in range(len(self.__out_pred_data)) if self.__sysModes[j] == i]
            ysi = [self.__out_si[j] for j in range(len(self.__out_pred_data)) if self.__sysModes[j] == i]
            yreal = [self.__out_pred_data[j] for j in range(len(self.__out_pred_data)) if self.__sysModes[j] == i]
            if len(ym) > 0:
                self.__cal_Error_Metrics(ym, ysi, yreal)

        # Evaluate the forecast prediction
        print('Forecasting evaluation metrics:')
        return self.__cal_Error_Metrics(self.__In_mean, self.__In_si, self.__In_pred_data)

    def __cal_Error_Metrics(self, y_m, y_si, y_real):
        MSE = 0
        MRSE = 0
        LD = 0
        n = len(y_real)
        Error = [y_m[i] - y_real[i] for i in range(n)]

        # Calculate Mean Square Error
        MSE = 1.0 / n * sum(np.power(Error, 2).tolist())

        # Calculate Mean Root Square Error
        MRSE = 1.0 * sqrt(sum(np.power(Error, 2).tolist()) / sum(np.power(y_real, 2).tolist()))

        # Calculate Log predictive density error
        LD = 0.5 * log(2 * pi) + 1 / (2 * n) * \
                                 sum(np.log(np.power(y_si[:n], 2)).tolist() + np.divide(np.power(Error, 2),
                                                                                        np.power(y_si[:n], 2)).tolist())

        print('MSE: ' + str(MSE))
        print('MRSE: ' + str(MRSE))
        print('LD: ' + str(LD))

        logging.info('MSE: ' + str(MSE))
        logging.info('MRSE: ' + str(MRSE))
        logging.info('LD: ' + str(LD))
        return MSE, MRSE, LD

    def plot_results(self):
        # Draw the Results
        n = len(self.__out_pred_data)
        colorArray = ['g', 'y', 'r']
        colors = [colorArray[i] for i in self.__sysModes]
        sigma = np.sqrt(self.__out_si[:n])

        lowerBound = np.array(self.__out_mean[:n]) - 2 * sigma
        upperBound = np.array(self.__out_mean[:n]) + 2 * sigma
        for i in range(len(lowerBound)):
            if lowerBound[i] < 0:
                lowerBound[i] = 0

        sigma = np.sqrt(self.__In_si[:n])
        lowerBoundIn = np.array(self.__In_mean[:n]) - 2 * sigma
        upperBoundIn = np.array(self.__In_mean[:n]) + 2 * sigma
        for i in range(len(lowerBoundIn)):
            if lowerBoundIn[i] < 0:
                lowerBoundIn[i] = 0
        # Create a subplot with 3 rows and 1 column
        fig, (ax1, ax2, ax3) = pl.subplots(3, 1)

        # time

        x = range(n)
        #######  PLOT latency prediction (output)
        ax1.plot(self.__out_pred_data, 'b-', markersize=5, label=u'Observations')
        ax1.plot(self.__out_mean[:n], 'r--', label=u'Prediction')
        ax1.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([lowerBound, upperBound[::-1]]),
                 #       alpha=.5, fc='b', ec='b', label='95% confidence interval')
                 alpha=.75, fc='w', ec='k', label='95% confidence interval')
        ax1.set_xlabel('$time$')
        ax1.set_ylabel('$Latency$')
        ax1.legend(loc='upper right')
        ax1.set_xlim(0, n + 1)
        ax1.grid(True)
        #### PLOT Request Rate and its predictions
        ax2.plot(self.__In_pred_data, 'b-', markersize=5, label=u'Observations')
        ax2.plot(self.__In_mean[:n], 'r--', label=u'Prediction')
        ax2.fill(np.concatenate([x, x[::-1]]),
                 np.concatenate([lowerBoundIn, upperBoundIn[::-1]]),
                 #       alpha=.5, fc='b', ec='b', label='95% confidence interval')
                 alpha=.75, fc='w', ec='k', label='95% confidence interval')

        ax2.set_xlabel('$time$')
        ax2.set_ylabel('$Request rate$')
        ax2.set_xlim(0, n + 1)
        ax2.legend(loc='upper right')
        ax2.grid(True)
        #### PLOT Cluster (mode) prediction
        ax3.scatter(x, [1] * n, marker='.', s=30, lw=0, alpha=1, c=colors[:n])
        ax3.set_xlim(0, n + 1)
        ax3.set_xlabel('$time$')
        ax3.set_ylabel('$cluster$')
        ax3.grid(True)
        # ax3.legend([1]*n,colors,loc='upper right')

        pl.show()

    def export_results(self):
        # Draw the Results
        n = len(self.__out_pred_data)
        colorArray = ['g', 'y', 'r']
        colors = [colorArray[i] for i in self.__sysModes]
        sigma = np.sqrt(self.__out_si[:n])

        lowerBound = np.array(self.__out_mean[:n]) - 2 * sigma
        upperBound = np.array(self.__out_mean[:n]) + 2 * sigma
        for i in range(len(lowerBound)):
            if lowerBound[i] < 0:
                lowerBound[i] = 0

        sigma = np.sqrt(self.__In_si[:n])
        lowerBoundIn = np.array(self.__In_mean[:n]) - 2 * sigma
        upperBoundIn = np.array(self.__In_mean[:n]) + 2 * sigma
        for i in range(len(lowerBoundIn)):
            if lowerBoundIn[i] < 0:
                lowerBoundIn[i] = 0

        # time
        l1 = self.__out_mean[:n]
        l2 = self.__out_pred_data[:n]
        l3 = lowerBound
        l4 = upperBound

        l5 = self.__In_mean[:n]
        l6 = self.__In_pred_data
        l7 = lowerBoundIn
        l8 = upperBoundIn

        l9 = colors[:n]

        l10 = self.__controlOut[:n]
        x = range(n)
        rows = zip(x, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10)
        with open('output.csv', 'w') as f:
            writer = csv.writer(f)
            writer.writerow(['time', 'Y', 'Ym', 'Yl', 'Yu', 'U', 'Um', 'Ul', 'Uu', 'Mode', 'Control'])
            writer.writerows(rows)
