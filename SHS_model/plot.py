
import csv
import matplotlib.pyplot as pl
import numpy as np

#loaddata
out_pred_data = []
out_mean = []
lowerBound = []
upperBound= []

In_pred_data= []
In_mean = []
lowerBoundIn = []
upperBoundIn = []
control_signal = []
colors =[]

fileName = 'output.csv'
with open(fileName, 'rt') as dataFile:
    reader = csv.reader(dataFile, delimiter=',')
    firstFlag = 1
    for row in reader:
        if firstFlag:
            firstFlag = 0
            continue
        dump = float(row[0])
        out_pred_data.append(float(row[1]) )
        out_mean.append(float(row[2]) )
        lowerBound.append(float(row[3]))
        upperBound.append(float(row[4]))

        In_pred_data.append(float(row[5]))
        In_mean.append(float(row[6]))
        lowerBoundIn.append(float(row[7]))
        upperBoundIn.append(float(row[8]))

        colors.append(row[9])

        control_signal.append(float(row[10]))
n = len(out_mean)

# Create a figure with 3 subplots (3 rows and 1 column)
fig, (ax1, ax2, ax3, ax4) = pl.subplots(4, 1)

# time indices

x = range(n)
#######  PLOT latency prediction (output)
ax1.plot(out_pred_data, 'b-', markersize=5, label=u'Observations')
ax1.plot(out_mean, 'r--', label=u'Prediction')
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
ax2.plot(In_pred_data, 'b-', markersize=5, label=u'Observations')
ax2.plot(In_mean, 'r--', label=u'Prediction')
ax2.fill(np.concatenate([x, x[::-1]]),
         np.concatenate([lowerBoundIn, upperBoundIn[::-1]]),
         #       alpha=.5, fc='b', ec='b', label='95% confidence interval')
         alpha=.75, fc='w', ec='k', label='95% confidence interval')

ax2.set_xlabel('$time$')
ax2.set_ylabel('$Request rate$')
ax2.set_xlim(0, n + 1)
ax2.legend(loc='upper right')
ax2.grid(True)

#### PLOT Cluster (mode) estimation
ax3.scatter(x, [1] * n, marker='.', s=30, lw=0, alpha=1, c=colors)
ax3.set_xlim(0, n + 1)
ax3.set_xlabel('$time$')
ax3.set_ylabel('$cluster$')
ax3.grid(True)

#### PLOT control output
ax4.plot(control_signal)
ax4.set_xlabel('$time$')
ax4.set_ylabel('$number of cores$')

ax4.set_xlim(0,n+1)
ax4.grid(True)

pl.show()
