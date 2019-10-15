import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from scipy.linalg import inv
from scipy.signal import find_peaks


def between(arr,low,high):
    arr2=[]
    for i in arr:
        if(i>low and i<=high):
            arr2.append(i)
    return len(arr2)
def lineerArr(arr,start,step,stepCount):
    arr2=[[],[]]
    for i in range(start,start+step*stepCount,step):
        arr2[0].append(i)
        arr2[1].append(between(arr,i,i+step))
    return arr2
def logarithmicArr(arr,start,step,stepCount,realStart):
    arr2=[[],[]]
    for i in np.arange(start,start+step*stepCount,step):

        arr2[0].append(i)
        arr2[1].append(between(arr,realStart+pow(10,i),realStart+pow(10,(i+step))))
    return arr2
def getAnnualExceedance(arr,groupSize):
  arr2=[[0],[]]
  count=0
  localMax=0
  year=0
  for i in range(len(arr)):
    if arr[i]>localMax:
      localMax=arr[i]
      arr2[0][year]=i
    if count>groupSize:
      arr2[1].append(localMax)
      year=year+1
      arr2[0].append(0)
      localMax=0
      count=0
    count=count+1
  return arr2

data = pd.read_csv('flowdata.txt',sep="	",skiprows=30, header = [1,2])
data['datetime'] = pd.to_datetime(data['datetime'].values.ravel(),format='%Y-%m-%d %H:%M')
time_series=data['66190_00060']
dates=data['datetime']

peaks = find_peaks(time_series.values.ravel(), height=1000,distance=192)[0]

peakValues=(time_series.values.ravel()[peaks])
peakValues.sort()

print('peaks with 1000')
for i in peakValues:
  print(i)


peaks2 = find_peaks(time_series.values.ravel(), height=700,distance=192)[0]

peakValues2=(time_series.values.ravel()[peaks2])
peakValues2.sort()


print('peaks with 700')
for i in peakValues2:
  print(i)

aepeaks = getAnnualExceedance(time_series.values.ravel(),35040)[0]

aepeakValues=(time_series.values.ravel()[aepeaks])
aepeakValues.sort()

print('peaks with Annual Excedence')
for i in aepeakValues:
  print(i)

linArr=lineerArr(peakValues,1000,40,50)
logArr=logarithmicArr(peakValues,0,0.1,40,1000)

aelinArr=lineerArr(aepeakValues,1000,40,50)
aelogArr=logarithmicArr(aepeakValues,0,0.1,40,1000)

plt.close('all')

data.plot(x=2,y=4)
plt.plot(dates.values.ravel()[peaks],time_series.values.ravel()[peaks], 'x')
plt.plot(np.ones_like(time_series.values.ravel())*1000, "--", color="gray")

data.plot(x=2,y=4)
plt.plot(dates.values.ravel()[peaks2],time_series.values.ravel()[peaks2], 'x')
plt.plot(np.ones_like(time_series.values.ravel())*700, "--", color="gray")

data.plot(x=2,y=4)
plt.plot(dates.values.ravel()[aepeaks],time_series.values.ravel()[aepeaks], 'x')



#plt.figure()
#plt.plot(linArr[0],linArr[1], '.-')
#
#
#tickValues=[[],[]]
#for i in np.arange(0,5):
#        tickValues[0].append(i)
#        tickValues[1].append(pow(10,i))
#
#plt.figure()
#plt.plot(logArr[0],logArr[1], '.-')
#plt.xticks(tickValues[0],tickValues[1])
#
#
plt.show(block=True)
