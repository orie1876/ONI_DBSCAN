import numpy as np
import pandas as pd
import os
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

pathList = []

eps_threshold=1.0
minimum_locs_threshold=20.0

Filename='experiment-1_sample-1-2_posXY0_channels_t0_posZ0_localizations.csv'   # This is the name of the SR file containing the localisations.

# Paths to analyse below:

pathList.append(r"/Users/Mathew/Dropbox (Cambridge University)/Ed Code/ONI code/sample-1-2/pos_0/")


for path in pathList:
    os.chdir(path)
    fit = pd.read_table(Filename,sep=',')
    fitcopy=pd.read_table(Filename,sep=',')
    F = np.array(zip(fit['X (pix)'],fit['Y (pix)']))
    try:
        db = DBSCAN(eps=eps_threshold, min_samples=minimum_locs_threshold).fit(F)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if-1 in labels else 0)  # This is to calculate the number of clusters.
        print('Estimated number of clusters: %d' % n_clusters_)
        
        fit['Cluster'] = labels
        fit.to_csv(path + '/' + 'Results_clustered_eps_'+str(eps_threshold)+'_locs_'+str(minimum_locs_threshold)+'.csv', sep = '\t')
    except ValueError:
        pass
    

    # This is to delete the rows which are not part of a cluster:
    fit_truncated=fit.copy()
    
    toDelete = fit[ fit['Cluster'] == -1 ].index
        
    
    
    
   
    fit_truncated = fit[['Frame', 'X raw (pix)','Y raw (pix)', 'Photons', 'PSF Sigma Y (pix)','PSF Sigma Y (pix)','Background','Photons','Z raw (pix)','X (pix)', 'Y (pix)','PSF Sigma X (pix)','PSF Sigma Y (pix)','X precision (nm)']].copy()
    fit_truncated['X raw (pix)'] = fit_truncated['X raw (pix)'].astype(int)
    fit_truncated['Y raw (pix)'] = fit_truncated['Y raw (pix)'].astype(int)
    
    fit_new=fit_truncated.copy()
  
    fit_truncated.drop(toDelete , inplace=True) # This deletes rows
    #fit_truncated.drop(fit_truncated.columns[[0,7,16]],axis=1,inplace=True) # This drops the columns that aren't required for GDSCSMLM 
   # fit.drop(fit.columns[[0,7,16]],axis=1,inplace=True)
    
   
    Out=open(path+'/'+'Clustered_FitResults_eps_'+str(eps_threshold)+'_locs_'+str(minimum_locs_threshold)+'.txt','w')   # Open file for writing to. 
    
    
    # Write the header of the file
    Out.write("""#Localisation Results File
#FileVersion Text.D0.E0.V2
#Name Clustered (LSE)
#Source <gdsc.smlm.ij.IJImageSource><name>Clustered</name><width>428</width><height>684</height><frames>200</frames><singleFrame>0</singleFrame><extraFrames>0</extraFrames><path></path></gdsc.smlm.ij.IJImageSource>
#Bounds x0 y0 w428 h684
#Calibration <gdsc.smlm.results.Calibration><nmPerPixel>117.0</nmPerPixel><gain>55.5</gain><exposureTime>50.0</exposureTime><readNoise>0.0</readNoise><bias>0.0</bias><emCCD>false</emCCD><amplification>0.0</amplification></gdsc.smlm.results.Calibration>
#Configuration <gdsc.smlm.engine.FitEngineConfiguration><fitConfiguration><fitCriteria>LEAST_SQUARED_ERROR</fitCriteria><delta>1.0E-4</delta><initialAngle>0.0</initialAngle><initialSD0>2.0</initialSD0><initialSD1>2.0</initialSD1><computeDeviations>false</computeDeviations><fitSolver>LVM</fitSolver><minIterations>0</minIterations><maxIterations>20</maxIterations><significantDigits>5</significantDigits><fitFunction>CIRCULAR</fitFunction><flags>20</flags><backgroundFitting>true</backgroundFitting><notSignalFitting>false</notSignalFitting><coordinateShift>4.0</coordinateShift><shiftFactor>2.0</shiftFactor><fitRegion>0</fitRegion><coordinateOffset>0.5</coordinateOffset><signalThreshold>0.0</signalThreshold><signalStrength>30.0</signalStrength><minPhotons>0.0</minPhotons><precisionThreshold>400.0</precisionThreshold><precisionUsingBackground>true</precisionUsingBackground><nmPerPixel>117.0</nmPerPixel><gain>55.5</gain><emCCD>false</emCCD><modelCamera>false</modelCamera><noise>0.0</noise><minWidthFactor>0.5</minWidthFactor><widthFactor>1.01</widthFactor><fitValidation>true</fitValidation><lambda>10.0</lambda><computeResiduals>false</computeResiduals><duplicateDistance>0.5</duplicateDistance><bias>0.0</bias><readNoise>0.0</readNoise><amplification>0.0</amplification><maxFunctionEvaluations>2000</maxFunctionEvaluations><searchMethod>POWELL_BOUNDED</searchMethod><gradientLineMinimisation>false</gradientLineMinimisation><relativeThreshold>1.0E-6</relativeThreshold><absoluteThreshold>1.0E-16</absoluteThreshold></fitConfiguration><search>2.5</search><border>1.0</border><fitting>3.0</fitting><failuresLimit>10</failuresLimit><includeNeighbours>true</includeNeighbours><neighbourHeightThreshold>0.3</neighbourHeightThreshold><residualsThreshold>1.0</residualsThreshold><noiseMethod>QUICK_RESIDUALS_LEAST_MEAN_OF_SQUARES</noiseMethod><dataFilterType>SINGLE</dataFilterType><smooth><double>0.5</double></smooth><dataFilter><gdsc.smlm.engine.DataFilter>MEAN</gdsc.smlm.engine.DataFilter></dataFilter></gdsc.smlm.engine.FitEngineConfiguration>
#Frame	origX	origY	origValue	Error	Noise	Background	Signal	Angle	X	Y	X SD	Y SD	Precision

    """)
    Out.write(fit_truncated.to_csv(sep = '\t',header=False,index=False))    # Write the columns that are required (without the non-clustered localisations)
    
    
    Out.close() # Close the file.
    
    
    Out_nc=open(path+'/'+'FitResults_withheader.txt','w')   # Open file for writing to.
        # Write the header of the file
    Out_nc.write("""#Localisation Results File
#FileVersion Text.D0.E0.V2
#Name Clustered (LSE)
#Source <gdsc.smlm.ij.IJImageSource><name>Clustered</name><width>428</width><height>684</height><frames>200</frames><singleFrame>0</singleFrame><extraFrames>0</extraFrames><path></path></gdsc.smlm.ij.IJImageSource>
#Bounds x0 y0 w428 h684
#Calibration <gdsc.smlm.results.Calibration><nmPerPixel>103.0</nmPerPixel><gain>55.5</gain><exposureTime>50.0</exposureTime><readNoise>0.0</readNoise><bias>0.0</bias><emCCD>false</emCCD><amplification>0.0</amplification></gdsc.smlm.results.Calibration>
#Configuration <gdsc.smlm.engine.FitEngineConfiguration><fitConfiguration><fitCriteria>LEAST_SQUARED_ERROR</fitCriteria><delta>1.0E-4</delta><initialAngle>0.0</initialAngle><initialSD0>2.0</initialSD0><initialSD1>2.0</initialSD1><computeDeviations>false</computeDeviations><fitSolver>LVM</fitSolver><minIterations>0</minIterations><maxIterations>20</maxIterations><significantDigits>5</significantDigits><fitFunction>CIRCULAR</fitFunction><flags>20</flags><backgroundFitting>true</backgroundFitting><notSignalFitting>false</notSignalFitting><coordinateShift>4.0</coordinateShift><shiftFactor>2.0</shiftFactor><fitRegion>0</fitRegion><coordinateOffset>0.5</coordinateOffset><signalThreshold>0.0</signalThreshold><signalStrength>30.0</signalStrength><minPhotons>0.0</minPhotons><precisionThreshold>400.0</precisionThreshold><precisionUsingBackground>true</precisionUsingBackground><nmPerPixel>117.0</nmPerPixel><gain>55.5</gain><emCCD>false</emCCD><modelCamera>false</modelCamera><noise>0.0</noise><minWidthFactor>0.5</minWidthFactor><widthFactor>1.01</widthFactor><fitValidation>true</fitValidation><lambda>10.0</lambda><computeResiduals>false</computeResiduals><duplicateDistance>0.5</duplicateDistance><bias>0.0</bias><readNoise>0.0</readNoise><amplification>0.0</amplification><maxFunctionEvaluations>2000</maxFunctionEvaluations><searchMethod>POWELL_BOUNDED</searchMethod><gradientLineMinimisation>false</gradientLineMinimisation><relativeThreshold>1.0E-6</relativeThreshold><absoluteThreshold>1.0E-16</absoluteThreshold></fitConfiguration><search>2.5</search><border>1.0</border><fitting>3.0</fitting><failuresLimit>10</failuresLimit><includeNeighbours>true</includeNeighbours><neighbourHeightThreshold>0.3</neighbourHeightThreshold><residualsThreshold>1.0</residualsThreshold><noiseMethod>QUICK_RESIDUALS_LEAST_MEAN_OF_SQUARES</noiseMethod><dataFilterType>SINGLE</dataFilterType><smooth><double>0.5</double></smooth><dataFilter><gdsc.smlm.engine.DataFilter>MEAN</gdsc.smlm.engine.DataFilter></dataFilter></gdsc.smlm.engine.FitEngineConfiguration>
#Frame	origX	origY	origValue	Error	Noise	Background	Signal	Angle	X	Y	X SD	Y SD	Precision

    """)
    Out_nc.write(fit_new.to_csv(sep = '\t',header=False,index=False))    # Write the columns that are required (without the non-clustered localisations)
    
    
    Out_nc.close() # Close the file.


    # Histogram of precisions
    precision=fit_truncated['X precision (nm)']
    plt.hist(precision, bins = 40,range=[0,40], rwidth=0.9,color='#607c8e')
    plt.xlabel('Precision (nm)')
    plt.ylabel('Number of Localisations')
    plt.title('Histogram of Precision')
    plt.show()
    
    
    # Calculate how many localisations per cluster
    clusters=labels.tolist()    # Need to convert the dataframe into a list- so that we can use the count() function. 
    maximum=max(labels)+1       # This is the last cluster number- +1 as the loop goes to <end. 
    cluster_contents=[]         # Make a list to store the number of clusters in
    
    for i in range(0,maximum):
        n=clusters.count(i)     # Count the number of times that the cluster number i is observed
        cluster_contents.append(n)  # Add to the list. 
        
    plt.hist(cluster_contents, bins = 20,range=[0,200], rwidth=0.9,color='#607c8e') # Plot a histogram. 
    plt.xlabel('Precision (nm)')
    plt.xlabel('Localisations per cluster')
    plt.ylabel('Number of clusters')
    plt.title('Histogram of cluster size')
    plt.show()
    