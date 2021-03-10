import numpy as np
import scipy
import sklearn
import math
import h5py
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class PerformanceMonitor():
    def __init__(self, featureDict):
        self.featureDict = featureDict
        self.scores = []
        self.truths = []
        self.gen = []
        self.pt = []
        
        self.scores_class_da = []
        self.scores_domain_da = []
        self.truth_domain_da = []
        self.xsecweight_da = []
        self.gen_da = []
        self.pt_da = []
        
    def analyze_batch(self,batch_value,batch_class_prediction):
        self.truths.append(batch_value['truth'])
        self.gen.append(batch_value['gen'])
        self.scores.append(batch_class_prediction)
        self.pt.append(batch_value['globalvars'][:,0])
        
    def analyze_batch_da(self,batch_value,batch_class_prediction,batch_domain_prediction,batch_gen):
        self.scores_class_da.append(batch_class_prediction)
        self.scores_domain_da.append(batch_domain_prediction)
        self.gen_da.append(batch_gen[:,0])
        self.truth_domain_da.append(batch_value['truth'][:,0])
        self.xsecweight_da.append(batch_value['xsecweight'][:,0])
        self.pt_da.append(batch_value['globalvars'][:,0])
        
        
    def concat(self):
        if type(self.pt)==type(list()) and len(self.pt)>0:
            self.truths = np.concatenate(self.truths,axis=0)
            self.gen = np.concatenate(self.gen,axis=0)
            self.scores = np.concatenate(self.scores,axis=0)
            self.pt = np.concatenate(self.pt,axis=0)
            
        if type(self.pt_da)==type(list()) and len(self.pt_da)>0:
            self.scores_class_da = np.concatenate(self.scores_class_da,axis=0)
            self.scores_domain_da = np.concatenate(self.scores_domain_da,axis=0)
            self.truth_domain_da = np.concatenate(self.truth_domain_da,axis=0)
            self.xsecweight_da = np.concatenate(self.xsecweight_da,axis=0)
            self.pt_da = np.concatenate(self.pt_da,axis=0)
            self.gen_da = np.concatenate(self.gen_da,axis=0)
        
    def plot(self,classNames,path):
        self.concat()
        
        plt.figure(figsize=[4.4*len(classNames),3.7*len(classNames)],dpi=120)
        index = 0
        for ibkg, bkg in enumerate(classNames):
            for isig, sig in enumerate(classNames):
                index+=1
                if ibkg==isig:
                    continue
                selectBkg = self.truths[:,ibkg]>0.5
                selectSignal = self.truths[:,isig]>0.5
                bkgScore = self.scores[selectBkg][:,isig]
                signalScore = self.scores[selectSignal][:,isig]
                plt.subplot(
                    len(classNames),len(classNames), index, 
                    ylabel='Normalized events', 
                    xlabel=sig+' discriminant'
                )
                              
                plt.hist(signalScore, bins=np.linspace(0,1,51), linewidth=2, histtype='step',  density=True, color='darkorange',label=sig)
                plt.hist(bkgScore, bins=np.linspace(0,1,51), linewidth=2, histtype='step',  density=True, color='blue',label=bkg)
                plt.xlim([0.0, 1.0])
                #plt.yscale('log')
                #plt.title('ROC curve')
                plt.grid(b=True,which='both',axis='both',linestyle='--')
                plt.legend(loc="upper center")
        plt.tight_layout()
        plt.savefig(path+"_discriminant.pdf",format='pdf')
        plt.close()
                
                
                
        
        plt.figure(figsize=[4.4*len(classNames),3.7*len(classNames)],dpi=120)
        index = 0
        
        aucs = np.zeros((len(classNames),len(classNames)))
        eff = np.zeros((len(classNames),len(classNames)))
        
        for ibkg, bkg in enumerate(classNames):
            for isig, sig in enumerate(classNames):
                index+=1
                if ibkg==isig:
                    continue
                    
                selectBkg = self.truths[:,ibkg]>0.5
                selectSignal = self.truths[:,isig]>0.5
                bkgScore = self.scores[selectBkg][:,isig]
                signalScore = self.scores[selectSignal][:,isig]
                fpr,tpr,thres = sklearn.metrics.roc_curve(
                    np.concatenate([np.zeros(bkgScore.shape),np.ones(signalScore.shape)],axis=0),
                    np.concatenate([bkgScore,signalScore],axis=0),
                    pos_label=1,
                    drop_intermediate=True
                )
                effIndex = np.argmin(np.abs(fpr-1e-3))
                if math.fabs(fpr[effIndex]-1e-3)<1e-4:
                    eff[isig,ibkg] = tpr[effIndex]
                else:
                    eff[isig,ibkg] = 0.
                
                auc = sklearn.metrics.auc(fpr, tpr)
                aucs[isig,ibkg] = auc
                #logging.info("AUC "+bkg+"/"+sig+": %.4f"%(auc))
                plt.subplot(
                    len(classNames),len(classNames), index, 
                    ylabel=bkg+' efficiency', 
                    xlabel=sig+' efficiency (AUC: %.2f%%)'%(100.*auc)
                )
                              
                plt.plot(tpr, fpr, color='darkorange')
                plt.xlim([0.0, 1.0])
                plt.ylim([1e-4, 1.0])
                plt.yscale('log')
                #plt.title('ROC curve')
                plt.grid(b=True,which='both',axis='both',linestyle='--')
                
        plt.tight_layout()
        plt.savefig(path+"_roc.pdf",format='pdf')
        plt.close()
        
        
        confusionMatrix = sklearn.metrics.confusion_matrix(
            np.argmax(self.truths,axis=1), 
            np.argmax(self.scores,axis=1), 
            #normalize='true'
        )
        confusionMatrix = 1.*confusionMatrix
        normSum = np.sum(confusionMatrix,axis=1)
        for itruth in range(len(classNames)):
            for ipred in range(len(classNames)):
                confusionMatrix[itruth,ipred]/=normSum[itruth]
                
                
        fig, ax = plt.subplots()
        ax.pcolor(confusionMatrix)

        # We want to show all ticks...
        ax.set_xticks(0.5+np.arange(len(classNames)))
        ax.set_yticks(0.5+np.arange(len(classNames)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(classNames)
        ax.set_yticklabels(classNames)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Truth')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(classNames)):
            for j in range(len(classNames)):
                text = ax.text(0.5+j, 0.5+i, "%.1f%%"%(100.*confusionMatrix[i, j]),
                               ha="center", va="center", color="w",fontsize=8)

        fig.tight_layout()
        fig.savefig(path+"_confusion.pdf",format='pdf')
        plt.close()
        
        
        fig, ax = plt.subplots()
        ax.pcolor(aucs)

        # We want to show all ticks...
        ax.set_xticks(0.5+np.arange(len(classNames)))
        ax.set_yticks(0.5+np.arange(len(classNames)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(classNames)
        ax.set_yticklabels(classNames)

        ax.set_xlabel('Background class')
        ax.set_ylabel('Signal class / score')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(classNames)):
            for j in range(len(classNames)):
                text = ax.text(0.5+j, 0.5+i, "%.1f%%"%(100.*aucs[i, j]),
                               ha="center", va="center", color="black",fontsize=8)

        fig.tight_layout()
        fig.savefig(path+"_auc.pdf",format='pdf')
        plt.close()
        
        
        
        
        
        fig, ax = plt.subplots()
        ax.pcolor(eff)

        # We want to show all ticks...
        ax.set_xticks(0.5+np.arange(len(classNames)))
        ax.set_yticks(0.5+np.arange(len(classNames)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(classNames)
        ax.set_yticklabels(classNames)

        ax.set_xlabel('Background class')
        ax.set_ylabel('Signal efficiency @ 0.1% bkg.')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(classNames)):
            for j in range(len(classNames)):
                text = ax.text(0.5+j, 0.5+i, "%.1f%%"%(100.*eff[i, j]),
                               ha="center", va="center", color="white",fontsize=8)

        fig.tight_layout()
        fig.savefig(path+"_eff.pdf",format='pdf')
        plt.close()
        
        
    def plotDA(self,classNames,path):
        self.concat()
        
        def ks_w2(data1, data2, wei1, wei2):
            ix1 = np.argsort(data1)
            ix2 = np.argsort(data2)
            data1 = data1[ix1]
            data2 = data2[ix2]
            wei1 = wei1[ix1]
            wei2 = wei2[ix2]
            data = np.concatenate([data1, data2])
            cwei1 = np.hstack([0, np.cumsum(wei1)/sum(wei1)])
            cwei2 = np.hstack([0, np.cumsum(wei2)/sum(wei2)])
            cdf1we = cwei1[np.searchsorted(data1, data, side='right')]
            cdf2we = cwei2[np.searchsorted(data2, data, side='right')]
            
            ks = np.max(np.abs(cdf1we - cdf2we))
            #p = 2*np.exp(-ks**2*(2*len(data1)/(1+len(data1)/len(data2))))
            return ks#,p
            
        def chi2(mcBins,dataBins):
            chi2_values = np.zeros(100)
            for i in range(len(chi2_values)):
                mcBinsSmeared = np.random.normal(mcBins,np.sqrt(np.abs(mcBins)))
                chi2_values[i] = np.mean(np.square(mcBinsSmeared[dataBins>0]-dataBins[dataBins>0])/(dataBins[dataBins>0]))
            return np.mean(chi2_values),np.std(chi2_values)
            
        gen_bins = np.linspace(-2,2,5)
        gen_values = 0.5*(gen_bins[:-1]+gen_bins[1:])
        ngen = len(gen_values)
        
        plt.figure(figsize=[4.4*len(classNames),3.7*ngen],dpi=120)
        index = 0
        
        for igen, gen in enumerate(range(ngen)):
            for iclass, className in enumerate(classNames):
                index+=1

                selectGen = (self.gen_da>gen_bins[igen])*(self.gen_da<gen_bins[igen+1])
                selectMC = (self.truth_domain_da<0.5)*selectGen
                selectData = (self.truth_domain_da>0.5)*selectGen
                
                mcScore = self.scores_class_da[selectMC][:,iclass]
                dataScore = self.scores_class_da[selectData][:,iclass]
                
                mcWeight = self.xsecweight_da[selectMC]
                
                if dataScore.shape[0]>0 and mcScore.shape[0]>0:
                    ks= ks_w2(mcScore,dataScore,mcWeight,np.ones(dataScore.shape[0]))
                else:
                    ks = 0
                    
                plt.subplot(
                    ngen,len(classNames), index, 
                    ylabel='Normalized events', 
                    xlabel=className+' discriminant [%.0f < $\\log_{10}(L_{xy} / 1\\mathrm{mm}$) < %.0f]'%(gen_bins[igen],gen_bins[igen+1])
                )
                binsMC,_,_ = plt.hist(mcScore, bins=np.linspace(0,1,21), weights=mcWeight, linewidth=2, histtype='step',  density=True, color='darkorange',label='MC')
                binsData,_,_ = plt.hist(dataScore, bins=np.linspace(0,1,21), linewidth=2, histtype='step',linestyle='--',  density=True, color='blue',label='Data')
                
                #chi2_value,chi2_std = chi2(binsMC,binsData)
                
                plt.plot([], [], ' ', label="KS = %.3f"%(ks))
                #plt.plot([], [], ' ', label="$\\chi^{2}$/ndof = %.2f$\\pm$%.2f"%(chi2_value,chi2_std))
                plt.xlim([0.0, 1.0])
                plt.yscale('log')

                plt.grid(b=True,which='both',axis='both',linestyle='--')
                plt.legend(loc="upper center")
        plt.tight_layout()
        plt.savefig(path+"_discriminant_bins.pdf",format='pdf')
        plt.close()
                
                
        plt.figure(figsize=[4.4*len(classNames),3.7*ngen],dpi=120)
        index = 0
        
        for igen, gen in enumerate(range(ngen)):
            for iclass, className in enumerate(classNames):
                index+=1

                selectGen = (self.gen_da>gen_bins[igen])*(self.gen_da<gen_bins[igen+1])
                selectMC = (self.truth_domain_da<0.5)*selectGen
                selectData = (self.truth_domain_da>0.5)*selectGen
                
                mcScore = self.scores_class_da[selectMC][:,iclass]
                dataScore = self.scores_class_da[selectData][:,iclass]
 
                mcWeight = self.xsecweight_da[selectMC]
                
                ks= ks_w2(mcScore,dataScore,mcWeight,np.ones(dataScore.shape[0]))
                
                plt.subplot(
                    ngen,len(classNames), index, 
                    ylabel='Normalized events', 
                    xlabel=className+' percentiles [%.0f < $\\log_{10}(L_{xy} / 1\\mathrm{mm}$) < %.0f]'%(gen_bins[igen],gen_bins[igen+1])
                )
                #use data here since unweighted
                binning = np.percentile(dataScore,np.linspace(0,100,21))
                #capture potential under/overflow
                binning[0] = 0.
                binning[-1] = 1.
                binsMC,_ = np.histogram(mcScore, bins=binning,weights=mcWeight,density=True)
                binsData,_ = np.histogram(dataScore, bins=binning,density=True)
                
                plt.plot(np.linspace(0,100,20), binsMC, linewidth=2,color='darkorange',label='MC')
                plt.plot(np.linspace(0,100,20), binsData, linewidth=2, linestyle='--', color='blue',label='Data')
                
                #chi2_value,chi2_std = chi2(binsMC,binsData)
                
                plt.plot([], [], ' ', label="KS = %.3f"%(ks))
                #plt.plot([], [], ' ', label="$\\chi^{2}$/ndof = %.2f$\\pm$%.2f"%(chi2_value,chi2_std))
                plt.xlim([0.0, 100.0])
                #plt.yscale('log')

                plt.grid(b=True,which='both',axis='both',linestyle='--')
                plt.legend(loc="upper center")
        plt.tight_layout()
        plt.savefig(path+"_discriminant_percentiles.pdf",format='pdf')
        plt.close()
                
        '''
        
        plt.figure(figsize=[4.4*len(classNames),3.7*len(classNames)],dpi=120)
        index = 0
        
        aucs = np.zeros((len(classNames),len(classNames)))
        eff = np.zeros((len(classNames),len(classNames)))
        
        for ibkg, bkg in enumerate(classNames):
            for isig, sig in enumerate(classNames):
                index+=1
                if ibkg==isig:
                    continue
                    
                selectBkg = self.truths[:,ibkg]>0.5
                selectSignal = self.truths[:,isig]>0.5
                bkgScore = self.scores[selectBkg][:,isig]
                signalScore = self.scores[selectSignal][:,isig]
                fpr,tpr,thres = sklearn.metrics.roc_curve(
                    np.concatenate([np.zeros(bkgScore.shape),np.ones(signalScore.shape)],axis=0),
                    np.concatenate([bkgScore,signalScore],axis=0),
                    pos_label=1,
                    drop_intermediate=True
                )
                effIndex = np.argmin(np.abs(fpr-1e-3))
                if math.fabs(fpr[effIndex]-1e-3)<1e-4:
                    eff[isig,ibkg] = tpr[effIndex]
                else:
                    eff[isig,ibkg] = 0.
                
                auc = sklearn.metrics.auc(fpr, tpr)
                aucs[isig,ibkg] = auc
                #logging.info("AUC "+bkg+"/"+sig+": %.4f"%(auc))
                plt.subplot(
                    len(classNames),len(classNames), index, 
                    ylabel=bkg+' efficiency', 
                    xlabel=sig+' efficiency (AUC: %.2f%%)'%(100.*auc)
                )
                              
                plt.plot(tpr, fpr, color='darkorange')
                plt.xlim([0.0, 1.0])
                plt.ylim([1e-4, 1.0])
                plt.yscale('log')
                #plt.title('ROC curve')
                plt.grid(b=True,which='both',axis='both',linestyle='--')
                
        plt.tight_layout()
        plt.savefig(path+"_roc.pdf",format='pdf')
        plt.close()
        
        
        confusionMatrix = sklearn.metrics.confusion_matrix(
            np.argmax(self.truths,axis=1), 
            np.argmax(self.scores,axis=1), 
            #normalize='true'
        )
        confusionMatrix = 1.*confusionMatrix
        normSum = np.sum(confusionMatrix,axis=1)
        for itruth in range(len(classNames)):
            for ipred in range(len(classNames)):
                confusionMatrix[itruth,ipred]/=normSum[itruth]
                
                
        fig, ax = plt.subplots()
        ax.pcolor(confusionMatrix)

        # We want to show all ticks...
        ax.set_xticks(0.5+np.arange(len(classNames)))
        ax.set_yticks(0.5+np.arange(len(classNames)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(classNames)
        ax.set_yticklabels(classNames)

        ax.set_xlabel('Predicted')
        ax.set_ylabel('Truth')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(classNames)):
            for j in range(len(classNames)):
                text = ax.text(0.5+j, 0.5+i, "%.1f%%"%(100.*confusionMatrix[i, j]),
                               ha="center", va="center", color="w",fontsize=8)

        fig.tight_layout()
        fig.savefig(path+"_confusion.pdf",format='pdf')
        plt.close()
        
        
        fig, ax = plt.subplots()
        ax.pcolor(aucs)

        # We want to show all ticks...
        ax.set_xticks(0.5+np.arange(len(classNames)))
        ax.set_yticks(0.5+np.arange(len(classNames)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(classNames)
        ax.set_yticklabels(classNames)

        ax.set_xlabel('Background class')
        ax.set_ylabel('Signal class / score')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(classNames)):
            for j in range(len(classNames)):
                text = ax.text(0.5+j, 0.5+i, "%.1f%%"%(100.*aucs[i, j]),
                               ha="center", va="center", color="black",fontsize=8)

        fig.tight_layout()
        fig.savefig(path+"_auc.pdf",format='pdf')
        plt.close()
        
        
        
        
        
        fig, ax = plt.subplots()
        ax.pcolor(eff)

        # We want to show all ticks...
        ax.set_xticks(0.5+np.arange(len(classNames)))
        ax.set_yticks(0.5+np.arange(len(classNames)))
        # ... and label them with the respective list entries
        ax.set_xticklabels(classNames)
        ax.set_yticklabels(classNames)

        ax.set_xlabel('Background class')
        ax.set_ylabel('Signal efficiency @ 0.1% bkg.')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for i in range(len(classNames)):
            for j in range(len(classNames)):
                text = ax.text(0.5+j, 0.5+i, "%.1f%%"%(100.*eff[i, j]),
                               ha="center", va="center", color="white",fontsize=8)

        fig.tight_layout()
        fig.savefig(path+"_eff.pdf",format='pdf')
        plt.close()
        '''
         
    ''' 
    def auc(self):
        self.concat()
        fpr,tpr,thres = sklearn.metrics.roc_curve(self.truths[:,5],self.scores[:,5],pos_label=1,drop_intermediate=True)
        return sklearn.metrics.auc(fpr, tpr)
    '''
    def save(self,path):
        self.concat()
        outputFile = h5py.File(path,"w")
        outputFile.create_dataset('truth',data=self.truths)
        outputFile.create_dataset('gen',data=self.gen)
        outputFile.create_dataset('score',data=self.scores)
        outputFile.create_dataset('pt',data=self.pt)
        
        '''
        fpr,tpr,thres = sklearn.metrics.roc_curve(self.truths[:,5],self.scores[:,5],pos_label=1,drop_intermediate=True)
        outputFile.create_dataset('fpr',data=fpr)
        outputFile.create_dataset('tpr',data=tpr)
        outputFile.create_dataset('thres',data=thres)
        
        for ptRange in [
            [10,20],
            [20,30],
            [30,35],
            [35,50],
            [50,75],
            [75,100]
        ]:
            select = (self.pt>ptRange[0])*(self.pt<ptRange[1])
            fpr,tpr,thres = sklearn.metrics.roc_curve(self.truths[select][:,5],self.scores[select][:,5],pos_label=1,drop_intermediate=True)
            group = outputFile.create_group("pt%i-%i"%(ptRange[0],ptRange[1]))
            group.create_dataset('fpr',data=fpr)
            group.create_dataset('tpr',data=tpr)
            group.create_dataset('thres',data=thres)
        '''
        outputFile.close()
        
        
