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
        
    def analyze_batch(self,batch_value,batch_prediction):
        self.truths.append(batch_value['truth'])
        self.gen.append(batch_value['gen'])
        self.scores.append(batch_prediction)
        self.pt.append(batch_value['globalvars'][:,0])
        
    def concat(self):
        if type(self.truths)==type(list()):
            self.truths = np.concatenate(self.truths,axis=0)
            self.gen = np.concatenate(self.gen,axis=0)
            self.scores = np.concatenate(self.scores,axis=0)
            self.pt = np.concatenate(self.pt,axis=0)
            
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
        
        
