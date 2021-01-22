import ROOT
import random

class HistPlotter():
    class Group():
        def __init__(
            self,
            histPlotter,
            label="",
            lineColor=ROOT.kBlack,
            fillColor=ROOT.kBlack,
            markerColor=ROOT.kBlack,   
            lineWidth=1,
            fillStyle=0,
            markerStyle=0,
            markerSize=0,
            drawOptions='L'
        ):
            self.histPlotter = histPlotter
            self.label = label
            self.hist = ROOT.TH1F(
                self.histPlotter.name+str(random.random()),
                ";"+self.histPlotter.xaxis+";"+self.histPlotter.yaxis,
                len(self.histPlotter.bins)-1,
                self.histPlotter.bins
            )
            self.hist.SetLineColor(lineColor)
            self.hist.SetFillColor(fillColor)
            self.hist.SetMarkerColor(markerColor)
            self.hist.SetLineWidth(lineWidth)
            self.hist.SetFillStyle(fillStyle)
            self.hist.SetMarkerStyle(markerStyle)
            self.hist.SetMarkerSize(markerSize)
            
            self.drawOptions = drawOptions
            
        def fillValue(self,value,weight=1.):
            self.hist.Fill(value,weight)
        
        def fillBatch(
            self, 
            batch, 
            valueFct = lambda batchRow: batchRow, 
            selectFct = lambda batchRow: 1.,
            weight = 1.
        ):
            for i in range(value.shape[0]):
                if selectFct(batch[i])>0.5:
                    value = valueFct(batch[i])
                    self.hist.Fill(value,weight)

    def __init__(self,name,bins,xaxis,yaxis,stacked=False):
        self.name = name
        self.bins = bins
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.stacked = stacked
        
        self.groups = []
        self.groupDict = {}
        
    def makeGroup(self,name,*args,**kwargs):
        group = HistPlotter.Group(self,*args,**kwargs)
        self.groups.append(group)
        self.groupDict[name] = group
        return group
        
    def __getitem__(self,k):
        return self.groupDict[k]
        
    def save(self,filePath):
        pass
        
    
