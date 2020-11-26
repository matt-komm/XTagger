#include "TFile.h"
#include "TTree.h"
#include "TChain.h"
#include "TTreeFormula.h"
#include "TH2F.h"
#include "TRandom3.h"
#include "TLorentzVector.h"

#include <fstream>
#include <iostream>
#include <vector>
#include <memory>
#include <unordered_map>
#include <string>
#include <random>
#include <algorithm>
#include <regex>

#include "cmdParser.hpp"
#include "exprtk.hpp"

class Feature
{
    public:
        enum Type
        {
            Float,
            Int,
            UInt,
            Double,
            Bool,
            Short,
            Char
        };

    protected:
        std::string name_;
        Type type_;

    public:
        Feature(const std::string name, const Type& type = Float):
            name_(name),
            type_(type)
        {
        }

        inline std::string name() const
        {
            return name_;
        }

        inline std::string rootTypeId() const
        {
            //https://root.cern.ch/doc/master/classTTree.html
            switch (type_)
            {
                case Float: return "F";
                case Int: return "I";
                case UInt: return "i";
                case Double: return "D";
                case Bool: return "O";
                case Short: return "S";
                case Char: return "C";
            }
        }

        inline Type type() const
        {
            return type_;
        }
};

class BranchData
{
    public:
        virtual void setFloat(size_t index, float value) = 0;
        virtual float getFloat(size_t index) = 0;
        template<class TYPE, size_t N> static std::shared_ptr<BranchData> makeBranch(TTree* tree, const std::string& branchName, const std::string& branchType, long bufferSize);
        template<class TYPE, size_t N> static std::shared_ptr<BranchData> branchAddress(TTree* tree, const std::string& branchName);

        virtual float mean() const = 0;
        virtual float std() const = 0;
        virtual float min() const = 0;
        virtual float max() const = 0;

        template<size_t N> static std::shared_ptr<BranchData> makeBranch(Feature::Type type, TTree* tree, const std::string& branchName, const std::string& branchType, long bufferSize)
        {
            switch (type)
            {
                case Feature::Float: return makeBranch<float,N>(tree,branchName,branchType,bufferSize);
                case Feature::Int: return makeBranch<int,N>(tree,branchName,branchType,bufferSize);
                case Feature::UInt: return makeBranch<unsigned int,N>(tree,branchName,branchType,bufferSize);
                case Feature::Double: return makeBranch<double,N>(tree,branchName,branchType,bufferSize);
                case Feature::Bool: return makeBranch<bool,N>(tree,branchName,branchType,bufferSize);
                case Feature::Short: return makeBranch<short,N>(tree,branchName,branchType,bufferSize);
                case Feature::Char: return makeBranch<char,N>(tree,branchName,branchType,bufferSize);
            }
        }

        template<size_t N> static std::shared_ptr<BranchData> branchAddress(Feature::Type type, TTree* tree, const std::string& branchName)
        {
            switch (type)
            {
                case Feature::Float: return branchAddress<float,N>(tree,branchName);
                case Feature::Int: return branchAddress<int,N>(tree,branchName);
                case Feature::UInt: return branchAddress<unsigned int,N>(tree,branchName);
                case Feature::Double: return branchAddress<double,N>(tree,branchName);
                case Feature::Bool: return branchAddress<bool,N>(tree,branchName);
                case Feature::Short: return branchAddress<short,N>(tree,branchName);
                case Feature::Char: return branchAddress<char,N>(tree,branchName);
            }
        }
};

template<class TYPE, size_t N>
class BranchDataTmpl:
    public BranchData
{
    protected:
        TYPE buffer_[N];
        float mean_;
        float mean2_;
        float max_;
        float min_;
        int n_;
    public:
        BranchDataTmpl():
            mean_(0),
            mean2_(0),
            max_(-1e32),
            min_(1e32),
            n_(0)
        {
        }

        virtual float mean() const
        {
            return mean_/n_;
        }

        virtual float std() const
        {
            return std::sqrt(mean2_/n_-mean_*mean_/n_/n_);
        }

        virtual float min() const
        {
            return min_;
        }

        virtual float max() const
        {
            return max_;
        }

        virtual void setFloat(size_t index, float value)
        {
            buffer_[index] = TYPE(value);
        }

        virtual float getFloat(size_t index)
        {
            n_+=1;
            mean_+=buffer_[index];
            mean2_+=buffer_[index]*buffer_[index];
            min_ = std::min<float>(min_,buffer_[index]);
            max_ = std::max<float>(max_,buffer_[index]);
            return buffer_[index];
        }

        inline TYPE* buffer()
        {
            return buffer_;
        }
};

//scalar specialization
template<class TYPE>
class BranchDataTmpl<TYPE,0>:
    public BranchData
{
    protected:
        TYPE buffer_;
        float mean_;
        float mean2_;
        float max_;
        float min_;
        int n_;

    public:
        BranchDataTmpl():
            mean_(0),
            mean2_(0),
            max_(-1e32),
            min_(1e32),
            n_(0)
        {
        }

        virtual float mean() const
        {
            return mean_/n_;
        }

        virtual float std() const
        {
            return std::sqrt(mean2_/n_-mean_*mean_/n_/n_);
        }

        virtual float min() const
        {
            return min_;
        }

        virtual float max() const
        {
            return max_;
        }

        virtual void setFloat(size_t index, float value)
        {

            buffer_ = TYPE(value);
        }
        virtual float getFloat(size_t index)
        {
            n_+=1;
            mean_+=buffer_;
            mean2_+=buffer_*buffer_;
            min_ = std::min<float>(min_,buffer_);
            max_ = std::max<float>(max_,buffer_);
            return buffer_;
        }

        inline TYPE* buffer()
        {
            return &buffer_;
        }
};



template<class TYPE, size_t N>
std::shared_ptr<BranchData> BranchData::makeBranch(TTree* tree, const std::string& branchName, const std::string& branchType, long bufferSize)
{
    std::shared_ptr<BranchDataTmpl<TYPE,N>> branchData(new BranchDataTmpl<TYPE,N>());
    tree->Branch(branchName.c_str(),branchData->buffer(),branchType.c_str(),bufferSize);
    return branchData;
}

template<class TYPE, size_t N>
std::shared_ptr<BranchData> BranchData::branchAddress(TTree* tree, const std::string& branchName)
{
    std::shared_ptr<BranchDataTmpl<TYPE,N>> branchData(new BranchDataTmpl<TYPE,N>());
    tree->SetBranchAddress(branchName.c_str(),branchData->buffer());
    return branchData;
}

static const std::vector<Feature> globalFeatures{
    Feature("global_pt"),
    Feature("global_eta"),
    Feature("global_phi"),
    Feature("global_mass"),
    Feature("global_energy"),
    
    Feature("global_area"),

    Feature("global_beta"),
    Feature("global_dR2Mean"),
    Feature("global_frac01"),
    Feature("global_frac02"),
    Feature("global_frac03"),
    Feature("global_frac04"),

    Feature("global_jetR"),
    Feature("global_jetRchg"),

    Feature("global_n60",Feature::Int),
    Feature("global_n90",Feature::Int),
    
    Feature("global_chargedEmEnergyFraction"),
    Feature("global_chargedHadronEnergyFraction"),
    Feature("global_chargedMuEnergyFraction"),
    Feature("global_electronEnergyFraction"),

    Feature("global_tau1"),
    Feature("global_tau2"),
    Feature("global_tau3"),
    
    Feature("global_relMassDropMassAK"),
    Feature("global_relMassDropMassCA"),
    Feature("global_relSoftDropMassAK"),
    Feature("global_relSoftDropMassCA"),
    
    Feature("global_thrust"),
    Feature("global_sphericity"),
    Feature("global_circularity"),
    Feature("global_isotropy"),
    Feature("global_eventShapeC"),
    Feature("global_eventShapeD"),
    
    Feature("global_numberCpf",Feature::Int),
    Feature("global_numberNpf",Feature::Int),
    Feature("global_numberSv",Feature::Int),
    Feature("global_numberSvAdapted",Feature::Int),
    Feature("global_numberMuon",Feature::Int),
    Feature("global_numberElectron",Feature::Int)
};

static const std::vector<Feature> csvFeatures{
    Feature("csv_trackSumJetEtRatio"),
    Feature("csv_trackSumJetDeltaR"),
    Feature("csv_vertexCategory",Feature::Int),
    Feature("csv_trackSip2dValAboveCharm"),
    Feature("csv_trackSip2dSigAboveCharm"),
    Feature("csv_trackSip3dValAboveCharm"),
    Feature("csv_trackSip3dSigAboveCharm"),
    Feature("csv_jetNTracksEtaRel",Feature::Int),
    Feature("csv_jetNSelectedTracks",Feature::Int)
};


static const std::vector<Feature> cpfFeatures{
    Feature("cpf_ptrel"),
    Feature("cpf_deta"),
    Feature("cpf_dphi"),
    Feature("cpf_deltaR"),
    
    Feature("cpf_px"),
    Feature("cpf_py"),
    Feature("cpf_pz"),

    Feature("cpf_trackEtaRel"),
    Feature("cpf_trackPtRel"),
    Feature("cpf_trackPPar"),
    Feature("cpf_trackDeltaR"),
    Feature("cpf_trackPParRatio"),
    Feature("cpf_trackPtRatio"),
    Feature("cpf_trackSip2dVal"),
    Feature("cpf_trackSip2dSig"),
    Feature("cpf_trackSip3dVal"),
    Feature("cpf_trackSip3dSig"),
    Feature("cpf_trackJetDistVal"),
    Feature("cpf_trackJetDistSig"),
    Feature("cpf_drminsv"),
    Feature("cpf_vertex_association",Feature::Int),
    Feature("cpf_fromPV"),
    Feature("cpf_puppi_weight"),
    Feature("cpf_track_chi2"),
    Feature("cpf_track_quality"),
    Feature("cpf_track_numberOfValidPixelHits",Feature::Int),
    Feature("cpf_track_pixelLayersWithMeasurement",Feature::Int),
    Feature("cpf_track_numberOfValidStripHits",Feature::Int),
    Feature("cpf_track_stripLayersWithMeasurement",Feature::Int),
    Feature("cpf_relmassdrop"),
    
    Feature("cpf_trackSip2dValSV"),
    Feature("cpf_trackSip2dSigSV"),
    Feature("cpf_trackSip3dValSV"),
    Feature("cpf_trackSip3dSigSV"),

    Feature("cpf_trackSip2dValSV_adapted"),
    Feature("cpf_trackSip2dSigSV_adapted"),
    Feature("cpf_trackSip3dValSV_adapted"),
    Feature("cpf_trackSip3dSigSV_adapted"),

    Feature("cpf_matchedMuon",Feature::Int),
    Feature("cpf_matchedElectron",Feature::Int),
    Feature("cpf_matchedSV",Feature::Int),
    Feature("cpf_matchedSV_adapted",Feature::Int),
    Feature("cpf_track_ndof",Feature::Int),

    Feature("cpf_dZmin")
};

static const std::vector<Feature> npfFeatures{
    Feature("npf_ptrel"),
    Feature("npf_deta"),
    Feature("npf_dphi"),
    Feature("npf_deltaR"),
    
    Feature("npf_px"),
    Feature("npf_py"),
    Feature("npf_pz"),
    
    Feature("npf_isGamma",Feature::Int),
    Feature("npf_hcal_fraction"),
    Feature("npf_drminsv"),
    Feature("npf_puppi_weight"),
    Feature("npf_relmassdrop")
};


static const std::vector<Feature> svFeatures{
    Feature("sv_ptrel"),
    Feature("sv_deta"),
    Feature("sv_dphi"),
    Feature("sv_deltaR"),
    Feature("sv_mass"),
    Feature("sv_ntracks",Feature::Int),
    Feature("sv_chi2"),
    Feature("sv_ndof",Feature::Int),
    Feature("sv_dxy"),
    Feature("sv_dxysig"),
    Feature("sv_d3d"),
    Feature("sv_d3dsig"),
    Feature("sv_costhetasvpv"),
    Feature("sv_enratio"),
    Feature("sv_vx"),
    Feature("sv_vy"),
    Feature("sv_vz")
};

static const std::vector<Feature> muonFeatures{
    Feature("muon_isGlobal",Feature::Int),
    Feature("muon_isTight",Feature::Int),
    Feature("muon_isMedium",Feature::Int),
    Feature("muon_isLoose",Feature::Int),
    Feature("muon_isStandAlone",Feature::Int),

    Feature("muon_ptrel"),
    Feature("muon_deta"),
    Feature("muon_dphi"),
    Feature("muon_px"),
    Feature("muon_py"),
    Feature("muon_pz"),
    Feature("muon_charge"),
    Feature("muon_energy"),
    Feature("muon_et"),
    Feature("muon_deltaR"),
    Feature("muon_numberOfMatchedStations",Feature::Int),

    Feature("muon_IP2d"),
    Feature("muon_IP2dSig"),
    Feature("muon_IP3d"),
    Feature("muon_IP3dSig"),

    Feature("muon_EtaRel"),
    Feature("muon_dxy"),
    Feature("muon_dxyError"),
    Feature("muon_dxySig"),
    Feature("muon_dz"),
    Feature("muon_dzError"),
    Feature("muon_dzSig"),
    Feature("muon_numberOfValidPixelHits",Feature::Int),
    Feature("muon_numberOfpixelLayersWithMeasurement",Feature::Int),
    Feature("muon_numberOfstripLayersWithMeasurement",Feature::Int), //that does not help. needs to be discussed.

    Feature("muon_chi2"),
    Feature("muon_ndof",Feature::Int),

    Feature("muon_caloIso"),
    Feature("muon_ecalIso"),
    Feature("muon_hcalIso"),

    Feature("muon_sumPfChHadronPt"),
    Feature("muon_sumPfNeuHadronEt"),
    Feature("muon_Pfpileup"),
    Feature("muon_sumPfPhotonEt"),

    Feature("muon_sumPfChHadronPt03"),
    Feature("muon_sumPfNeuHadronEt03"),
    Feature("muon_Pfpileup03"),
    Feature("muon_sumPfPhotonEt03"),


    Feature("muon_timeAtIpInOut"),
    Feature("muon_timeAtIpInOutErr"),
    Feature("muon_timeAtIpOutIn")
};

static const std::vector<Feature> electronFeatures{
    Feature("electron_ptrel"),
    Feature("electron_deltaR"),
    Feature("electron_deta"),
    Feature("electron_dphi"),
    Feature("electron_px"),
    Feature("electron_py"),
    Feature("electron_pz"),
    Feature("electron_charge"),
    Feature("electron_energy"),
    Feature("electron_EtFromCaloEn"),
    Feature("electron_isEB",Feature::Int), 
    Feature("electron_isEE",Feature::Int),
    Feature("electron_ecalEnergy"),
    Feature("electron_isPassConversionVeto",Feature::Int),
    Feature("electron_convDist"),
    Feature("electron_convFlags",Feature::Int),
    Feature("electron_convRadius"),
    Feature("electron_hadronicOverEm"),
    Feature("electron_ecalDrivenSeed"),
    Feature("electron_IP2d"),
    Feature("electron_IP2dSig"),
    Feature("electron_IP3d"),
    Feature("electron_IP3dSig"),

    Feature("electron_elecSC_energy"),
    Feature("electron_elecSC_deta"),
    Feature("electron_elecSC_dphi"),
    Feature("electron_elecSC_et"),
    Feature("electron_elecSC_eSuperClusterOverP"),
    Feature("electron_scPixCharge"),
    Feature("electron_superClusterFbrem"),

    Feature("electron_eSeedClusterOverP"),
    Feature("electron_eSeedClusterOverPout"),
    Feature("electron_eSuperClusterOverP"),

    // shower shape
    Feature("electron_sigmaEtaEta"),
    Feature("electron_sigmaIetaIeta"),
    Feature("electron_sigmaIphiIphi"),
    Feature("electron_e5x5"),
    Feature("electron_e5x5Rel"),
    Feature("electron_e1x5Overe5x5"),
    Feature("electron_e2x5MaxOvere5x5"),
    Feature("electron_r9"),
    Feature("electron_hcalOverEcal"),
    Feature("electron_hcalDepth1OverEcal"),
    Feature("electron_hcalDepth2OverEcal"),

    // Track-Cluster Matching Attributes
    Feature("electron_deltaEtaEleClusterTrackAtCalo"),
    Feature("electron_deltaEtaSeedClusterTrackAtCalo"),
    Feature("electron_deltaPhiSeedClusterTrackAtCalo"), 
    Feature("electron_deltaEtaSeedClusterTrackAtVtx"),
    Feature("electron_deltaEtaSuperClusterTrackAtVtx"),
    Feature("electron_deltaPhiEleClusterTrackAtCalo"),
    Feature("electron_deltaPhiSuperClusterTrackAtVtx"),

    Feature("electron_sCseedEta"),

    // electron gsf variables. 
    Feature("electron_EtaRel"),
    Feature("electron_dxy"),
    Feature("electron_dxyError"),
    Feature("electron_dxySig"),
    Feature("electron_dz"),
    Feature("electron_dzError"),
    Feature("electron_dzSig"),
    Feature("electron_nbOfMissingHits",Feature::Int),
    Feature("electron_gsfCharge"),
    Feature("electron_ndof",Feature::Int),
    Feature("electron_chi2"),
    Feature("electron_numberOfBrems",Feature::Int),
    Feature("electron_fbrem"),

    // Isolation block
    Feature("electron_neutralHadronIso"),
    Feature("electron_particleIso"),
    Feature("electron_photonIso"),
    Feature("electron_puChargedHadronIso"),
    Feature("electron_trackIso"),
    Feature("electron_ecalPFClusterIso"),
    Feature("electron_hcalPFClusterIso"),
    
    Feature("electron_pfSumPhotonEt"),
    Feature("electron_pfSumChargedHadronPt"), 
    Feature("electron_pfSumNeutralHadronEt"),
    Feature("electron_pfSumPUPt"),

    Feature("electron_dr04TkSumPt"),
    Feature("electron_dr04EcalRecHitSumEt"),
    Feature("electron_dr04HcalDepth1TowerSumEt"),
    Feature("electron_dr04HcalDepth1TowerSumEtBc"),
    Feature("electron_dr04HcalDepth2TowerSumEt"),
    Feature("electron_dr04HcalDepth2TowerSumEtBc"),
    Feature("electron_dr04HcalTowerSumEt"),
    Feature("electron_dr04HcalTowerSumEtBc")
};


constexpr static size_t nClasses = 40;
static const std::vector<Feature> jetLabels{
    Feature("jetorigin_isPrompt_E",Feature::Int),
    Feature("jetorigin_isPrompt_MU",Feature::Int),
    Feature("jetorigin_isPrompt_TAU",Feature::Int),
    Feature("jetorigin_isPrompt_PHOTON",Feature::Int),

    Feature("jetorigin_isPU",Feature::Int),
    Feature("jetorigin_isB",Feature::Int),
    Feature("jetorigin_isBB",Feature::Int),
    Feature("jetorigin_isLeptonic_B",Feature::Int),
    Feature("jetorigin_isC",Feature::Int),
    Feature("jetorigin_isCC",Feature::Int),
    Feature("jetorigin_isLeptonic_C",Feature::Int),
    Feature("jetorigin_isS",Feature::Int),
    Feature("jetorigin_isUD",Feature::Int),
    Feature("jetorigin_isG",Feature::Int),

    //Include LLP flavour :
    Feature("jetorigin_isLLP_RAD",Feature::Int), //no flavour match (likely from wide angle radiation)
    Feature("jetorigin_isLLP_Q",Feature::Int),
    Feature("jetorigin_isLLP_QQ",Feature::Int),

    Feature("jetorigin_isLLP_B",Feature::Int),
    Feature("jetorigin_isLLP_BB",Feature::Int),

    Feature("jetorigin_isLLP_MU",Feature::Int), //prompt lepton
    Feature("jetorigin_isLLP_QMU",Feature::Int),
    Feature("jetorigin_isLLP_QQMU",Feature::Int),

    Feature("jetorigin_isLLP_BMU",Feature::Int),
    Feature("jetorigin_isLLP_BBMU",Feature::Int),

    Feature("jetorigin_isLLP_E",Feature::Int), //prompt lepton
    Feature("jetorigin_isLLP_QE",Feature::Int),
    Feature("jetorigin_isLLP_QQE",Feature::Int),

    Feature("jetorigin_isLLP_BE",Feature::Int),
    Feature("jetorigin_isLLP_BBE",Feature::Int),

    Feature("jetorigin_isLLP_TAU",Feature::Int),
    Feature("jetorigin_isLLP_QTAU",Feature::Int),
    Feature("jetorigin_isLLP_QQTAU",Feature::Int),
    Feature("jetorigin_isLLP_BTAU",Feature::Int),
    Feature("jetorigin_isLLP_BBTAU",Feature::Int),
    
    Feature("jetorigin_isLLP_PHOTON",Feature::Int),
    Feature("jetorigin_isLLP_QPHOTON",Feature::Int),
    Feature("jetorigin_isLLP_QQPHOTON",Feature::Int),
    Feature("jetorigin_isLLP_BPHOTON",Feature::Int),
    Feature("jetorigin_isLLP_BBPHOTON",Feature::Int),

    Feature("jetorigin_isUndefined",Feature::Int)
};


static const std::vector<Feature> jetoriginFeatures{
    Feature("jetorigin_tauDecay_NO_TAU",Feature::Int),
    Feature("jetorigin_tauDecay_INVISIBLE",Feature::Int),
    Feature("jetorigin_tauDecay_E",Feature::Int),
    Feature("jetorigin_tauDecay_MU",Feature::Int),
    Feature("jetorigin_tauDecay_H",Feature::Int),
    Feature("jetorigin_tauDecay_H_1PI0",Feature::Int),
    Feature("jetorigin_tauDecay_H_XPI0",Feature::Int),
    Feature("jetorigin_tauDecay_HHH",Feature::Int),
    Feature("jetorigin_tauDecay_HHH_XPI0",Feature::Int),
    
    Feature("jetorigin_partonFlavor",Feature::Int),
    Feature("jetorigin_hadronFlavor",Feature::Int),
    Feature("jetorigin_llpId",Feature::Int),
    Feature("jetorigin_llp_mass"),
    Feature("jetorigin_llp_pt"),

    Feature("jetorigin_displacement"),
    Feature("jetorigin_displacement_xy"),
    Feature("jetorigin_displacement_z"),
    Feature("jetorigin_decay_angle"),
    Feature("jetorigin_betagamma"),
        
    Feature("jetorigin_matchedGenJetDeltaR"),
    Feature("jetorigin_matchedGenJetPt"),
    Feature("jetorigin_sharedVertexFraction"),
        
    Feature("jetorigin_genTauMass"),
    Feature("jetorigin_recoTauMass")
};


class UnpackedTree
{
    public:
        const bool addTruth_;
        std::unique_ptr<TFile> outputFile_;
        TTree* tree_;


        static constexpr int maxEntries_cpf = 25;
        static constexpr int maxEntries_npf = 25;
        static constexpr int maxEntries_sv = 4;
        static constexpr int maxEntries_muon = 2;
        static constexpr int maxEntries_electron = 2;

        static constexpr int bufferSize = 64000; //default is 32kB

        float isData;
        float xsecweight;
        float processId;

        std::vector<std::shared_ptr<BranchData>> globalBranches;
        std::vector<std::shared_ptr<BranchData>> csvBranches;
        
        std::vector<std::shared_ptr<BranchData>> jetLabelBranches;
        std::vector<std::shared_ptr<BranchData>> jetoriginBranches;

        unsigned int ncpf;
        std::vector<std::shared_ptr<BranchData>> cpfBranches;

        unsigned int nnpf;
        std::vector<std::shared_ptr<BranchData>> npfBranches;

        unsigned int nsv;
        std::vector<std::shared_ptr<BranchData>> svBranches;

        unsigned int nmuon;
        std::vector<std::shared_ptr<BranchData>> muonBranches;

        unsigned int nelectron;
        std::vector<std::shared_ptr<BranchData>> electronBranches;

        template<size_t N>
        std::vector<std::shared_ptr<BranchData>> makeBranches(TTree* tree, const std::vector<Feature>& features, const std::string& lengthName="") const
        {
            std::vector<std::shared_ptr<BranchData>> branches;
            for (size_t ifeature = 0; ifeature < features.size(); ++ifeature)
            {
                auto const& feature = features[ifeature];
                auto branchData = BranchData::makeBranch<N>(
                    feature.type(),
                    tree,
                    feature.name().c_str(),
                    lengthName.size()>0 ? (feature.name()+"["+lengthName+"]/"+feature.rootTypeId()).c_str() : (feature.name()+"/"+feature.rootTypeId()).c_str(),
                    bufferSize
                );
                branches.push_back(branchData);
            }
            return branches;
        }

    public:
        UnpackedTree(const std::string& fileName, bool addTruth=true):
            addTruth_(addTruth),
            outputFile_(new TFile(fileName.c_str(),"RECREATE"))
        {
            if (not outputFile_->IsOpen())
            {
                throw std::runtime_error("Output file cannot be created: "+fileName);
            }

            tree_ = new TTree("jets","jets");
            tree_->SetDirectory(outputFile_.get());
            tree_->SetAutoSave(500); //save after 200 fills

            if (addTruth)
            {
                jetLabelBranches = makeBranches<0>(tree_,jetLabels);
                jetoriginBranches = makeBranches<0>(tree_,jetoriginFeatures);
            }
            else
            {
                tree_->Branch("xsecweight",&xsecweight,"xsecweight/F",bufferSize);
                tree_->Branch("isData",&isData,"isData/F",bufferSize);
                tree_->Branch("processId",&processId,"processId/F",bufferSize);
            }


            globalBranches = makeBranches<0>(tree_,globalFeatures);
            csvBranches = makeBranches<0>(tree_,csvFeatures);

            tree_->Branch("ncpf",&ncpf,"ncpf/I",bufferSize);
            cpfBranches = makeBranches<maxEntries_cpf>(tree_,cpfFeatures,"ncpf");

            tree_->Branch("nnpf",&nnpf,"nnpf/I",bufferSize);
            npfBranches = makeBranches<maxEntries_npf>(tree_,npfFeatures,"nnpf");

            tree_->Branch("nsv",&nsv,"nsv/I",bufferSize);
            svBranches = makeBranches<maxEntries_sv>(tree_,svFeatures,"nsv");

    	    tree_->Branch("nmuon",&nmuon,"nmuon/I",bufferSize);
            muonBranches = makeBranches<maxEntries_muon>(tree_,muonFeatures,"nmuon");

            tree_->Branch("nelectron",&nelectron,"nelectron/I",bufferSize);
            electronBranches = makeBranches<maxEntries_electron>(tree_,electronFeatures,"nelectron");

            tree_->SetBasketSize("*",bufferSize); //default is 16kB
        }

        //root does not behave properly
        UnpackedTree(UnpackedTree&&) = delete;
        UnpackedTree(const UnpackedTree&) = delete;

        ~UnpackedTree()
        {

            //Note: TTree is managed by TFile and gets deleted by ROOT when file is closed
        }


        void fill()
        {
            //outputFile_->cd();
            //tree_->SetDirectory(outputFile_);
            tree_->Fill();
        }

        void close()
        {
            outputFile_->cd();
            //tree_->SetDirectory(outputFile_);
            tree_->Write();
            outputFile_->Close();
        }
};

class NanoXTree
{
    public:
        //std::shared_ptr<TFile> file_;
        TTree* tree_;
        const bool addTruth_;

        int ientry_;

        static constexpr int maxJets = 50; //allows for a maximum of 50 jets per event
        static constexpr int maxEntries_global = maxJets;
        static constexpr int maxEntries_cpf = UnpackedTree::maxEntries_cpf*maxJets;
        static constexpr int maxEntries_npf = UnpackedTree::maxEntries_npf*maxJets;
        static constexpr int maxEntries_sv = UnpackedTree::maxEntries_sv*maxJets;
        static constexpr int maxEntries_muon = UnpackedTree::maxEntries_muon*maxJets;
        static constexpr int maxEntries_electron = UnpackedTree::maxEntries_electron*maxJets;


        unsigned int nJet;
        float Jet_eta[maxEntries_global];
        float Jet_phi[maxEntries_global];
        float Jet_pt[maxEntries_global];
        unsigned int Jet_jetId[maxEntries_global];
        unsigned int Jet_nConstituents[maxEntries_global];

        int Jet_muonIdx1[maxEntries_global];
        int Jet_muonIdx2[maxEntries_global];
        int Jet_electronIdx1[maxEntries_global];
        int Jet_electronIdx2[maxEntries_global];

        unsigned int nMuon;
        float Muon_pt[50];
        float Muon_eta[50];
        float Muon_phi[50];
        unsigned int nElectron;
        float Electron_pt[50];
        float Electron_eta[50];
        float Electron_phi[50];

        float Jet_forDA[maxEntries_global];
        int Jet_genJetIdx[maxEntries_global];

        float GenJet_pt[maxEntries_global];

        unsigned int njetorigin;
        int jetorigin_jetIdx[maxEntries_global];

        unsigned int nlength;
        int length_cpf[maxEntries_global];
        int length_npf[maxEntries_global];
        int length_sv[maxEntries_global];
        int length_muon[maxEntries_global];
        int length_electron[maxEntries_global];

        float xsecweight;
        float processId;
        float isData;


        unsigned int nglobal;
        int global_jetIdx[maxEntries_global];

        std::vector<std::shared_ptr<BranchData>> globalBranches;
        std::unordered_map<std::string,std::shared_ptr<BranchData>> globalBranchMap;
        
        std::vector<std::shared_ptr<BranchData>> jetoriginBranches;
        std::vector<std::shared_ptr<BranchData>> jetLabelBranches;
        
        std::unordered_map<std::string,std::shared_ptr<BranchData>> jetoriginBranchMap;
        std::unordered_map<std::string,std::shared_ptr<BranchData>> jetLabelBranchMap;
        std::unordered_map<std::string,float> jetPropertiesMapForSelection;

        unsigned int ncsv;
        std::vector<std::shared_ptr<BranchData>> csvBranches;

        unsigned int ncpf;
        int cpf_jetIdx[maxEntries_cpf];
        std::vector<std::shared_ptr<BranchData>> cpfBranches;

        unsigned int nnpf;
        int npf_jetIdx[maxEntries_npf];
        std::vector<std::shared_ptr<BranchData>> npfBranches;

        unsigned int nsv;
        int sv_jetIdx[maxEntries_sv];
        std::vector<std::shared_ptr<BranchData>> svBranches;

        unsigned int nmuon;
        int muon_jetIdx[maxEntries_muon];
        std::vector<std::shared_ptr<BranchData>> muonBranches;

        unsigned int nelectron;
        int electron_jetIdx[maxEntries_electron];
        std::vector<std::shared_ptr<BranchData>> electronBranches;

        std::mt19937 randomGenerator_;
        std::uniform_real_distribution<> uniform_dist_;

        typedef exprtk::symbol_table<float> SymbolTable;
        typedef exprtk::expression<float> Expression;
        typedef exprtk::parser<float> Parser;

        float isPrompt_ANY;
        float isB_ANY;
        float isC_ANY;
        float isUDSG_ANY;
        float isLLP_ANY;

        float rand;
        float pt;
        float eta;
        float phi;

        float ctau;

        Parser parser_;
        SymbolTable symbolTable_;
        std::vector<Expression> selections_;
        std::vector<Expression> setters_;


        template<size_t N>
        std::vector<std::shared_ptr<BranchData>> branchAddresses(TTree* tree, const std::vector<Feature>& features) const
        {
            std::vector<std::shared_ptr<BranchData>> branches;
            for (size_t ifeature = 0; ifeature < features.size(); ++ifeature)
            {
                auto const& feature = features[ifeature];
                auto branchData = BranchData::branchAddress<N>(
                    feature.type(),
                    tree,
                    feature.name().c_str()
                );
                branches.push_back(branchData);
            }
            return branches;
        }

    public:
        NanoXTree(
            TTree* tree,
            const std::vector<std::string>& selectors={},
            const std::vector<std::string>& setters={},
            bool addTruth=true
        ):
            tree_(tree),
            addTruth_(addTruth),
            ientry_(0),
            randomGenerator_(12345),
            uniform_dist_(0,1.)
        {
            tree_->SetBranchAddress("nJet",&nJet);
            tree_->SetBranchAddress("Jet_eta",&Jet_eta);
            tree_->SetBranchAddress("Jet_phi",&Jet_phi);
            tree_->SetBranchAddress("Jet_pt",&Jet_pt);
            tree_->SetBranchAddress("Jet_jetId",&Jet_jetId);
            tree_->SetBranchAddress("Jet_nConstituents",&Jet_nConstituents);
            tree_->SetBranchAddress("Jet_genJetIdx", &Jet_genJetIdx);
            tree_->SetBranchAddress("GenJet_pt", &GenJet_pt);

            tree_->SetBranchAddress("Jet_muonIdx1",&Jet_muonIdx1);
            tree_->SetBranchAddress("Jet_muonIdx2",&Jet_muonIdx2);
            tree_->SetBranchAddress("Jet_electronIdx1",&Jet_electronIdx1);
            tree_->SetBranchAddress("Jet_electronIdx2",&Jet_electronIdx2);

            tree_->SetBranchAddress("nMuon",&nMuon);
            tree_->SetBranchAddress("Muon_pt",&Muon_pt);
            tree_->SetBranchAddress("Muon_eta",&Muon_eta);
            tree_->SetBranchAddress("Muon_phi",&Muon_phi);

            tree_->SetBranchAddress("nElectron",&nElectron);
            tree_->SetBranchAddress("Electron_pt",&Electron_pt);
            tree_->SetBranchAddress("Electron_eta",&Electron_eta);
            tree_->SetBranchAddress("Electron_phi",&Electron_phi);

            if (addTruth)
            {
                tree_->SetBranchAddress("njetorigin",&njetorigin);
                tree_->SetBranchAddress("jetorigin_jetIdx",&jetorigin_jetIdx);
                
                jetoriginBranches = branchAddresses<maxEntries_global>(tree_,jetoriginFeatures);
                jetLabelBranches = branchAddresses<maxEntries_global>(tree_,jetLabels);
                
                for (size_t ifeature = 0; ifeature < jetLabels.size(); ++ifeature)
                {
                    jetLabelBranchMap[jetLabels[ifeature].name()] = jetLabelBranches[ifeature];
                    jetPropertiesMapForSelection[jetLabels[ifeature].name()] = 0.; //init
                }
                for (size_t ifeature = 0; ifeature < jetoriginFeatures.size(); ++ifeature)
                {
                    jetoriginBranchMap[jetoriginFeatures[ifeature].name()] = jetoriginBranches[ifeature];
                    jetPropertiesMapForSelection[jetoriginFeatures[ifeature].name()] = 0.; //init
                }
            }
            else
            {
                tree_->SetBranchAddress("Jet_forDA",&Jet_forDA);
                tree_->SetBranchAddress("xsecweight",&xsecweight);
                tree_->SetBranchAddress("isData",&isData);
                tree_->SetBranchAddress("processId",&processId);
            }

            tree_->SetBranchAddress("nlength",&nlength);
            tree_->SetBranchAddress("length_cpf",&length_cpf);
            tree_->SetBranchAddress("length_npf",&length_npf);
            tree_->SetBranchAddress("length_sv",&length_sv);
            tree_->SetBranchAddress("length_mu",&length_muon);
            tree_->SetBranchAddress("length_ele",&length_electron);

            tree_->SetBranchAddress("nglobal",&nglobal);
            tree_->SetBranchAddress("global_jetIdx",&global_jetIdx);

            globalBranches = branchAddresses<maxEntries_global>(tree_,globalFeatures);
            for (size_t ifeature = 0; ifeature < globalFeatures.size(); ++ifeature)
            {
                globalBranchMap[globalFeatures[ifeature].name()] = globalBranches[ifeature];
                jetPropertiesMapForSelection[globalFeatures[ifeature].name()] = 0.;
            }

            tree_->SetBranchAddress("ncsv",&ncsv);
            csvBranches = branchAddresses<maxEntries_global>(tree_,csvFeatures);

            tree_->SetBranchAddress("ncpf",&ncpf);
            tree_->SetBranchAddress("cpf_jetIdx",&cpf_jetIdx);
            cpfBranches = branchAddresses<maxEntries_cpf>(tree_,cpfFeatures);

            tree_->SetBranchAddress("nnpf",&nnpf);
            tree_->SetBranchAddress("npf_jetIdx",&npf_jetIdx);
            npfBranches = branchAddresses<maxEntries_npf>(tree_,npfFeatures);

            tree_->SetBranchAddress("nsv",&nsv);
            tree_->SetBranchAddress("sv_jetIdx",&sv_jetIdx);
            svBranches = branchAddresses<maxEntries_sv>(tree_,svFeatures);

		    tree_->SetBranchAddress("nmuon",&nmuon);
		    tree_->SetBranchAddress("muon_jetIdx",&muon_jetIdx);
            muonBranches = branchAddresses<maxEntries_muon>(tree_,muonFeatures);

            tree_->SetBranchAddress("nelectron",&nelectron);
            tree_->SetBranchAddress("electron_jetIdx",&electron_jetIdx);
            electronBranches = branchAddresses<maxEntries_electron>(tree_,electronFeatures);

            tree_->SetCacheSize(10);
            tree_->SetMaxVirtualSize(16000);

            for (auto& featureBranchPair: jetPropertiesMapForSelection)
            {
                symbolTable_.add_variable(featureBranchPair.first,featureBranchPair.second);
            }
            

            symbolTable_.add_variable("isPrompt_ANY",isPrompt_ANY);
            symbolTable_.add_variable("isB_ANY",isB_ANY);
            symbolTable_.add_variable("isC_ANY",isC_ANY);
            symbolTable_.add_variable("isUDSG_ANY",isUDSG_ANY);
            symbolTable_.add_variable("isLLP_ANY",isLLP_ANY);

            symbolTable_.add_variable("rand",rand);
            symbolTable_.add_variable("ctau",ctau);


            for (auto selectstring: selectors)
            {
                std::cout<<"register selection: "<<selectstring<<std::endl;
                Expression exp;
                exp.register_symbol_table(symbolTable_);
                if (not parser_.compile(selectstring,exp))
                {
                    for (std::size_t i = 0; i < parser_.error_count(); ++i)
                    {
                        auto error = parser_.get_error(i);
                        std::cout<<"Expression compilation error #"<<i<<std::endl;
                        std::cout<<" -> Position: "<<error.token.position;
                        std::cout<<", Type: "<<exprtk::parser_error::to_str(error.mode);
                        std::cout<<", Msg: "<<error.diagnostic<<std::endl<<std::endl;
                    }
                    throw std::runtime_error("Compilation error");
                }
                else
                {
                    selections_.emplace_back(std::move(exp));
                }
            }

            for (auto setstring: setters)
            {
                std::cout<<"register setter: "<<setstring<<std::endl;
                Expression exp;
                exp.register_symbol_table(symbolTable_);
                if (not parser_.compile(setstring,exp))
                {
                    for (std::size_t i = 0; i < parser_.error_count(); ++i)
                    {
                        auto error = parser_.get_error(i);
                        std::cout<<"Expression compilation error #"<<i<<std::endl;
                        std::cout<<" -> Position: "<<error.token.position;
                        std::cout<<", Type: "<<exprtk::parser_error::to_str(error.mode);
                        std::cout<<", Msg: "<<error.diagnostic<<std::endl<<std::endl;
                    }
                    throw std::runtime_error("Compilation error");
                }
                else
                {
                    setters_.emplace_back(std::move(exp));
                }

            }

        }

        //this class does not play well with memory -> prevent funny usage
        NanoXTree(NanoXTree&&) = delete;
        NanoXTree(const NanoXTree&) = delete;

        ~NanoXTree()
        {
            //file_->Close();
        }

        inline unsigned int entries() const
        {
            return tree_->GetEntries();
        }

        inline int entry() const
        {
            return ientry_;
        }

        bool getEvent(int entry, bool force=false)
        {
            if (force or entry!=ientry_)
            {
                tree_->GetEntry(entry);
                ientry_ = entry;
                return true;
            }
            if (entry>=entries())
            {
                return false;
            }
            return true;
        }

        bool nextEvent()
        {
            return getEvent((ientry_+1)%entries());
        }

        inline int njets()
        {
            return std::min<int>(nJet,maxJets);
        }

        void printInfo() const
        {
            for (size_t ifeature = 0; ifeature < globalBranches.size(); ++ifeature)
            {
                std::cout<<globalFeatures[ifeature].name()<<": "<<std::to_string(globalBranches[ifeature]->mean())<<" +- "<<std::to_string(globalBranches[ifeature]->std())<<" ["<<std::to_string(globalBranches[ifeature]->min())<<", "<<std::to_string(globalBranches[ifeature]->max())<<"]"<<std::endl;
            }

            for (size_t ifeature = 0; ifeature < csvBranches.size(); ++ifeature)
            {
                std::cout<<csvFeatures[ifeature].name()<<": "<<std::to_string(csvBranches[ifeature]->mean())<<" +- "<<std::to_string(csvBranches[ifeature]->std())<<" ["<<std::to_string(csvBranches[ifeature]->min())<<", "<<std::to_string(csvBranches[ifeature]->max())<<"]"<<std::endl;
            }

            for (size_t ifeature = 0; ifeature < cpfBranches.size(); ++ifeature)
            {
                std::cout<<cpfFeatures[ifeature].name()<<": "<<std::to_string(cpfBranches[ifeature]->mean())<<" +- "<<std::to_string(cpfBranches[ifeature]->std())<<" ["<<std::to_string(cpfBranches[ifeature]->min())<<", "<<std::to_string(cpfBranches[ifeature]->max())<<"]"<<std::endl;
            }

            for (size_t ifeature = 0; ifeature < npfBranches.size(); ++ifeature)
            {
                std::cout<<npfFeatures[ifeature].name()<<": "<<std::to_string(npfBranches[ifeature]->mean())<<" +- "<<std::to_string(npfBranches[ifeature]->std())<<" ["<<std::to_string(npfBranches[ifeature]->min())<<", "<<std::to_string(npfBranches[ifeature]->max())<<"]"<<std::endl;
            }

            for (size_t ifeature = 0; ifeature < svBranches.size(); ++ifeature)
            {
                std::cout<<svFeatures[ifeature].name()<<": "<<std::to_string(svBranches[ifeature]->mean())<<" +- "<<std::to_string(svBranches[ifeature]->std())<<" ["<<std::to_string(svBranches[ifeature]->min())<<", "<<std::to_string(svBranches[ifeature]->max())<<"]"<<std::endl;
            }

            for (size_t ifeature = 0; ifeature < muonBranches.size(); ++ifeature)
            {
                std::cout<<muonFeatures[ifeature].name()<<": "<<std::to_string(muonBranches[ifeature]->mean())<<" +- "<<std::to_string(muonBranches[ifeature]->std())<<" ["<<std::to_string(muonBranches[ifeature]->min())<<", "<<std::to_string(muonBranches[ifeature]->max())<<"]"<<std::endl;
            }

            for (size_t ifeature = 0; ifeature < electronBranches.size(); ++ifeature)
            {
                std::cout<<electronFeatures[ifeature].name()<<": "<<std::to_string(electronBranches[ifeature]->mean())<<" +- "<<std::to_string(electronBranches[ifeature]->std())<<" ["<<std::to_string(electronBranches[ifeature]->min())<<", "<<std::to_string(electronBranches[ifeature]->max())<<"]"<<std::endl;
            }
        }

        bool isSelected(unsigned int jet)
        {

            //nJet should be lower than e.g. njetorigin since pT selection on Jet's are applied
            if (jet>=nJet)
            {
                return false;
            }

            //reverse search for indices
            int indexGlobal = -1;
            int indexOrigin = -1;

            if (nglobal!=nlength)
            {
                std::cout<<"Encountered mismatch between length of candidates and global jets"<<std::endl;
                return false;
            }

            for (int ijet = 0; ijet < nglobal; ++ijet)
            {
                if (global_jetIdx[ijet]==jet) indexGlobal = ijet;
                if (addTruth_)
                {
                    if (jetorigin_jetIdx[ijet]==jet) indexOrigin = ijet;
                }
            }
            if (indexGlobal<0 or (addTruth_ and indexOrigin<0))
            {
                return false;
            }

            //at least 10 GeV uncorrected
            if (globalBranchMap["global_pt"]->getFloat(indexGlobal)<10.)
            {
                return false;
            }
            
            //just a sanity check
            if (std::fabs(Jet_eta[jet]/globalBranchMap["global_eta"]->getFloat(indexGlobal)-1)>0.01 or std::fabs(Jet_phi[jet]/globalBranchMap["global_phi"]->getFloat(indexGlobal)-1)>0.01)
            {
                std::cout<<"Encountered mismatch between standard nanoaod jets and xtag info"<<std::endl;
                return false;
            }


            //ignore jet if reco/gen pt largely disagree -> likely random PU match
            //require minimum of genjet pt of 10 GeV
            if (addTruth_ and jetLabelBranchMap["jetorigin_isPU"]->getFloat(indexOrigin)<0.5 and Jet_genJetIdx[jet]>-1 and Jet_genJetIdx[jet]<maxJets)
            {
                if ((GenJet_pt[Jet_genJetIdx[jet]]<10.) or ((Jet_pt[jet]/GenJet_pt[Jet_genJetIdx[jet]]) < 0.5))
                {
                    //std::cout << "Skipping jet with mismatched genpt: reco pt="<<Jet_pt[jet] << ", genpt="<<GenJet_pt[Jet_genJetIdx[jet]] << std::endl;
                    return false;
                }
            }

            if (this->njets()<=jet)
            {
                std::cout<<"Not enough jets to unpack"<<std::endl;
                return false;
            }

            if (std::fabs(Jet_eta[jet])>2.4)
            {
                return false;
            }

            

            if (Jet_nConstituents[jet]<2) return false;

            
            if (jetLabelBranchMap["jetorigin_isPrompt_E"]->getFloat(indexOrigin)<0.5 
                and jetLabelBranchMap["jetorigin_isPrompt_MU"]->getFloat(indexOrigin)<0.5 
                and jetLabelBranchMap["jetorigin_isPrompt_TAU"]->getFloat(indexOrigin)<0.5)
            {
                TLorentzVector jetP4(0,0,0,0);
                jetP4.SetPtEtaPhiM(Jet_pt[jet],Jet_eta[jet],Jet_phi[jet],0.);

                TLorentzVector leptonP4Sum(0,0,0,0);

                for (int imu = 0; imu < nMuon; ++imu)
                {
                    TLorentzVector leptonP4(0,0,0,0);
                    leptonP4.SetPtEtaPhiM(Muon_pt[imu],Muon_eta[imu],Muon_phi[imu],0.);
                    if (leptonP4.DeltaR(jetP4)<0.4)
                    {
                        leptonP4Sum+=leptonP4;
                    }
                }

                for (int iele = 0; iele < nElectron; ++iele)
                {
                    TLorentzVector leptonP4(0,0,0,0);
                    leptonP4.SetPtEtaPhiM(Electron_pt[iele],Electron_eta[iele],Electron_phi[iele],0.);
                    if (leptonP4.DeltaR(jetP4)<0.4)
                    {
                        leptonP4Sum+=leptonP4;
                    }
                }

                TLorentzVector jetP4Subtracted = jetP4 - leptonP4Sum;

                if (jetP4Subtracted.Pt()<10.)
                {
                    return false;
                }
            }

            if (addTruth_)
            {
                if (jetLabelBranchMap["jetorigin_isUndefined"]->getFloat(indexOrigin)>0.5)
                {
                    return false;
                }
                
                for (size_t ifeature = 0; ifeature < jetLabels.size(); ++ifeature)
                {
                    const auto name = jetLabels[ifeature].name();
                    jetPropertiesMapForSelection[name] = jetLabelBranchMap[name]->getFloat(indexOrigin);
                }
                
                for (size_t ifeature = 0; ifeature < jetoriginFeatures.size(); ++ifeature)
                {
                    const auto name = jetoriginFeatures[ifeature].name();
                    jetPropertiesMapForSelection[name] = jetoriginBranchMap[name]->getFloat(indexOrigin);
                }
                for (size_t ifeature = 0; ifeature < globalFeatures.size(); ++ifeature)
                {
                    const auto name = globalFeatures[ifeature].name();
                    jetPropertiesMapForSelection[name] = globalBranchMap[name]->getFloat(indexOrigin);
                }
                
                isPrompt_ANY = 0;
                isB_ANY = 0;
                isC_ANY = 0;
                isUDSG_ANY = 0;
                isLLP_ANY = 0;
                
                const static std::regex regexPrompt("jetorigin_isPrompt_\\S*");
                const static std::regex regexB("jetorigin_isB\\S*");
                const static std::regex regexC("jetorigin_isC\\S*");
                const static std::regex regexLLP("jetorigin_isLLP_\\S*");
                
                for (size_t ifeature = 0; ifeature < jetLabels.size(); ++ifeature)
                {
                    const auto name = jetLabels[ifeature].name();
                    float value = jetLabelBranchMap[name]->getFloat(indexOrigin);
                    jetPropertiesMapForSelection[name] = value;
                   
                    if (std::regex_match(name,regexPrompt))
                    {
                        isPrompt_ANY+=value;
                    }
                    if (std::regex_match(name,regexB) or name=="jetorigin_isLeptonic_B")
                    {
                        isB_ANY+=value;
                    }
                    if (std::regex_match(name,regexC) or name=="jetorigin_isLeptonic_C")
                    {
                        isC_ANY+=value;
                    }
                    if (name=="jetorigin_isUD" or name=="jetorigin_isS" or name=="jetorigin_isG")
                    {
                        isUDSG_ANY+=value;
                    }
                    if (std::regex_match(name,regexLLP))
                    {
                        isLLP_ANY+=value;
                    }
                }

                float isANY = isPrompt_ANY
                            +isB_ANY+isC_ANY
                            +isUDSG_ANY+jetPropertiesMapForSelection["jetorigin_isPU"]
                            +isLLP_ANY+jetPropertiesMapForSelection["jetorigin_isUndefined"];
            
                if (isANY<0.5 or isANY>1.5)
                {
                    std::cout<<"Error - label sum is not 1 (="<<isANY<<")"<<std::endl;
                    for (size_t ifeature = 0; ifeature < jetLabels.size(); ++ifeature)
                    {
                        const auto name = jetLabels[ifeature].name();
                        std::cout<<"   "<<name<<": "<<jetPropertiesMapForSelection[name]<<std::endl;
                    }
                    std::cout<<"   isPrompt_ANY: "<<isPrompt_ANY<<std::endl;
                    std::cout<<"   isB_ANY: "<<isB_ANY<<std::endl;
                    std::cout<<"   isC_ANY: "<<isC_ANY<<std::endl;
                    std::cout<<"   isUDSG_ANY: "<<isUDSG_ANY<<std::endl;
                    std::cout<<"   isLLP_ANY: "<<isLLP_ANY<<std::endl;
                    
                    
                    return false;
                }

            }
            else
            {
                if (Jet_forDA[jet]<0.5)
                {
                    return false;
                }

                for (size_t ifeature = 0; ifeature < jetoriginFeatures.size(); ++ifeature)
                {
                    const auto name = jetoriginFeatures[ifeature].name();
                    jetPropertiesMapForSelection[name] = 0;
                }
                for (size_t ifeature = 0; ifeature < jetLabels.size(); ++ifeature)
                {
                    const auto name = jetLabels[ifeature].name();
                    jetPropertiesMapForSelection[name] = 0;
                }
                for (size_t ifeature = 0; ifeature < globalFeatures.size(); ++ifeature)
                {
                    const auto name = globalFeatures[ifeature].name();
                    jetPropertiesMapForSelection[name] = globalBranchMap[name]->getFloat(indexOrigin);
                }
                isPrompt_ANY = 0;
                isB_ANY = 0;
                isC_ANY = 0;
                isUDSG_ANY = 0;
                isLLP_ANY = 0;
            }


            rand = uniform_dist_(randomGenerator_);
            ctau = -10;

            for (auto setter: setters_)
            {
                //std::cout<<ctau;
                setter.value();
                //std::cout<<" -> "<<ctau<<std::endl;
            }


            for (auto exp: selections_)
            {
                if (exp.value()<0.5)
                {
                    return false;
                }
            }

            return true;
        }

        float getJetPt(unsigned int jet)
        {
            int indexGlobal = -1;
            for (int ijet = 0; ijet < nJet; ++ijet)
            {
                if (global_jetIdx[ijet]==jet) indexGlobal = ijet;
            }
            if (indexGlobal<0) return -1;

            return globalBranchMap["global_pt"]->getFloat(indexGlobal);
        }

        int getJetClass(unsigned int jet)
        {
            if (not addTruth_) return 0; //default class

            int indexOrigin = -1;
            for (int ijet = 0; ijet < nJet; ++ijet)
            {
                if (jetorigin_jetIdx[ijet]==jet) indexOrigin = ijet;
            }

            if (indexOrigin==-1) return -1;
            
            for (size_t ifeature = 0; ifeature < jetLabelBranches.size(); ++ifeature)
            {
                if (jetLabelBranches[ifeature]->getFloat(indexOrigin)>0.5) return ifeature;
            }

            return -1;
        }

        bool unpackJet(
            unsigned int jet,
            UnpackedTree& unpackedTree
        )
        {
            if (this->njets()<jet) return false;

            //reverse search for indices
            int indexGlobal = -1;
            int indexOrigin = -1;

            for (int ijet = 0; ijet < nglobal; ++ijet)
            {
                if (global_jetIdx[ijet]==jet) indexGlobal = ijet;
                if (addTruth_)
                {
                    if (jetorigin_jetIdx[ijet]==jet) indexOrigin = ijet;
                }
            }

            if (indexGlobal<0 or (addTruth_ and indexOrigin<0))
            {
                return false;
            }

            if (addTruth_)
            {
                if (unpackedTree.jetLabelBranches.size()!=jetLabelBranches.size()) throw std::runtime_error("Jet label branches have different size! "+std::to_string(unpackedTree.jetLabelBranches.size())+"!="+std::to_string(jetLabelBranches.size()));
                for (size_t ifeature = 0; ifeature < jetLabelBranches.size(); ++ifeature)
                {
                    unpackedTree.jetLabelBranches[ifeature]->setFloat(0,jetLabelBranches[ifeature]->getFloat(indexGlobal));
                }
                
                if (unpackedTree.jetoriginBranches.size()!=jetoriginBranches.size()) throw std::runtime_error("Jet origin branches have different size! "+std::to_string(unpackedTree.jetoriginBranches.size())+"!="+std::to_string(jetoriginBranches.size()));
                for (size_t ifeature = 0; ifeature < jetoriginBranches.size(); ++ifeature)
                {
                    unpackedTree.jetoriginBranches[ifeature]->setFloat(0,jetoriginBranches[ifeature]->getFloat(indexGlobal));
                }

            }
            else
            {
                unpackedTree.isData = isData;
                unpackedTree.xsecweight = xsecweight;
                unpackedTree.processId = processId;
            }


            if (unpackedTree.globalBranches.size()!=globalBranches.size()) throw std::runtime_error("Global branches have different size! "+std::to_string(unpackedTree.globalBranches.size())+"!="+std::to_string(globalBranches.size()));
            for (size_t ifeature = 0; ifeature < globalBranches.size(); ++ifeature)
            {
                unpackedTree.globalBranches[ifeature]->setFloat(0,globalBranches[ifeature]->getFloat(indexGlobal));
            }

            if (unpackedTree.csvBranches.size()!=csvBranches.size()) throw std::runtime_error("CSV branches have different size! "+std::to_string(unpackedTree.csvBranches.size())+"!="+std::to_string(csvBranches.size()));
            for (size_t ifeature = 0; ifeature < csvBranches.size(); ++ifeature)
            {
                unpackedTree.csvBranches[ifeature]->setFloat(0,csvBranches[ifeature]->getFloat(indexGlobal));
            }


            int cpf_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                cpf_offset += length_cpf[i];
            }
            if (length_cpf[indexGlobal]>0 and jet!=cpf_jetIdx[cpf_offset])
            {
                throw std::runtime_error("CPF jet index different than global one");
            }

            int ncpf = std::min<int>(UnpackedTree::maxEntries_cpf,length_cpf[indexGlobal]);
            unpackedTree.ncpf = ncpf;

            if (unpackedTree.cpfBranches.size()!=cpfBranches.size()) throw std::runtime_error("CPF branches have different size! "+std::to_string(unpackedTree.cpfBranches.size())+"!="+std::to_string(cpfBranches.size()));

            for (size_t ifeature = 0; ifeature < cpfBranches.size(); ++ifeature)
            {
                for (int i = 0; i < ncpf; ++i)
                {
                    unpackedTree.cpfBranches[ifeature]->setFloat(i,cpfBranches[ifeature]->getFloat(cpf_offset+i));
                }
            }


            int npf_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                npf_offset += length_npf[i];
            }
            if (length_npf[indexGlobal]>0 and jet!=npf_jetIdx[npf_offset])
            {
                throw std::runtime_error("NPF jet index different than global one");
            }

            int nnpf = std::min<int>(UnpackedTree::maxEntries_npf,length_npf[indexGlobal]);
            unpackedTree.nnpf = nnpf;

            if (unpackedTree.npfBranches.size()!=npfBranches.size()) throw std::runtime_error("NPF branches have different size! "+std::to_string(unpackedTree.npfBranches.size())+"!="+std::to_string(npfBranches.size()));

            for (size_t ifeature = 0; ifeature < npfBranches.size(); ++ifeature)
            {
                for (int i = 0; i < nnpf; ++i)
                {
                    unpackedTree.npfBranches[ifeature]->setFloat(i,npfBranches[ifeature]->getFloat(npf_offset+i));
                }
            }


            int sv_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                sv_offset += length_sv[i];
            }
            if (length_sv[indexGlobal]>0 and jet!=sv_jetIdx[sv_offset])
            {
                throw std::runtime_error("SV jet index different than global one");
            }

            int nsv = std::min<int>(UnpackedTree::maxEntries_sv,length_sv[indexGlobal]);
            unpackedTree.nsv = nsv;

            if (unpackedTree.svBranches.size()!=svBranches.size()) throw std::runtime_error("SV branches have different size! "+std::to_string(unpackedTree.svBranches.size())+"!="+std::to_string(svBranches.size()));

            for (size_t ifeature = 0; ifeature < svBranches.size(); ++ifeature)
            {
                for (int i = 0; i < nsv; ++i)
                {
                    unpackedTree.svBranches[ifeature]->setFloat(i,svBranches[ifeature]->getFloat(sv_offset+i));
                }
            }


            int muon_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                muon_offset += length_muon[i];
            }
            if (length_muon[indexGlobal]>0 and jet!=muon_jetIdx[muon_offset])
            {
                throw std::runtime_error("Muon jet index different than global one");
            }

            int nmuon = std::min<int>(UnpackedTree::maxEntries_muon,length_muon[indexGlobal]);
            unpackedTree.nmuon = nmuon;

            if (unpackedTree.muonBranches.size()!=muonBranches.size()) throw std::runtime_error("Muon branches have different size! "+std::to_string(unpackedTree.muonBranches.size())+"!="+std::to_string(muonBranches.size()));
            for (size_t ifeature = 0; ifeature < muonBranches.size(); ++ifeature)
            {
                for (int i = 0; i < nmuon; ++i)
                {
                    unpackedTree.muonBranches[ifeature]->setFloat(i,muonBranches[ifeature]->getFloat(muon_offset+i));
	            }
            }
            /*
            if (nmuon == 0)
            {
                if (unpackedTree.jetorigin_isLLP_MU>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_MU = 0.0;
                    unpackedTree.jetorigin_isLLP_RAD = 1.0;
                }
                else if (unpackedTree.jetorigin_isLLP_QMU>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_QMU = 0.0;
                    unpackedTree.jetorigin_isLLP_Q = 1.0;
                }
                else if (unpackedTree.jetorigin_isLLP_QQMU>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_QQMU = 0.0;
                    unpackedTree.jetorigin_isLLP_QQ = 1.0;
                }
                else if (unpackedTree.jetorigin_isLLP_BMU>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_BMU = 0.0;
                    unpackedTree.jetorigin_isLLP_B = 1.0;
                }
                else if (unpackedTree.jetorigin_isLLP_BBMU>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_BBMU = 0.0;
                    unpackedTree.jetorigin_isLLP_BB = 1.0;
                }
            }
            */


            int electron_offset = 0;
            for (size_t i = 0; i < indexGlobal; ++i)
            {
                electron_offset += length_electron[i];
            }
            if (length_electron[indexGlobal]>0 and jet!=electron_jetIdx[electron_offset])
            {
                throw std::runtime_error("Electron jet index different than global one");
            }

            int nelectron = std::min<int>(UnpackedTree::maxEntries_electron,length_electron[indexGlobal]);
            unpackedTree.nelectron = nelectron;

            if (unpackedTree.electronBranches.size()!=electronBranches.size()) throw std::runtime_error("Electron branches have different size! "+std::to_string(unpackedTree.electronBranches.size())+"!="+std::to_string(electronBranches.size()));
            for (size_t ifeature = 0; ifeature < electronBranches.size(); ++ifeature)
            {
                for (int i = 0; i < nelectron; ++i)
                {
                    unpackedTree.electronBranches[ifeature]->setFloat(i,electronBranches[ifeature]->getFloat(electron_offset+i));
                }
            }
            /*
            if (nelectron == 0)
            {
                if (unpackedTree.jetorigin_isLLP_E>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_E = 0.0;
                    unpackedTree.jetorigin_isLLP_RAD = 1.0;
                }
                else if (unpackedTree.jetorigin_isLLP_QE>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_QE = 0.0;
                    unpackedTree.jetorigin_isLLP_Q = 1.0;
                }
                else if (unpackedTree.jetorigin_isLLP_QQE>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_QQE = 0.0;
                    unpackedTree.jetorigin_isLLP_QQ = 1.0;
                }
                else if (unpackedTree.jetorigin_isLLP_BE>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_BE = 0.0;
                    unpackedTree.jetorigin_isLLP_B = 1.0;
                }
                else if (unpackedTree.jetorigin_isLLP_BBE>0.5) 
                {
                    unpackedTree.jetorigin_isLLP_BBE = 0.0;
                    unpackedTree.jetorigin_isLLP_BB = 1.0;
                }
            }
            */
            unpackedTree.fill();
            return true;
        }
};

inline bool ends_with(std::string const & value, std::string const & ending)
{
    if (ending.size() > value.size()) return false;
    return std::equal(ending.rbegin(), ending.rend(), value.rbegin());
}

inline bool begins_with(std::string const & value, std::string const & start)
{
    if (start.size() > value.size()) return false;
    return std::equal(start.begin(), start.end(), value.begin());
}

unsigned int calcHash(unsigned int value)
{
    unsigned int hash = ((value >> 16) ^ value) * 0x45d9f3b;
    hash = ((hash >> 16) ^ hash) * 0x45d9f3b;
    hash = (hash >> 16) ^ hash;
    return hash;
}

int main(int argc, char **argv)
{
    cli::Parser parser(argc, argv);
    parser.set_required<std::string>("o", "output", "Output prefix");
  	parser.set_optional<int>("n", "number", 10, "Number of output files");
  	parser.set_optional<int>("f", "testfraction", 15, "Fraction of events for testing in percent [0-100]");
  	parser.set_optional<int>("s", "split", 1, "Number of splits for batched processing");
  	parser.set_optional<int>("b", "batch", 0, "Current batch number (<number of splits)");
  	parser.set_optional<bool>("t", "truth", true, "Add truth from jetorigin (deactivate for DA)");
    parser.set_required<std::vector<std::string>>("i", "input", "Input files");
    parser.run_and_exit_if_error();
    
    if (nClasses!=jetLabels.size()) 
    {
        std::cout<<"Error - number of classes ("<<std::to_string(nClasses)<<") does not correspond to number of labels ("<<std::to_string(jetLabels.size())<<")"<<std::endl;
        return 1;
    }
    std::string outputPrefix = parser.get<std::string>("o");
    std::cout<<"output file prefix: "<<outputPrefix<<std::endl;

    int nOutputs = parser.get<int>("n");
    std::cout<<"output files: "<<nOutputs<<std::endl;
    if (nOutputs<=0)
    {
        std::cout<<"Error: Number of output files (-n) needs to be >=1"<<std::endl;
        return 1;
    }

    int nTestFrac = parser.get<int>("f");
    std::cout<<"test fraction: "<<nTestFrac<<"%"<<std::endl;
    if (nTestFrac<0 or nTestFrac>100)
    {
        std::cout<<"Error: Test fraction needs to be within [0;100]"<<std::endl;
        return 1;
    }

    int nSplit = parser.get<int>("s");
    std::cout<<"total splits: "<<nSplit<<std::endl;
    if (nSplit<=0)
    {
        std::cout<<"Error: Total split number needs to be >=1!"<<std::endl;
        return 1;
    }

    int iSplit = parser.get<int>("b");
    std::cout<<"current split: "<<iSplit<<"/"<<nSplit<<std::endl;
    if (iSplit>=nSplit)
    {
        std::cout<<"Error: Current split number (-b) needs to be smaller than total split (-s) number!"<<std::endl;
        return 1;
    }

    bool addTruth = parser.get<bool>("t");
    std::cout<<"add truth from jetorigin: "<<(addTruth ? "true" : "false")<<std::endl;

    std::vector<std::unique_ptr<NanoXTree>> trees;
    std::cout<<"Input files: "<<std::endl;

    std::vector<int> entries;
    int total_entries = 0;

    std::vector<std::vector<std::string>> inputFileNames;
    std::vector<std::vector<std::string>> selectors;
    std::vector<std::vector<std::string>> setters;
    std::vector<int> caps;

    std::vector<std::string> inputs = parser.get<std::vector<std::string>>("i");
    if (inputs.size()==0)
    {
        std::cout<<"Error: At least one input file (-i) required!"<<std::endl;
        return 1;
    }

    for (const std::string& s: inputs)
    {
        if (ends_with(s,".root"))
        {
            inputFileNames.push_back(std::vector<std::string>{s});
            selectors.push_back(std::vector<std::string>{});
            setters.push_back(std::vector<std::string>{});
            caps.push_back(-1);
        }
        else if(ends_with(s,".txt"))
        {
            std::ifstream input(s);
            std::vector<std::string> files;
            std::vector<std::string> select;
            std::vector<std::string> setter;
            int cap = -1;
            for( std::string line; getline( input, line ); )
            {
                if (line.size()>0)
                {
                    if (begins_with(line,"#select"))
                    {
                        select.emplace_back(line.begin()+7,line.end());
                    }
                    else if (begins_with(line,"#set"))
                    {
                        setter.emplace_back(line.begin()+4,line.end());
                    }
                    else if (begins_with(line,"#cap"))
                    {
                        cap = atoi(std::string(line.begin()+4,line.end()).c_str());
                    }
                    else if (begins_with(line,"#"))
                    {
                        std::cout<<"Ignore unknown instruction: "<<line<<std::endl;
                    }
                    else
                    {
                        files.push_back(line);
                    }
                }
            }
            auto rng = std::default_random_engine {123456};
            std::shuffle(files.begin(), files.end(), rng);
            selectors.push_back(select);
            setters.push_back(setter);
            caps.push_back(cap);
            inputFileNames.push_back(files);
        }
        else
        {
            std::cout<<"Error: Cannot parse file '"<<s<<"'"<<std::endl;
            return 1;
        }
    }

    for (size_t i = 0; i < inputFileNames.size(); ++i)
    {
        auto inputFileNameList = inputFileNames[i];
        TChain* chain = new TChain("Events","Events");
        //int nfiles = 0;
        int totalEntries = 0;
        for (const auto& inputFileName: inputFileNameList)
        {
            //std::cout<<"   "<<argv[iarg]<<", nEvents="<<;
            TFile* file = TFile::Open(inputFileName.c_str());
            if (not file)
            {
                std::cout<<"Warning: File '"<<inputFileName<<"' cannot be read"<<std::endl;
                continue;
            }

            TTree* tree = dynamic_cast<TTree*>(file->Get("Events"));

            if (not tree)
            {
                std::cout<<"Warning: Tree in file '"<<inputFileName<<"' cannot be read"<<std::endl;
                continue;
            }
            int nEvents = tree->GetEntries();
            totalEntries += nEvents;
            std::cout<<"   "<<inputFileName<<", nEvents="<<nEvents<<std::endl;
            file->Close();
            chain->AddFile(inputFileName.c_str());

            if (caps[i]>0 and totalEntries>caps[i])
            {
                std::cout<<"   "<<inputFileName<<"number of "<<caps[i]<<" events reached"<<std::endl;
                break;
            }
            //nfiles+=1;
            //if (nfiles>1) break;
        }

        int nEvents = chain->GetEntries();
        std::cout<<"Total per chain:  "<<nEvents<<std::endl;
        entries.push_back(nEvents);
        total_entries += nEvents;
        trees.emplace_back(std::unique_ptr<NanoXTree>(new NanoXTree (chain,selectors[i],setters[i],addTruth)));
    }
    if (inputFileNames.size()==0)
    {
        std::cout<<"Error: No input files readable!"<<std::endl;
        return 1;
    }
    if (total_entries==0)
    {
        std::cout<<"Error: Total number of entries=0!"<<std::endl;
        return 1;
    }

    std::cout<<"Number of independent inputs: "<<trees.size()<<std::endl;
    std::cout<<"Total number of events: "<<total_entries<<std::endl;


    std::vector<std::unique_ptr<UnpackedTree>> unpackedTreesTrain;
    std::vector<std::vector<int>> eventsPerClassPerFileTrain(nClasses,std::vector<int>(nOutputs,0));

    std::vector<std::unique_ptr<UnpackedTree>> unpackedTreesTest;
    std::vector<std::vector<int>> eventsPerClassPerFileTest(nClasses,std::vector<int>(nOutputs,0));

    std::array<float,21> ptBins{{
          10.        ,    15.        ,    20.        ,    25.        ,
          31.05838232,    38.58492449,    47.93541347,    59.55185592,
          73.98337237,    91.91215457,   114.18571346,   141.85694176,
         176.23388528,   218.94157546,   271.99884626,   337.9137663 ,
         419.8021978 ,   521.53508632,   647.92144416,   804.93567703,
        1000.
	}};
	std::array<float,nClasses+1> classBins;
	for (size_t i = 0; i < (nClasses+1); ++i)
	{
	    classBins[i] = i-0.5;
	}
    TH2F ptSample(
        "ptSample","",
        ptBins.size()-1,ptBins.data(),
        classBins.size()-1,classBins.data()
    );

    TRandom3 rng(1234);

    for (unsigned int i = 0; i < nOutputs; ++i)
    {
        unpackedTreesTrain.emplace_back(std::unique_ptr<UnpackedTree>(
            new UnpackedTree(outputPrefix+"_train"+std::to_string(iSplit+1)+"_"+std::to_string(i+1)+".root",addTruth
        )));

        unpackedTreesTest.emplace_back(std::unique_ptr<UnpackedTree>(
            new UnpackedTree(outputPrefix+"_test"+std::to_string(iSplit+1)+"_"+std::to_string(i+1)+".root",addTruth
        )));
    }

    int eventsInBatch = int(1.*total_entries/nSplit);

    std::cout<<"Batch number of events: "<<eventsInBatch<<std::endl;

    //offset reading for each input tree
    for (size_t itree = 0; itree < trees.size(); ++itree)
    {
        trees[itree]->getEvent(int(1.*iSplit*trees[itree]->entries()/nSplit),true);
    }

    std::vector<int> readEvents(entries.size(),0);
    std::vector<int> writtenJets(entries.size(),0);
    std::vector<int> skippedJets(entries.size(),0);
    for (int ientry = 0; ientry<eventsInBatch; ++ientry)
    {
        if (ientry%10000==0)
        {
            std::cout<<"Processing ... "<<100.*ientry/eventsInBatch<<std::endl;
        }

        //choose input file pseudo-randomly
        long hash = calcHash(47*ientry+iSplit*23);
        long hashEntries = (hash+hash/eventsInBatch)%eventsInBatch;


        int sum_entries = 0;
        int ifile = 0;

        for (;ifile<(entries.size()-1); ++ifile)
        {
            sum_entries += int(1.*entries[ifile]/nSplit);
            if (hashEntries<sum_entries) break;
        }
        trees[ifile]->nextEvent(); //this loops back to 0 in case it was the last event

        readEvents[ifile]+=1;

        //take only the 6 hardest jets
        for (size_t j = 0; j < std::min<size_t>(6,trees[ifile]->njets()); ++j)
        {
            if (trees[ifile]->isSelected(j))
            {
                int jet_class = trees[ifile]->getJetClass(j);
                float jet_pt = trees[ifile]->getJetPt(j);
                float binContent = ptSample.GetBinContent(
                    ptSample.GetXaxis()->FindBin(jet_pt),
                    ptSample.GetYaxis()->FindBin(jet_class)
                );
                if (ptSample.GetEntries()>10 and ptSample.GetMaximum()<rng.Gaus(binContent+1,1))
                {
                    skippedJets[ifile]+=1;
                    continue;
                }

                if (jet_class>eventsPerClassPerFileTrain.size())
                {
                    throw std::runtime_error("Number of classes do not match");
                }
                long hashTest = calcHash(97*trees[ifile]->entry()+79*j)%100;
                if (hashTest<nTestFrac)
                {
                    if (jet_class>=0 and jet_class<eventsPerClassPerFileTest.size())
                    {
                        unsigned int ofile = std::distance(
                            eventsPerClassPerFileTest[jet_class].begin(),
                            std::min_element(
                                eventsPerClassPerFileTest[jet_class].begin(),
                                eventsPerClassPerFileTest[jet_class].end()
                            )
                        );

                        if (trees[ifile]->unpackJet(j,*unpackedTreesTest[ofile]))
                        {
                            eventsPerClassPerFileTest[jet_class][ofile]+=1;
                            writtenJets[ifile]+=1;
                            ptSample.Fill(jet_pt,jet_class);
                        }
                        else
                        {
                            skippedJets[ifile]+=1;
                        }
                    }
                }
                else
                {
                    if (jet_class>=0 and jet_class<eventsPerClassPerFileTrain.size())
                    {
                        unsigned int ofile = std::distance(
                            eventsPerClassPerFileTrain[jet_class].begin(),
                            std::min_element(
                                eventsPerClassPerFileTrain[jet_class].begin(),
                                eventsPerClassPerFileTrain[jet_class].end()
                            )
                        );
                        //std::cout<<ofile<<std::endl;
                        if (trees[ifile]->unpackJet(j,*unpackedTreesTrain[ofile]))
                        {
                            eventsPerClassPerFileTrain[jet_class][ofile]+=1;
                            writtenJets[ifile]+=1;
                            ptSample.Fill(jet_pt,jet_class);
                        }
                        else
                        {
                            skippedJets[ifile]+=1;
                        }
                    }
                }
            }
            else
            {
                skippedJets[ifile]+=1;
            }

        }
    }

    for (size_t i = 0; i < entries.size(); ++i)
    {
        std::cout<<"infile "<<inputs[i]<<":"<<std::endl;
        std::cout<<"\tevents: found = "<<entries[i]<<", read = "<<readEvents[i]<<"/"<<int(1.*entries[i]/nSplit)<<std::endl;
        std::cout<<"\tjets: written = "<<writtenJets[i]<<", skipped = "<<skippedJets[i]<<std::endl;
    }

    std::cout<<"----- Sample ----- "<<std::endl;
    std::cout<<"                    ";
    for (int ibin = 0; ibin < ptSample.GetNbinsX(); ++ibin)
    {
        printf("%7.1f ",ptSample.GetXaxis()->GetBinCenter(ibin+1));
    }
    std::cout<<std::endl;
    for (int c = 0; c < ptSample.GetNbinsY(); ++c)
    {
        printf("%18s: ",jetLabels[c].name().substr(10).c_str());
        //printf("%2i: ",c+1);
        for (int ibin = 0; ibin < ptSample.GetNbinsX(); ++ibin)
        {
            printf("%5.0f   ",ptSample.GetBinContent(ibin+1,c+1));
        }
        std::cout<<std::endl;
    }
    std::cout<<"----- Train ----- "<<std::endl;
    for (size_t c = 0; c < eventsPerClassPerFileTrain.size(); ++c)
    {
        printf("jet class: %18s: ",jetLabels[c].name().substr(10).c_str());
        //std::cout<<"jet class "<<c<<": ";
        for (size_t i = 0; i < nOutputs; ++i)
        {
            std::cout<<eventsPerClassPerFileTrain[c][i]<<", ";
        }
        std::cout<<std::endl;
    }
    std::cout<<"----- Test ----- "<<std::endl;
    for (size_t c = 0; c < eventsPerClassPerFileTest.size(); ++c)
    {
        printf("jet class: %18s: ",jetLabels[c].name().substr(10).c_str());
        //std::cout<<"jet class "<<c<<": ";
        for (size_t i = 0; i < nOutputs; ++i)
        {
            std::cout<<eventsPerClassPerFileTest[c][i]<<", ";
        }
        std::cout<<std::endl;
    }

    for (auto& unpackedTree: unpackedTreesTrain)
    {
        unpackedTree->close();
    }

    for (auto& unpackedTree: unpackedTreesTest)
    {
        unpackedTree->close();
    }

    for (size_t i = 0; i < inputs.size(); ++i)
    {
        std::cout<<"============= "<<inputs[i]<<" =============="<<std::endl;
        trees[i]->printInfo();

    }

    return 0;
}


