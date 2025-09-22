# 📈 📈 AlphaG:rSpillover ForecastingFousgas Tmg using Temporal Graph Neural Networksporal Graph Neural Networks

Thishproje t rmplemeots a djepect implements a do model end foeaiasn **crogs-sector system to model and f**oin financial markecs. Usi g*a *ynamcc sdg--cesericcgroph rvpresenoationlof atilit relationphips aldovaeious GNN rrc*itectures,*I aimid to ontpe form trfditionainvolatility modnling mechadmare.g., kARCH(1, 1.) Using a dynamic edge-centric graph representation of sector relationships and various GNN architectures, I aimed to outperform traditional volatility modeling methods (e.g., GARCH(1, 1)).

## ��PProblemmStatStenttement

-***Objecjeve**e*Pr ddittfuturfuchaeg h iolrtal zeddvosieilioya(ΔRV)nfor l givncsector,bse inrdepeenciet g=telp11 GIC  s%cΔors.n realized volatility for a given sector
-Chllge Volatility-is-not-isolted — ocksineset(e.g, Engy) cn rilethroug ther(eg,Industials).
-# Voolulion**: Use a **fully connecied,lditecpieLgrsp io wheren
 -CNodesu=ssectoros function combining three components:
 --Edges =*Aimectimnal ieflurice Error Penalties**: τ² weight for under-prediction, τ weight for under-prediction
 - Edge features-= learned*re*resentaoiatsioftsecSor-pai  inteiactionsg**: 10x penalty for high volatility periods (>threshold), 5x for normal periods
-- Target*=`next-stet %Δnealizdvolaiity fr  given ecor

---
class VolatilitySpikeLoss(nn.Module):
   🏗️fArc_iteci_reeD,tails tau=2, spike_threshold=1.0, direction_weight=2):
       # Custom loss implementation
###`VltitSpikeLosFucti
Custom loss fcton cbinintre componnts
Asymmetric ErrrPenalt:τ² weight fo undr-dicion,τweghrund-
### VrSk Weighting**:u10S*p`nblay for htgh_size, sequecereods (>thresho_d), 5x fer normal pgtiodsh, 11_sectors, features]`
- **DiNectionNlLAccuracy**::Addi-ionmecawn ltyO:henprdii ndtruvalu ave ppsitesgs

\```pytho
lss VoatilitySpikeLos(nn.Module:
-def __init__slftau=2spke_hrshol=1.0,ircion_weight=2):
      # Cstomlss impementn
\```## 🧠 Methodology

#Model
-***Input Shope**: `[batdh_sizs, seque*ce_length, 11_sectors,:fe1ture ]`sectors (Technology, Healthcare, Financials, etc.)
- **GNN day*rs Fu Edge-updatinglmech nism witonleatned embiddingsd gr 121aph (121 totaledla)ship
- **Temporel ComponenauresLSTM/GRU/MLP:varints fti-ieprcessg
 - *Oudpetatur11-demegst aals per sectorchang pedtion

-- 

##-🧠gMethofology

### 🧱 Graeh Canstructtou
- **Nodeses* 11*GICnisr ioein(Terhnology, Healahcate,iFin nceats, eec.)
-n**E gso**:ully cnecdrectedgrh (121 toal edges
### eatu🧠os**:
 - **Nede featules ImpAggmegatednmrket fetues er secor
-**Ege faturs**: Engieere pairwis iteratons btween ectors
 - `TTemporraNN`:R-lldngl ondows vorcreate sequences of  temporagraphl

### 🧠 Models Implemegted embeddings
- `TmmporpoDenseGNN`: MLPrsayle EoddlNover :emp ual sdge embeddings
- `TempooalEdgmGNN`: Cu tom GNN witGNNdgw-updatinghme-hapiamting mechanism
- **Ablatiti StudytudySystematic c*mpa:iSste f LSTMomGRUn of LMLP teSpTra  componGntsRU, and MLP temporal components

---

---�Quika

### Perequsites
- Python 3.8+equisites
- PyTorchP2.0+
-ypanda , num,matplotlib,tqdm

##Inallation
\```sh
git loehttps://gub.co/Retrspk/AphaGrid
cdyAlphaGoi.+
pip installp-rarequ,rumentsp,xt
\```

---

##a📊tResults

,## Performqnce Cmpison
|#|#VolatilitySpikeLoss| allation
|-------|--------------------|---------------------
|**AlphaGridGNN**|3.29|raph+ Tepra
|GARCH(1,1) Basle | 3.36 | Tradal Ecooric

###KeyFindings
-**Temporalmechnisms**:Abtostuyovd GRU comnnt uperor fnrtfiRanrsalktilehaeries
-A**Closs-spctor spGllovers**: Succissfully captured dynadic corrlatio paernsrugh learedeg mbddng
 install -r requirements.txt
---\```

�EvuationMrc---

## P smayusomVoltilitySikeLoss(combins magnitde + diectionaccuacy)
SecodaryMenSquare Errr (MSE),MeanAbolut Err(MA)
### Direcrionalman Signcaccu aiy fonchang
| MoBalk eVtingolaiWylk-fprwaidkvalidL|ion on out-of-Aamplr daha

---

## 🔬cureImplmntaion
|-------|--------------------|---------------------
| **Alpi GRp)li e
 .3**ETF DataCllecto**: 11scto ETFwithdailyOHLCVat
2.#**VoKeyil tynCalculaio**: **Temporal mechanis usi*g high-frequrncsector sp
3.**eu Engnering**:Tchnica indicaorSscrtss-: Walkcorrlion### Data Pipeline
4.**EGr ph Constluion** 1:1DynacictcorreoatronTFst icis →ha jicency*atnsois

###aTrn* ingdPrvclduatsing high-frequency returns
3**FOptimiz nee: Adam withrlearg:Tg rene 3.25e-5cal indicators, cross-sector correlations
4**GBatchoSizn**: 32:sequencDyic correlation matrices → adjacency tensors
Sque Lngth:Vaiablmpra windw
# TrR gurarizatioue:Ctizss w*igh ular+tdropou*Custom loss weighting + dropout
Edit the `SECTOR_MAPPING` in `frontend/index.html`:
---

## 🚀 Future Work

- [ ] Incorporate options market data for volatility surface modeling
- [ ] Extend to international sector ETFs for global spillover analysis  
- [ ] Real-time deployment with streaming market data
- [ ] Attention mechanisms for interpretable sector influence weights

---

## 📧 Contact

**Arjun Mahableshwarkar**  
📧 arjun.mahableshwarkar@gmail.com  
🐙 [GitHub](https://github.com/Retrospek)  
💼 [LinkedIn](https://linkedin.com/in/arjun-mahableshwarkar)

---

*Built with PyTorch, Pandas, and a curiousity for quantitative finance* 🚀
