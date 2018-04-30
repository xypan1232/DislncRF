# DislncRF

Long non-coding RNAs (lncRNAs) are important regulators in different biological processes, which are linked to many different diseases. Compared to protein coding genes (PCGs), the association between diseases and lncRNAs is still not well studied. 
Thus, inferring disease-associated lncRNAs genome-widely becomes imperative.
In this study, we propose a machine learning based method, DislncRF, to infer disease-associated lncRNAs on a genomic scale using tissue expression data and 
known disease-associated PCGs. DislncRF first trains random forest models on disease-associated PCG expression profiles across human tissues to generalize expression patterns 
for disease-associated genes. Second, it applies the trained models to predict the association scores between diseases and lncRNAs. The method was benchmarked against 
a gold standard set and compared with other commonly-used methods. The results show that DislncRF yields promising performance and outperforms the other methods. In addition, DislncRF can automatically identify 
disease-associated tissues.
