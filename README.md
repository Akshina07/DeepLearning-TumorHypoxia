# DeepLearning-TumorHypoxia
Transfer Learning on whole slide tumor images to classify tumors as hypoxic or non-hypoxic
# BACKGROUND:
Many primary-tumor subregions have low levels of molecular oxygen, termed hypoxia. Hypoxic tumors are at elevated risk for local failure and distant metastasis, but the molecular hallmarks of tumor hypoxia remain poorly defined. It is observed hypoxia is associated with elevated genomic instability, hypoxic tumors exhibited characteristic driver-mutation signatures and a widespread hypoxia-associated dysregulation of microRNAs (miRNAs) across cancers. Hypoxia may also be associated with elevated rates of chromothripsis, allelic loss of PTEN and shorter telomeres. Thus, tumor hypoxia may drive aggressive molecular features across cancers and shape the clinical trajectory of individual tumors. Therefore finding the correlation between cancer tumors and level of molecular oxygen can find potential targeted genes associated with hypoxia which can also be used to treat tumor types. We applied deep learning techniques like  transfer learning and deep k-means clustering to classify tumors as hypoxic or non-hypoxic on TCGA whole slide tumor type for breast cancer (BRCA)

# DATA COLLECTION: 
Downloaded TCGA whole slide images from GDC data portal. The patient cases considered were extracted from a previously conducted study on Molecular landmarks of tumor hypoxia across cancer type(ref.1) Supplementary Table1. The pan cancer hypoxia scores were used to generate labels (described in Labelling) which were considered as the ground truth. A total of 13031 wsi images were downloaded (all tumor types and including associated images)

# RESOURCES:
1.	LINK TO DATA: https://static-content.springer.com/esm/art%3A10.1038%2Fs41588-018-0318-2/MediaObjects/41588_2018_318_MOESM3_ESM.txt 

# DATA PRE-PROCESSING:
Whole slide images are very large and can only be viewed using specially designed libraries or software. The images were processed using a C library extension in python open slide-python. The slides were tiled into 299 X 299 pixels (input size for input to ImageNet InceptionV3) in a non-overlapping fashion such that tiles with more than 25% background were rejected. On average 2000 tiles were generated per image. The tiles were split into 3 sets: train (70%), validation set (15%) and test set (15%), making sure that tiles of a particular patient are completely contained in one of the three partitions.

# LABELLLING: 
Labelled the patient tumor slides using pan cancer hypoxia scores (of all the signatures Buffa_hypoxia_pan_cancer_score from resource 1. was used). For each tumor type the displacement from the median of Z-scores was calculated and scaled to the range [-1,1]. All the tumor with median scores greater than 0 were labelled as hypoxic and others as non-hypoxic. 

# TRAINING AND EVALUTION:
Used two pre-trained models- Resnet50 and InceptionV3 for fine-tuning and feature extraction respectively. The transfer values of bottle neck values generated were used as input to a top-up model used to classify into two classes= hypoxic or non-hypoxic. The model evaluation metrics used were f1-scores, auc scores and accuracy. 
 
# REFERNCES:
1.	https://github.com/ncoudray/DeepPATH
2.	Nature article, Molecular landmarks of tumor hypoxia across cancer types published on 14th January 2019 (https://www.nature.com/articles/s41588-018-0318-2)
3.	Nature article, Classification and mutation prediction from non–small cell lung cancer histopathology images using deep learning published on 17th September 2018 (https://www.nature.com/articles/s41591-018-0177-5)
4.	Nature article, Dermatologist-level classification of skin cancer with deep neural networks published on 25th January 2017 (https://www.nature.com/articles/nature21056)
5.	https://medium.com/@sh.tsang/review-inception-v3-1st-runner-up-image-classification-in-ilsvrc-2015-17915421f77c
6.	https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
7.	Predicting cancer outcomes from histology and genomics using convolutional networks published in PNAS on 27th March 2018 (https://www.pnas.org/content/115/13/E2970)
8.	Structured Crowdsourcing Enables Convolutional Segmentation of Histology Images (https://www.ncbi.nlm.nih.gov/pubmed/30726865)
9.	Precision histology: how deep learning is poised to revitalize histomorphology for personalized cancer care (https://www.nature.com/articles/s41698-017-0022-1)
10.	 The Digital Slide Archive: A Software Platform for Management, Integration, and Analysis of Histology for Cancer Research (https://www.ncbi.nlm.nih.gov/pubmed/29092945)
11.	Nature Review Deep learning: new computational modelling techniques for genomics (https://www.nature.com/articles/s41576-019-0122-6)
12.	Deep learning detects virus presence in cancer histology (https://www.biorxiv.org/content/10.1101/690206v1)
13.	https://www.learnopencv.com/keras-tutorial-transfer-learning-using-pre-trained-models/



