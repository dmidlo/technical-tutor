# nlp-technology_text_human_readabilty_index

## Abstract

Given a piece of text written for technologists, how difficult will it be for the reader to comprehend?

the goal is to give a "Cognitive Ease" score to a given text based it's "absolute" difficulty or complexity.

"Absolute" cognitive ease would be defined as the generic score given to a text that gives the text and places the ease score against against traditional readability and complexity score algorithms and 

## Method

Ensemble method using a diverse collection of 6617 technical publicationsBooks, articles, papers, posts as training set.

Initial training data is stored in pdf documents.


## Problems

### Metadata Extraction

pdf as a very flexible format, and as such, it's also can be a beast to parse.  At the document level, the first problem here is simply the Title (which, in the end probably won't be a highly weighted feature). PDF metadata is not required by the specification, so many pdfs do not have Title info.

### Taxonomies

#### Code Block or Not
`{"code", "not code"}`

#### Programming Language
`{"Python", "C++", "Bash", ...}`

#### Needs OCR
`{"Needs OCR", "Does Not Need OCR", "Already OCRed"}`

#### Page Types

```Python
{"Cover Page", "Title Page", "Publisher Page", "Blank Page", "Preface Page", "Appendix Page", "Chapter Start Page",
 "Chapter End Page", "Chapter Transition Page", "Biography Page", "Advertisement Page", "Forward Page", "ToC Page" ...}
```

#### Text Block

```Python
{"Document", "Topic", "Sub-Topic", "Chapter", "Section", "Sub-Section", "Article", "Figure Description", "Table", 
 "Paragraph", "Sub-Paragraph", "Sentence", "Phrase", "Word"}
```



## Common Code Complexity Metrics

From:
- https://linearb.io/blog/what-is-code-complexity/
- https://blog.codacy.com/an-in-depth-explanation-of-code-complexity/
- https://www.codegrip.tech/productivity/a-simple-understanding-of-code-complexity/


- Cyclomatic complexity - Thomas McCabe
- Switch Statement and Logic Condition Complexity
- Software Developer Skill Level
- Source Lines of Code (SLOC)
- Source Lines of Executable Code (SLOEC)
- Class Coupling
- Depth of Inheritance
- Maintainability Index
- Cognitive Complexity
- Halstead Volume
- Rework Ratio
- defect probability
- Number of Dependencies
- Number of Anonymous methods

## Bibliography

### References

- https://www.youtube.com/watch?v=N0o-Bjiwt0M

- https://prolingo.com/blog/what-readability-algorithm-scores-fail-to-tell-you/
- http://cs231n.stanford.edu/reports/2015/pdfs/kggriswo_FinalReport.pdf
- https://www.diva-portal.org/smash/get/diva2:721646/FULLTEXT01.pdf
- https://en.wikipedia.org/wiki/Readability
- https://readabilityformulas.com/search/pages/Readability_Formulas/
- https://en.wikipedia.org/wiki/HOCR
- https://www.deeplearning.ai/the-batch/how-metas-llama-nlp-model-leaked/
- https://machinelearningmastery.com/best-practices-document-classification-deep-learning/
- https://ieeexplore.ieee.org/document/8125990
- https://maelfabien.github.io/machinelearning/NLP_5/
- https://dylancastillo.co/text-classification-using-python-and-scikit-learn/
- https://www.atmosera.com/blog/text-classification-with-neural-networks/
- https://www.educative.io/answers/text-classification-in-nlp
- https://txt.cohere.ai/10-must-read-text-classification-papers/
- https://few-shot-text-classification.fastforwardlabs.com/
- https://stackabuse.com/text-classification-with-python-and-scikit-learn/
- https://www.mdpi.com/2078-2489/10/4/150
- http://www.scholarpedia.org/article/Text_categorization
- https://keras.io/examples/nlp/text_classification_from_scratch/
- https://www.kaggle.com/code/matleonard/text-classification
- https://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
- https://docs.uipath.com/ai-center/automation-cloud/latest/user-guide/text-classification
- https://nlp.stanford.edu/IR-book/html/htmledition/text-classification-and-naive-bayes-1.html
- https://autokeras.com/tutorial/text_classification/
- https://medium.com/text-classification-algorithms/text-classification-algorithms-a-survey-a215b7ab7e2d
- https://neptune.ai/blog/text-classification-tips-and-tricks-kaggle-competitions
- https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
- https://developer.apple.com/documentation/naturallanguage/creating_a_text_classifier_model
- https://www.sciencedirect.com/topics/computer-science/text-classification
- https://realpython.com/python-keras-text-classification/
- https://paperswithcode.com/task/text-classification
- https://www.datacamp.com/tutorial/text-classification-python
- https://towardsdatascience.com/machine-learning-nlp-text-classification-using-scikit-learn-python-and-nltk-c52b92a7c73a
- https://www.analyticsvidhya.com/blog/2018/04/a-comprehensive-guide-to-understand-and-implement-text-classification-in-python/
- http://cs231n.stanford.edu/reports/2015/pdfs/kggriswo_FinalReport.pdf
- https://aclanthology.org/N04-1042.pdf
- https://www.microsoft.com/en-us/research/publication/automatic-extraction-of-titles-from-general-documents-using-machine-learning/
- https://docear.org/papers/SciPlore%20Xtract%20--%20Extracting%20Titles%20from%20Scientific%20PDF%20Documents%20by%20Analyzing%20Style%20Information%20(Font%20Size)-preprint.pdf
- https://clgiles.ist.psu.edu/papers/JCDL-2003-automata-metdata.pdf
- https://ieeexplore.ieee.org/document/1204842
- https://www.researchgate.net/publication/262171677_Docear's_PDF_inspector_Title_extraction_from_PDF_files

### Github Topics

- pdf-files
- text-classification

### Google Searches

- grade level reading algroithms
- exploring pdf files in python
- scrape title of pdf
- text classification
- readability algorithms
- weebit corpora
- Newsela corpus
- text classification semi-supervised learning
- extract title from pdf
- extract title from pdf python
- python modify pdf metadata
- ocr scanned multi-page pdf python
- python tesseract sandwhich
- pdf renderer sandwich
- pathlib name without extension
- stanford alpaca 7b download
- normal distribution python
- text classification semi-supervised learning
- partial training machine learning
- extract title from pdf python
- code complexity metrics

### StackExchange Titles

- using pytesseract to generate a PDF from image
- Python with pytesseract - How to get the same output for pytesseract.image_to_data in a searchable PDF?
- How do I get the filename without the extension from a path in Python?
- How to extract the title of a PDF document from within a script for renaming?
- Extracting titles from PDF files?
- Extract titles from each page of a PDF?
- Extracting the actual in-text title from a PDF

### Tools

- (pypdfium2)[https://github.com/pypdfium2-team/pypdfium2]
- (pdfplumber)[https://github.com/jsvine/pdfplumber]
- (pdftotext)[https://github.com/jalan/pdftotext]
- (pikepdf)[https://github.com/pikepdf/pikepdf]
- (tabula-py)[https://github.com/chezou/tabula-py]
- (fpdf2)[https://github.com/PyFPDF/fpdf2]
- (pdfminer.six)[https://github.com/pdfminer/pdfminer.six]
- (pypdf)[https://github.com/py-pdf/pypdf]
- (pdftitle)[https://pypi.org/project/pdftitle/]

- (gTTS)[https://github.com/pndurette/gTTS]
- (robot)[https://github.com/robotframework/robotframework]
- (robotframework-pdf2textlibrary)[https://github.com/qahive/robotframework-pdf2textlibrary]
- (pathlib)[https://docs.python.org/3/library/pathlib.html]
- (python sh)[https://github.com/amoffat/sh]
- (python re)[https://docs.python.org/3/library/re.html]
- (pillow)[https://github.com/python-pillow/Pillow]
- (python io)[https://docs.python.org/3/library/io.html]
- (pytesseract)[https://github.com/madmaze/pytesseract]
- (wand)[https://github.com/emcconville/wand]
- (doctr)[https://github.com/mindee/doctr]
- (hocr-tools)[https://github.com/ocropus/hocr-tools]
- (pypdfocr)[https://pypi.org/project/pypdfocr/]
- (ocrmypdf)[https://ocrmypdf.readthedocs.io/en/latest/index.html]
- (unpaper)[https://github.com/Flameeyes/unpaper]
- (pdfsandwich)[http://www.tobias-elze.de/pdfsandwich/]
- (imagemagik -convert)[http://www.imagemagick.org/script/convert.php]

- (pdfviewer)[https://github.com/naiveHobo/pdfviewer]
- (xpdf)[https://github.com/ecatkins/xpdf_python]
- (pdfquery)[https://github.com/jcushman/pdfquery]
- (reportlab)[https://docs.reportlab.com/install/open_source_installation/]
- (pdfrw)[https://github.com/pmaupin/pdfrw]
- (slate)[https://github.com/timClicks/slate]
- (pdflib)[https://github.com/alephdata/pdflib]

- (paperwork)[https://gitlab.gnome.org/World/OpenPaperwork/paperwork]
- (ocrfeeder)[https://gitlab.gnome.org/GNOME/ocrfeeder]
- (teedy docs)[https://github.com/sismics/docs]
- (Papermerge)[https://github.com/ciur/papermerge]
- (Mayan EDMS)[https://gitlab.com/mayan-edms/mayan-edms]
- (paperless ngx)[https://github.com/paperless-ngx/paperless-ngx]
- (docspell)[https://github.com/eikek/docspell]

- https://nlp.gsu.edu/
- (nltk)[https://www.nltk.org/]
- (spaCy)[https://spacy.io/]
- (SciPy)[https://scipy.org/]

### Text Classification Techniques

Can you provide a brief description of these text-classification techniques?

Can you order these text-classification techniques according to levels of complexity, from least to most complex?

Can you group the following text-classification techniques according to levels of complexity? Such as basic, intermediate, or advanced?

This is a list of natural language processing and text classification techniques, can you separate them into two groups, where the groups are 'easier to learn as a human' and 'harder to learn as a human'?

These are two lists of natural language processing and text classification techniques. How would you rearrange these two lists to make them more accurate?

Can you rearrange this list of natural language processing and text classification techniques in order of increasing difficulty to learn as a human?






#### Level 1
Word Frequency-based methods
Rule-based Classification
Part-of-Speech (POS) Tagging

#### Level 2

Naive Bayes
SentiWordNet-based Classification
Sentiment Analysis

#### Level 3

Term Frequency-Inverse Document Frequency (TF-IDF)
Decision Stumps
K-Nearest Neighbors (KNN)

#### Level 4

Sentence Similarity Measures
Text Summarization
Entity Linking (EL)

#### Level 5

Named Entity Recognition (NER)
Word Sense Disambiguation (WSD)
Few-shot Learning

#### Level 6

Fuzzy Logic
Hierarchical Classification
Hyperparameter Optimization Techniques

# Level 7

Principal Component Analysis (PCA)
Linear Discriminant Analysis (LDA)
Semi-Supervised Learning with Clustering (SSLC)

#### Harder yet
Active Learning with Query-By-Committee (AL-QBC)
Viterbi Training for HMMs
Word Embeddings (Word2Vec, GloVe, etc.)
Transfer Learning
Ensemble of Decision Trees
Stochastic Gradient Descent (SGD)
Label Propagation (LP)
Style Transfer
Bayesian Networks
Curriculum Learning with Dynamic Difficulty Adjustment
Word Embeddings
Logistic Regression
Clustering
Independent Component Analysis (ICA)
Transfer Learning with Pre-Trained Word Embeddings (such as GloVe or FastText)
Maximum Entropy Models
Multi-Task Learning with Shared Encoders (MTL-SE)
Topic Modeling
Subword Encoding (Byte Pair Encoding, WordPiece)
Emotion Recognition

#### Even Harder
Document Embedding with Topic Models (DE-TM)
Relevance Vector Machines (RVMs)
Gradient Boosting
Monte Carlo Tree Search (MCTS)
Maximum Margin Clustering (MMC)
Feedforward Neural Network
Bayesian Optimization
Transfer Learning with Domain Adaptation (TL-DA)
Domain-Specific Text Generation
Extreme Learning Machines (ELMs)
Text Classification with Recurrent Neural Networks (RNNs)
Active Learning
Balanced Bagging
One-Class Classification
Text Augmentation with Back-Translation (TABT)
Text Classification with Deep Neural Networks (TC-DNN)
Universal Language Model Fine-tuning (ULMFiT)
Ensemble Techniques
Learning to Rank with Support Vector Machines (LTR-SVM)
Adaptive Boosting (AdaBoost)
Hidden Markov Models (HMMs)
Decision Fusion
Learning to Rank for Text Classification (LTR-TC)
Data Augmentation with Text Generation
Support Vector Machines (SVM)
Latent Dirichlet Allocation (LDA)
Exemplar-SVM (E-SVM)
Minimum Error Rate Training (MERT)
Latent Semantic Analysis (LSA)



#### Even Harder Still
Bag-Of-Words
One-vs-All Classification
Gradient Descent
Ensemble of Support Vector Machines (ESVM)
Multi-Layer Perceptron (MLP)
Probabilistic Graphical Models (PGMs)
Universal Sentence Encoder (USE)
Named Entity Recognition with Bidirectional LSTMs (BiLSTM-NER)
Gated Recurrent Unit (GRU)
Recurrent Neural Networks with Long Short-Term Memory (RNN-LSTM)
Pointer Networks
Transfer Learning with Pre-Trained Models (TL-PM)
Restricted Boltzmann Machines (RBMs)
Open-Domain Question Answering (ODQA)
Bayesian Neural Networks
Neural Dialogue Systems
Conditional Random Fields (CRFs)
Curriculum Learning with Reinforcement Learning (RL)
Text Classification with Cross-Lingual Pre-Trained Multilingual Language Models (CLTC-PMLM)
Cross-Modal Retrieval with Graph Convolutional Networks (CMR-GCN)
Neural Machine Translation (NMT)
Contextualized Word Embeddings (such as ELMo, BERT, and GPT-2)
Biaffine Parsing
Multi-View Learning
Dynamic Topic Models (DTM)
Unsupervised Learning for Natural Language Processing
Constituency Parsing
AdaBoost
Cross-Lingual Text Classification with Pre-Trained Multilingual Language Models (CLTC-PMLM)
Fine-Tuning Pre-Trained Language Models with Multi-Task Learning (FT-PLM)
Attention-based Recurrent Neural Networks (Att-RNN)
Dynamic Convolutional Neural Network (DCNN)
CatBoost
Federated Learning
Cross-Lingual Named Entity Recognition (CLNER)
Probabilistic Latent Semantic Analysis (pLSA)
Word2Vec with Negative Sampling (Word2Vec-NS)
Convolutional Neural Networks with Spatial Pyramid Pooling (CNN-SPP)
Extreme Gradient Boosting (XGBoost)
Mini-Batch Gradient Descent
Learning to Rank with Gradient Boosting Decision Trees (LTR-GBDT)
Multi-Task Learning with Graph Neural Networks (MTL-GNN)
Attention Mechanisms
Multi-Head Attention Mechanisms
Learning to Rank with Boosted Decision Trees (LTR-BDT)
BYOL (Bootstrap Your Own Latent)
Gradient-Based Learning (GBL)
Dependency Parsing
Domain-Specific Language Models
Hierarchical Attention Networks with Self-Attention (HAN-SA)
Stacked Denoising Autoencoders (SDA)
Long Short-Term Memory (LSTM) Networks
Hidden Semi-Markov Models (HSMMs)
Factorization Machines (FM)
Bayesian Networks (BN)
Structured Perceptron
Multi-Task Learning with Deep Contextualized Embeddings (MTL-DCE)
Graph Convolutional Networks for Text Classification (GCN-TC)
Meta-Learning with Memory-Augmented Neural Networks (MANN)
Reinforcement Learning (RL)
Recurrent Convolutional Neural Networks (RCNN)
Kernel Methods
Convolutional Neural Networks (CNN)
Multi-Label Learning
Siamese Networks
Non-negative Matrix Factorization (NMF)
Teacher-Student Learning
LightGBM
Deep Autoencoder Neural Network (DANN)

#### Hardest
Convolutional Recurrent Neural Networks (CRNNs)
Semantic Role Labeling (SRL)
Text Classification with Kernel Methods
Topic Modeling with Non-negative Matrix Factorization (NMF)
Knowledge Graphs (KG)
Pre-training and Fine-tuning Techniques
Ensemble of Convolutional Neural Networks (ECNN)
Deep Metric Learning (DML)- Convolutional Neural Networks with Global Average Pooling (CNN-GAP)
Generative Adversarial Networks (GANs) for Text Generation
Deep Attentive Sentence Ordering (DASO)
Self-Attention Networks with Relative Positional Encoding (SAN-RPE)
Collaborative Filtering with Word Embeddings (CF-WE)
Generative Pre-trained Transformer 3 (GPT-3)
Semi-Supervised Embedding
Text-to-Speech Synthesis
Attention-based Convolutional Neural Networks (Att-CNN)
Hierarchical Recurrent Encoder-Decoder Models (HRED)
Self-Attention Networks with Multi-Headed Attention and Relative Positional Encoding (SAN-MHA-RPE)
Capsule Networks with Attentional Routing (CapsAR)
Adversarial Training with Data Augmentation (ATDA)
Adversarial Training with Gradient Reversal (ATGR)
Bayesian Nonparametric Classification
Hierarchical Attention-based Prototype Networks (HAPN)
Attention-based Ensemble Models (Att-Ensemble)
Attention-based Meta-Learning (Att-Meta)
Few-Shot Text Classification with Pre-Trained Language Model (FSTC-PLM)
Graph Neural Networks with Edge Convolution (GCN-EC)
Convolutional Neural Networks with Stochastic Depth (CNN-SD)
Self-Attention Networks with Multi-Headed Attention (SAN-MHA)
Multi-Task Learning with Adversarial Training (MTL-AT-ADV)
Event Extraction
Multi-Task Learning with Parameter Sharing (MTL-PS)
Self-Training for Semi-Supervised Learning (ST-SSL)
Generative Adversarial Networks (GANs)
SimCLR (Simple Framework for Contrastive Learning of Representations)
Capsule Routing by Agreement (CRA)
Bayesian Belief Networks
Convolutional Neural Networks with Adversarial Training (CNN-AT)
Adversarial Training with Gradient Perturbation (ATGP)
Syntax-Based Machine Translation (SBMT)
Convolutional Neural Networks with Multi-Branch Architecture (CNN-MBA)
Contrastive Attention Mechanism for Text Classification (CAM-TC)
Transformer-based Sequence-to-Sequence Models (TSeq2Seq)
Multi-Task Learning with Attention Mechanisms (MTL-AM)
Semi-Supervised Learning with Consistency Regularization (SSL-CR)
Convolutional Neural Networks with DropBlock Regularization (CNN-DB)
Attentive Hierarchical Multi-Label Text Classification (AHMLTC)
Regularized Dual Averaging (RDA)
Learning to Rank with LambdaMART (LTR-LM)
Multi-Task Learning with Transformer-Based Language Model (MTL-TLM)
Transfer Learning with Multi-Task Learning (TL-MTL)
Bidirectional Encoder Representations from Transformers (BERT)
Few-shot Text Classification
Active Transfer Learning
Convolutional Neural Networks with Dynamic k-Max Pooling (CNN-DkMP)
Recurrent Neural Networks with Attention Mechanisms (RNN-AM)
Relation Extraction (RE)
Deep Graph Convolutional Networks (DGCN)
Adversarial Learning with Label Smoothing (ALLS)
Variational Bayesian Inference
Dynamic Co-Attention Networks (DCAN)
Attentive Topic Modeling (ATM)
Residual Networks (ResNets)
Multi-Head Multi-Layer Perceptron (MH-MLP)
Joint Learning of Named Entity Recognition and Entity Linking (NER-EL)
Cross-lingual Transfer Learning with Pre-Trained Language Models (CLTL-PLM)
Temporal Convolutional Networks with Residual Blocks (TCN-R)
Flow-based Generative Models
Deep Belief Networks (DBN)
Capsule Networks with Dynamic Convolutional Routing (CapsDC)
Contrastive Learning with SimSiam (CL-SimSiam)
Convolutional Neural Networks with Dilated Convolutions
Hierarchical Attention Networks (HANs)
Multi-Head Hierarchical Attention Networks (MH-HAN)
Semi-Supervised Learning with Deep Generative Models (SSL-DGM)
Sequence-to-Sequence Models (Seq2Seq)
Temporal Convolutional Attention Networks (TCAN)
Temporal Convolutional Networks with Dilations (TCN-D)
Adversarial Autoencoders (AAEs)
Variational Autoencoder with Attentional Flows (VAAFs)
Variational Autoencoders with Normalizing Flows (VAEs + NFs)
Convolutional-LSTM (ConvLSTM) networks
Self-Organizing Incremental Neural Tensor Networks (SOINT)
Reinforcement Learning with Natural Language Rewards
Adversarial Training with Natural Language Generation (AT-NLG)
Deep Reinforcement Learning
Reinforcement Learning with Policy Gradient Methods (RL-PG)
Graph Neural Networks (GNN)
Graph Convolutional Networks (GCNs)
Multi-Head Self-Attention Networks (MHSAN)
Transformer with Positional Encoding and Masking (TPM)
Task-Specific Pre-Training with Transformers (such as ALBERT or ELECTRA)
Knowledge Graph Embeddings
Differential Privacy
Active Semi-Supervised Learning (ASSL)
Deep Hybrid Convolutional Neural Networks (DHCNNs)
Capsule Networks with Dynamic Routing (CapsDR)
Variational Autoencoders with Adversarial Training (VAEs + AT)
Joint Sentiment Topic Modeling (JSTM)
Convolutional Neural Networks with Softmax-Margin Loss (CNN-SSL)
Multi-Task Learning with Knowledge Distillation (MTL-KD)
Knowledge Graph Embeddings for Text Classification (KGETC)
Self-Organizing Maps (SOMs)
Knowledge Graph-based Text Classification
Convolutional Neural Networks with Deep Supervision (CNN-DS)
Attention-based Deep Reinforcement Learning (Att-DRL)
Structured Neural Topic Models (SNTMs)
Multilingual Language Models (e.g., mBERT, XLM-R)
Knowledge Distillation
Compositional Perturbation Autoencoder (CPA)
Transformer-based Models
Deep Belief Networks (DBNs)
Dual Attention Networks (DANs)
Multi-View Learning with Correlated Views (MV-CV)
Actor-Critic Methods
Self-Supervised Learning with Instance Discrimination (SSLD)
Dynamic Memory Networks (DMN)
Long Short-Term Memory (LSTM)- Adversarial Autoencoder (AAE)
Multi-Objective Learning
Convolutional Neural Networks with Dilated Convolutions (CNN-DC)
Graph Convolutional Networks (GCN)
Mixture-of-Experts Networks (MENs)
Neural Networks with Relational Reasoning (NNRR)
Transformer-based Autoencoder
Semi-Supervised Learning with Generative Adversarial Networks (SSL-GAN)
Variational Autoencoders (VAEs)
Self-Attention Networks (SANs)
Attention-based Hybrid Models (Att-Hybrid)
Cross-Domain Text Classification with Domain-Adversarial Training (CD-DA)
Self-supervised Learning
Domain Adaptation
Language Generation (such as text generation, machine translation, and summarization)
Transformer Models (BERT, GPT-2, etc.)
Named Entity Disambiguation (NED)
Self-Supervised Learning with Masked Language Modeling (SSL-MLM)
Domain-Specific Word Embeddings
Recurrent Convolutional Neural Tensor Network (RCNTN)
Temporal Convolutional Networks (TCN)
Convolutional Neural Networks with Adaptive Pooling (CNN-AP)
Generative Adversarial Networks with Reinforcement Learning (GAN-RL)
Multi-Task Learning with Deep Convolutional Neural Networks (MTL-DCNN)
Structured Prediction
Recursive Neural Networks (RvNN)
Conditional Variational Autoencoders (CVAEs)
Variational Inference for Topic Modeling (VITM)
Differentiable Architecture Search (DARTS)
Dynamic Routing Capsule Networks (DRCN)
Multilingual Translation with Pre-trained Language Models (MT-PLM)
Convolutional Neural Networks with Dynamic Routing (CNN-DR)
Adversarial Training with Gaussian Noise (ATGN)
Capsule Networks
Convolutional Neural Networks with Attention over Input Sequences (CNN-AIS)
Dual Encoder Models
Generative Adversarial Networks for Text (GANs-T)
Relevance Vector Machine (RVM)
Semi-Supervised Learning with Co-Training (SSL-CT)
Deep Convolutional Inverse Graphics Network (DC-IGN)
Dynamic Embeddings for Language Evolution (DELE)
Multi-Task Learning with Adversarial Training (MTL-AT)
Semi-Supervised Learning with Label Propagation (SSL-LP)
Capsule Networks with Self-Attention Routing (CapsSA)
Convolutional Neural Networks with Residual Blocks and Multi-Scale Feature Fusion (CNN-RESMS)
Self-Organizing Incremental Neural Networks (SOINN)
Transformer-based Language Models (TLM)
Text Classification with Convolutional Neural Networks (TC-CNN)
Semi-Supervised Learning with Label Propagation (SSL-LP)
Generative Pre-trained Transformer 2 (GPT-2)
Semi-Supervised Learning with Cluster Assumption (SSL-CA)
Multiple Instance Learning (MIL)
Reinforcement Learning with Monte Carlo Tree Search (RL-MCTS)
Hierarchical Recurrent Neural Networks (HRNN)
Zero-shot Learning for Text Classification
Contrastive Learning with SimCLR (CL-SimCLR)
Self-Organizing Maps with Gaussian Mixture Model (SOM-GMM)
Deep Q-Learning
Hierarchical Reinforcement Learning for Dialogue Generation
Text Generation with Variational Autoencoders (VAE-TG)
Cross-Lingual Learning with Multilingual Language Models
Recursive Neural Networks (RecNN)
Global-Local Hierarchical Attention Networks (GLHAN)
Self-Attention Networks with Task-Specific Projection (SAN-TP)
Neural Topic Model (NTM)
Deep Reinforcement Fuzzy C-Means (DRFCM)
Contrastive Self-supervised Learning
Contrastive Learning
Graph Attention Networks (GATs)
Zero-shot Learning with Pre-trained Language Models (ZSL-PLM)
Multi-Head Self-Attention with Layer Normalization (MHSA-LN)
Adversarial Training with Weighted Sampling (ATWS)
Attention-based Hierarchical Multi-Label Text Classification (AHMLTC)
Convolutional Neural Networks with Dynamic Pooling (CNN-DP)
Text Style Transfer
Contrastive Learning with SwAV
OpenAI's GPT-3 language model
Temporal Convolutional Networks (TCNs)
Attention-Based Graph Neural Networks (AB-GNN)
Semi-supervised Sequence Learning with LSTMs
Bayesian Deep Learning
Multi-Task Learning with Task-Specific Attention (MTL-TSA)
Deep Residual Learning (DRL)
Learning to Rank with Neural Networks (LTR-NN)
Semi-Supervised Learning with Graph-Based Methods (SSL-GB)
Graph Attention Networks (GAT)
Recursive Neural Tensor Networks (RNTNs)
Convolutional Neural Networks with Spatial Attention (CNN-SA)
Convolutional Neural Networks with Mixup (CNN-Mixup)
Multi-Label Text Classification with Label Dependencies (MLCD)
Neural Style Transfer (NST)
Semi-Supervised Learning with Self-Training (SSL-ST)
Deep Bayesian Neural Networks
Graph Attention Networks (GAT)
Deep Bayesian Neural Networks
Recursive Neural Tensor Networks (RNTNs)
Convolutional Neural Networks with Mixup (CNN-Mixup)
Convolutional Neural Networks with Spatial Attention (CNN-SA)
Multi-Label Text Classification with Label Dependencies (MLCD)
Neural Style Transfer (NST)
Semi-Supervised Learning with Self-Training (SSL-ST)
Hierarchical Attention Networks (HAN)
Attention-based Multi-Label Text Classification (AMLC)
Attention-based Pointer Networks for Text Generation
Self-Attention Networks with Gated Positional Encoding (SAN-GPE)
ELMo (Embeddings from Language Models)
Semi-Supervised Learning with Generative Models (SSGM)
Cross-lingual Text Classification with Adversarial Discriminative Domain Adaptation (CLTC-ADDA)
Convolutional Neural Networks with Curriculum Learning (CNN-CL)
Deep Reinforcement Learning with Monte Carlo Tree Search (DRL-MCTS)