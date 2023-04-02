"""
A script... generator. in the truest sense.  To help to have a conversation
with chatGPT to help generate a list ordered by difficulty of nlp and text
classification techniques.

Here's the prompts:

Can you provide a brief description of these text-classification techniques?

Can you order these text-classification techniques according to levels of complexity, from least to most complex?

Can you group the following text-classification techniques according to levels of complexity? Such as basic, intermediate, or advanced?

This is a list of natural language processing and text classification techniques, can you separate them into two groups, where the groups are 'easier to learn as a human' and 'harder to learn as a human'?

These are two lists of natural language processing and text classification techniques. How would you rearrange these two lists to make them more accurate?

Can you rearrange this list of natural language processing and text classification techniques in order of increasing difficulty to learn as a human?
"""


from rich import print

import random

ml_types = [
    'Semi-Supervised Learning with Generative Models (SSGM)',
    'Reinforcement Learning with Policy Gradient Methods (RL-PG)',
    'Recurrent Neural Networks with Long Short-Term Memory (RNN-LSTM)',
    'Convolutional Neural Networks with Deep Supervision (CNN-DS)',
    'Capsule Networks with Attentional Routing (CapsAR)',
    'Multi-Task Learning with Deep Contextualized Embeddings (MTL-DCE)',
    'Semi-supervised Embedding',
    'Bag-Of-Words',
    'Hidden Markov Models (HMMs)',
    'Self-Training for Semi-Supervised Learning (ST-SSL)',
    'TextRank',
    'Probabilistic Latent Semantic Analysis (pLSA)',
    'Principal Component Analysis (PCA)',
    'Semantic Role Labeling (SRL)',
    'Adversarial Training with Gradient Reversal (ATGR)',
    'Recurrent Convolutional Neural Tensor Network (RCNTN)',
    'Ensemble Learning',
    'Active Transfer Learning',
    'CatBoost',
    'Transfer Learning',
    'Word Embeddings (Word2Vec, GloVe, etc.)',
    'Graph Attention Networks (GAT)',
    'Multi-Task Learning with Attention Mechanisms (MTL-AM)',
    'Extreme Gradient Boosting (XGBoost)',
    'Neural Bag-of-Ngrams (NBoW)',
    'Multi-Objective Learning',
    'Hierarchical Recurrent Encoder-Decoder Models (HRED)',
    'Text-to-Speech Synthesis',
    'Gradient-Based Learning (GBL)',
    'Cross-Modal Retrieval with Graph Convolutional Networks (CMR-GCN)',
    'Joint Sentiment Topic Modeling (JSTM)',
    'Multi-View Learning with Correlated Views (MV-CV)',
    'Self-Attention Networks with Multi-Headed Attention and Relative Positional Encoding (SAN-MHA-RPE)',
    'Self-supervised Learning',
    'Ensemble of Decision Trees',
    'Multi-Task Learning with Graph Neural Networks (MTL-GNN)',
    'Self-Organizing Maps (SOMs)',
    'Neural Machine Translation (NMT)',
    'Subword Encoding (Byte Pair Encoding, WordPiece)',
    'Teacher-Student Learning',
    'Convolutional Neural Networks with Dynamic k-Max Pooling (CNN-DkMP)',
    'Convolutional Recurrent Neural Networks (CRNNs)',
    'Hierarchical Attention Networks with Self-Attention (HAN-SA)',
    'Entity Linking (EL)',
    'Attention-based Hierarchical Multi-Label Text Classification (AHMLTC)',
    'Deep Graph Convolutional Networks (DGCN)',
    'Monte Carlo Tree Search (MCTS)',
    'Topic Modeling',
    'Convolutional Neural Networks with Dynamic Routing (CNN-DR)',
    'Style Transfer',
    'Text Summarization',
    'Ensemble Techniques',
    'Hidden Semi-Markov Models (HSMMs)',
    'Emotion Recognition',
    'Transfer Learning with Multi-Task Learning (TL-MTL)',
    'Gated Recurrent Unit (GRU)',
    'Self-Attention Networks with Multi-Headed Attention (SAN-MHA)',
    'Long Short-Term Memory (LSTM)- Adversarial Autoencoder (AAE)',
    'Convolutional Neural Networks with Adaptive Pooling (CNN-AP)',
    'Named Entity Recognition with Bidirectional LSTMs (BiLSTM-NER)',
    'Contrastive Self-supervised Learning',
    'Sequence Labeling with Conditional Random Fields (CRF)',
    'Temporal Convolutional Networks with Dilations (TCN-D)',
    'Recursive Neural Tensor Networks (RNTNs)',
    'Graph Attention Networks (GATs)',
    'Multi-Task Learning with Adversarial Training (MTL-AT)',
    'Variational Autoencoders with Adversarial Training (VAEs + AT)',
    'Sentence Similarity Measures',
    'Pointer Networks',
    'Clustering',
    'Named Entity Disambiguation (NED)',
    'Semi-supervised Sequence Learning with LSTMs',
    'Minimum Error Rate Training (MERT)',
    'Attention Mechanisms',
    'Federated Learning',
    'BYOL (Bootstrap Your Own Latent)',
    'Cross-Lingual Learning with Multilingual Language Models',
    'Conditional Variational Autoencoders (CVAEs)',
    'Multi-Task Learning with Adversarial Training (MTL-AT-ADV)',
    'Text Classification with Kernel Methods',
    'Multi-Task Learning',
    'Deep Metric Learning (DML)- Convolutional Neural Networks with Global Average Pooling (CNN-GAP)',
    'Adversarial Training with Gaussian Noise (ATGN)',
    'Dual Attention Networks (DANs)',
    'Viterbi Training for HMMs',
    'Language Modeling with Convolutional Neural Networks (LM-CNN)',
    'Contrastive Learning with SimSiam (CL-SimSiam)',
    'Adversarial Training with Weighted Sampling (ATWS)',
    'Multilingual Translation with Pre-trained Language Models (MT-PLM)',
    'Bi-Directional LSTM with Attention Mechanism (BiLSTM-ATT)',
    'Neural Style Transfer (NST)',
    'Multi-Head Multi-Layer Perceptron (MH-MLP)',
    'Part-of-Speech (POS) Tagging',
    'Multi-Task Learning with Knowledge Distillation (MTL-KD)',
    'Recursive Neural Networks (RecNN)',
    'Transformer-based Language Models (TLM)',
    'Variational Bayesian Inference',
    'Attention-based Meta-Learning (Att-Meta)',
    'Bayesian Networks (BN)',
    'Contrastive Learning',
    'Dynamic Routing Capsule Networks (DRCN)',
    'Bayesian Optimization',
    'Semi-Supervised Learning with Clustering (SSLC)',
    'Graph Neural Networks (GNN)',
    'Independent Component Analysis (ICA)',
    'Reinforcement Learning',
    'Latent Semantic Analysis (LSA)',
    'Dynamic Convolutional Neural Network (DCNN)',
    'Attention-Based Graph Neural Networks (AB-GNN)',
    'Generative Pre-trained Transformer 3 (GPT-3)',
    'Neural Topic Model (NTM)',
    'Hierarchical Classification',
    'Graph Convolutional Networks (GCNs)',
    'Few-shot Learning',
    'Cross-Lingual Named Entity Recognition (CLNER)',
    'Word Sense Disambiguation (WSD)',
    'Capsule Networks',
    'Generative Adversarial Networks for Text (GANs-T)',
    'Fuzzy Logic',
    'Temporal Convolutional Attention Networks (TCAN)',
    'Convolutional Neural Networks with Stochastic Depth (CNN-SD)',
    'Hierarchical Attention-based Prototype Networks (HAPN)',
    'Convolutional Neural Networks with Adversarial Training (CNN-AT)',
    'Neural Dialogue Systems',
    'Universal Sentence Encoder (USE)',
    'Autoencoder-based Classification (AEC)',
    'Semi-Supervised Learning with Cluster Assumption (SSL-CA)',
    'Universal Language Model Fine-tuning (ULMFiT)',
    'Sentiment Analysis',
    'Convolutional Neural Networks with Spatial Attention (CNN-SA)',
    'Ensemble of Support Vector Machines (ESVM)',
    'Linear Discriminant Analysis (LDA)',
    'Task-Specific Pre-Training with Transformers (such as ALBERT or ELECTRA)',
    'Dynamic Co-Attention Networks (DCAN)',
    'Dual Encoder Models',
    'Self-Attention Networks with Relative Positional Encoding (SAN-RPE)',
    'Multi-Layer Perceptron (MLP)',
    'Generative Pre-trained Transformer 2 (GPT-2)',
    'Transfer Learning with Pre-Trained Word Embeddings (such as GloVe or FastText)',
    'Transformer with Positional Encoding and Masking (TPM)',
    'Deep Residual Learning (DRL)',
    'Contrastive Attention Mechanism for Text Classification (CAM-TC)',
    'Learning to Rank with Neural Networks (LTR-NN)',
    'Hierarchical Reinforcement Learning for Dialogue Generation',
    'LightGBM',
    'Differentiable Architecture Search (DARTS)',
    'Deep Attentive Sentence Ordering (DASO)',
    'Zero-shot Learning for Text Classification',
    'Semi-Supervised Learning with Self-Training (SSL-ST)',
    'Semi-supervised Learning with Co-Training (SSL-CT)',
    'Attention-based Deep Reinforcement Learning (Att-DRL)',
    'Adversarial Learning with Label Smoothing (ALLS)',
    'Curriculum Learning with Dynamic Difficulty Adjustment',
    'Term Frequency-Inverse Document Frequency (TF-IDF)',
    'Attentive Topic Modeling (ATM)',
    'Temporal Convolutional Networks (TCNs)',
    'Reinforcement Learning with Natural Language Rewards',
    'Exemplar-SVM (E-SVM)',
    'Fine-Tuning Pre-Trained Language Models with Multi-Task Learning (FT-PLM)',
    'Convolutional-LSTM (ConvLSTM) networks',
    'Variational Autoencoders (VAEs)',
    'Adversarial Training with Natural Language Generation (AT-NLG)',
    'Hierarchical Recurrent Neural Networks (HRNN)',
    'Stacked Denoising Autoencoders (SDA)',
    'Bayesian Nonparametric Classification',
    'Named Entity Recognition (NER)',
    'Relation Extraction (RE)',
    'Long Short-Term Memory (LSTM) networks',
    'Deep Reinforcement Fuzzy C-Means (DRFCM)',
    'Generative Adversarial Networks (GANs)',
    'Structured Perceptron',
    'Deep Belief Networks (DBNs)',
    'Graph Neural Networks with Edge Convolution (GCN-EC)',
    'Structured Prediction',
    'Curriculum Learning',
    'Knowledge Graph Embeddings for Text Classification (KGETC)',
    'Actor-Critic Methods',
    'Transformer-based Models',
    'Knowledge Graph Embeddings',
    'Multi-Task Learning with Shared Encoders (MTL-SE)',
    'Domain Adaptation',
    'Attention-based Pointer Networks for Text Generation',
    'Knowledge Graph-based Text Classification',
    'Transformer-based Sequence-to-Sequence Models (TSeq2Seq)',
    'Active Learning with Query-By-Committee (AL-QBC)',
    'Dynamic Memory Networks (DMN)',
    'Long Short-Term Memory Networks (LSTMs)',
    'Unsupervised Learning for Natural Language Processing',
    'Topic Modeling with Non-negative Matrix Factorization (NMF)',
    'Multi-Label Text Classification with Label Dependencies (MLCD)',
    'Graph Convolutional Networks (GCN)',
    'Text Classification with Deep Neural Networks (TC-DNN)',
    'Cross-lingual Transfer Learning with Pre-Trained Language Models (CLTL-PLM)',
    'Language Generation (such as text generation, machine translation, and summarization)',
    'Bayesian Belief Networks',
    'Multi-Task Learning with Parameter Sharing (MTL-PS)',
    'Gradient Descent',
    'Adversarial Training with Data Augmentation (ATDA)',
    'Cross-lingual Word Embeddings',
    'Relevance Vector Machine (RVM)',
    'Learning to Rank for Text Classification (LTR-TC)',
    'Multi-View Learning',
    'Self-Attention Networks (SANs)',
    'Deep Belief Networks (DBN)',
    'Contrastive Learning with SwAV',
    'Reinforcement Learning with Monte Carlo Tree Search (RL-MCTS)',
    'Capsule Networks with Dynamic Convolutional Routing (CapsDC)',
    'Dynamic Topic Models (DTM)',
    'Attention-based Ensemble Models (Att-Ensemble)',
    'SentiWordNet-based Classification',
    'Domain-Specific Language Models',
    'Attention-based Recurrent Neural Networks (Att-RNN)',
    'Multi-Scale Convolutional Neural Networks (MSCNNs)',
    'Cross-Lingual Text Classification with Pre-Trained Multilingual Language Models (CLTC-PMLM)',
    'Deep Hybrid Convolutional Neural Networks (DHCNNs)',
    'Biaffine Parsing',
    'Transformer Models (BERT, GPT-2, etc.)',
    'One-Class Classification',
    'Recurrent Convolutional Neural Networks (RCNN)',
    'Word2Vec with Negative Sampling (Word2Vec-NS)',
    'Multi-Task Learning with Task-Specific Attention (MTL-TSA)',
    'Flow-based Generative Models',
    'Capsule Routing by Agreement (CRA)',
    'Contrastive Learning with SimCLR (CL-SimCLR)',
    'Learning to Rank with Support Vector Machines (LTR-SVM)',
    'Data Augmentation with Text Generation',
    'Convolutional Neural Networks with Curriculum Learning (CNN-CL)',
    'Knowledge Distillation',
    'Convolutional Neural Networks with Recurrent Neural Networks (CNN-RNN)',
    'Self-Attention Networks with Gated Positional Encoding (SAN-GPE)',
    'Kernel Methods',
    'Temporal Convolutional Networks with Residual Blocks (TCN-R)',
    'Attention-based Multi-Label Text Classification (AMLC)',
    'Few-shot Text Classification',
    'SimCLR (Simple Framework for Contrastive Learning of Representations)',
    'Semi-Supervised Learning with Co-Training (SSL-CT)',
    'Multi-Task Learning with Recurrent Neural Networks (RNNs)',
    'Capsule Networks with Self-Attention Routing (CapsSA)',
    'Attention-based Convolutional Neural Networks (Att-CNN)',
    'Transformer-based Autoencoder',
    'Self-Supervised Learning with Instance Discrimination (SSLD)',
    'Multi-Label Learning',
    'Residual Networks (ResNets)',
    'Neural Networks with Relational Reasoning (NNRR)',
    'Meta-Learning with Memory-Augmented Neural Networks (MANN)',
    'Syntax-Based Machine Translation (SBMT)',
    'Joint Learning of Named Entity Recognition and Entity Linking (NER-EL)',
    'Semi-Supervised Learning with Generative Adversarial Networks (SSL-GAN)',
    'Convolutional Neural Networks with Dilated Convolutions (CNN-DC)',
    'Graph Convolutional Networks for Text Classification (GCN-TC)',
    'Convolutional Neural Networks with Mixup (CNN-Mixup)',
    'Multi-Head Hierarchical Attention Networks (MH-HAN)',
    'Convolutional Neural Networks with Residual Blocks and Multi-Scale Feature Fusion (CNN-RESMS)',
    'Temporal Convolutional Networks (TCN)',
    'Cross-lingual Text Classification with Adversarial Discriminative Domain Adaptation (CLTC-ADDA)',
    'Open-Domain Question Answering (ODQA)',
    'One-vs-All Classification',
    'Multi-Head Self-Attention Networks (MHSAN)',
    'Self-Attention Networks with Task-Specific Projection (SAN-TP)',
    'Bayesian Neural Networks',
    'Recursive Neural Networks (RvNN)',
    'Active Learning',
    'Extreme Learning Machines (ELMs)',
    'Convolutional Neural Networks with Attention Mechanism (CNN-ATT)',
    'Multi-Task Learning with Deep Convolutional Neural Networks (MTL-DCNN)',
    'FastText',
    'Zero-shot Learning with Pre-trained Language Models (ZSL-PLM)',
    'feedforward neural network',
    'Convolutional Neural Networks with Dilated Convolutions',
    'Mini-Batch Gradient Descent',
    'Maximum Entropy Models',
    'Generative Adversarial Networks with Reinforcement Learning (GAN-RL)',
    'Bayesian Deep Learning',
    'Dynamic Embeddings for Language Evolution (DELE)',
    'Self-Organizing Incremental Neural Tensor Networks (SOINT)',
    'Text Classification with Recurrent Neural Networks (RNNs)',
    'Domain-Specific Text Generation',
    'Knowledge Graphs (KG)',
    'Domain-Specific Word Embeddings',
    'Attentive Hierarchical Multi-Label Text Classification (AHMLTC)',
    'Hierarchical Attention Networks (HAN)',
    'Label Propagation (LP)',
    'Gradient Boosting',
    'Text Style Transfer',
    'Self-Attention with Contextual Embeddings (SACE)',
    'Non-negative Matrix Factorization (NMF)',
    'Deep Reinforcement Learning',
    'Convolutional Neural Networks with Dynamic Pooling (CNN-DP)',
    'Text Augmentation with Back-Translation (TABT)',
    'Relevance Vector Machines (RVMs)',
    'Dependency Parsing',
    'Probabilistic Graphical Models (PGMs)',
    'Convolutional Neural Networks with Spatial Pyramid Pooling (CNN-SPP)',
    'Semi-Supervised Learning with Label Propagation (SSL-LP)',
    'Ensemble of Convolutional Neural Networks (ECNN)',
    'Random Forest',
    'Generative Adversarial Networks for Text Generation (GANs-G)',
    'Multi-Task Learning with Shared Representations',
    'Latent Dirichlet Allocation (LDA)',
    'Adversarial Autoencoders (AAEs)',
    'Recurrent Neural Networks with Attention Mechanisms (RNN-AM)',
    'Sequence-to-Sequence Models (Seq2Seq)',
    'Adversarial Training with Gradient Perturbation (ATGP)',
    'Text Classification with Convolutional Neural Networks (TC-CNN)',
    'Multi-Head Attention Mechanisms',
    "OpenAI's GPT-3 language model",
    'Balanced Bagging',
    'Variational Autoencoders with Normalizing Flows (VAEs + NFs)',
    'Semi-Supervised Learning with Deep Generative Models (SSL-DGM)',
    'Multi-Head Attention Networks (MHAN)',
    'Learning to Rank with Gradient Boosting Decision Trees (LTR-GBDT)',
    'Adversarial Training with Label Noise (ATLN)',
    'Regularization Techniques',
    'Restricted Boltzmann Machines (RBMs)',
    'Stochastic Gradient Descent (SGD)',
    'Regularized Dual Averaging (RDA)',
    'Hyperparameter Optimization Techniques',
    'Convolutional Neural Networks with DropBlock Regularization (CNN-DB)',
    'Deep Q-Learning',
    'Multi-Task Learning with Transformer-Based Language Model (MTL-TLM)',
    'Learning to Rank with Boosted Decision Trees (LTR-BDT)',
    'Active Semi-Supervised Learning (ASSL)',
    'Self-Supervised Learning with Masked Language Modeling (SSL-MLM)',
    'Capsule Networks with Dynamic Routing (CapsDR)',
    'Event Extraction',
    'Deep Autoencoder Neural Network (DANN)',
    'Deep Convolutional Inverse Graphics Network (DC-IGN)',
    'Multi-Task Learning with Auxiliary Tasks (MTL-AT)',
    'Recurrent Neural Networks (RNN)',
    'Pre-training and Fine-tuning Techniques',
    'Multilingual Language Models (e.g., mBERT, XLM-R)',
    'ELMo (Embeddings from Language Models)',
    'Transfer Learning with Domain Adaptation (TL-DA)',
    'Self-Organizing Incremental Neural Networks (SOINN)',
    'Deep Reinforcement Learning with Monte Carlo Tree Search (DRL-MCTS)',
    'Structured Neural Topic Models (SNTMs)',
    'Self-Organizing Maps with Gaussian Mixture Model (SOM-GMM)',
    'Text Generation with Variational Autoencoders (VAE-TG)',
    'Rule-based Classification',
    'Word Embeddings',
    'Decision Fusion',
    'Convolutional Neural Networks with Multi-Branch Architecture (CNN-MBA)',
    'Deep Bayesian Neural Networks',
    'Factorization Machines (FM)',
    'Transfer Learning with Pre-trained Models (TL-PM)',
    'Text Generation with Transformers',
    'Attention-based Hybrid Models (Att-Hybrid)',
    'Cross-Domain Text Classification with Domain-Adversarial Training (CD-DA)',
    'Convolutional Neural Networks with Attention over Input Sequences (CNN-AIS)',
    'Hierarchical Attention Networks (HANs)',
    'Multi-Head Self-Attention with Layer Normalization (MHSA-LN)',
    'Maximum Margin Clustering (MMC)',
    'Coreference Resolution',
    'Bidirectional Encoder Representations from Transformers (BERT)',
    'Differential Privacy',
    'Convolutional Neural Networks with Softmax-Margin Loss (CNN-SSL)',
    'Compositional Perturbation Autoencoder (CPA)',
    'Collaborative Filtering with Word Embeddings (CF-WE)',
    'Contextualized Word Embeddings (such as ELMo, BERT, and GPT-2)',
    'Constituency Parsing',
    'Adaptive Boosting (AdaBoost)',
    'Document Embedding with Topic Models (DE-TM)',
    'Global-Local Hierarchical Attention Networks (GLHAN)- Word Frequency-based methods',
    'Conditional Random Fields (CRFs)',
    'Semi-Supervised Learning with Consistency Regularization (SSL-CR)',
    'Variational Inference for Topic Modeling (VITM)',
    'Variational Autoencoder with Attentional Flows (VAAFs)',
    'Support Vector Machines (SVM)',
    'Multiple Instance Learning (MIL)',
    'Bayesian Networks',
    'Convolutional Neural Networks (CNN)',
    'Siamese Networks',
    'Semi-Supervised Learning with Graph-Based Methods (SSL-GB)',
    'Decision Stumps',
    'K-Nearest Neighbors (KNN)',
    'Few-Shot Text Classification with Pre-Trained Language Model (FSTC-PLM)',
    'Mixture-of-Experts Networks (MENs)',
    'Learning to Rank with LambdaMART (LTR-LM)',
    'AdaBoost',
    'Naive Bayes',
    'Gated Recurrent Units (GRUs)',
    'Logistic Regression'
]

ml_easier_types = [
    "Sentence Similarity Measures",
    "Text Summarization",
    "Few-shot Learning",
    "Named Entity Recognition (NER)",
    "Entity Linking (EL)",
    "Word Sense Disambiguation (WSD)",
]

easier = {
    "Word Frequency-based methods",
    "Rule-based Classification",
    "Text Summarization",
    "Part-of-Speech (POS) Tagging",
    "Named Entity Recognition (NER)"
}
    
harder = {
    "SentiWordNet-based Classification",
    "Naive Bayes",
    "Entity Linking (EL)",
    "Sentiment Analysis"
}



def check():
    if input == output:
        print(True)

def split_list(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def rearrange_easy_hard():
    easer_dedup = easier - harder
    harder_dedup = harder - easier

    easier_list = list(easer_dedup)
    harder_list = list(harder_dedup)

    for _ in range(0, 10_000):
        random.shuffle(easier_list)
        random.shuffle(harder_list)

    number_of_easier_sublists = 3
    number_of_harder_sublists = 2

    easier_sublists = list(split_list(easier_list, number_of_easier_sublists))
    harder_sublists = list(split_list(harder_list, number_of_harder_sublists))

    for i, j in zip(easier_sublists, harder_sublists):
        print("These are two lists of natural language processing and text classification techniques. How would you rearrange these two lists to make them more accurate? (Feel free to move items between lists if it makes sense.)\n")
        print("easier to learn as a human:", "\n") 
        
        for _ in i:
            print(f"- {_}")
            
        print("\n", "harder to learn as a human:", "\n")
              
        for _ in j:
            print(f"- {_}")
        print("\n")

    print(len(easier_sublists))
    print(len(harder_sublists))

def bifracate_easy_hard():
    ml_easier_types_list = list(ml_easier_types)

    for _ in range(0, 10_000):
        random.shuffle(ml_easier_types_list)

    number_of_sublists = 14

    sublists = list(split_list(ml_easier_types_list, number_of_sublists))
    

    for i in sublists:
        print("This is a list of natural language processing and text classification techniques, can you separate them into two groups, where the groups are 'easier to learn as a human' and 'harder to learn as a human'?\n")
        
        for _ in i:
            print(f"- {_}")
            
        print("\n")


def run():
    bifracate_easy_hard()
    # rearrange_easy_hard()
    # print(harder)


if __name__ == "__main__":
    run()
    # check()