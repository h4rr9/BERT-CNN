# CNN on BERT Embeddings

Testing the performance of CNN and pretrained BERT embeddings on the GLUE Tasks

# BERT Model

The BERT model used is the BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters

and BERT-Large, Uncased: 24-layer, 1024-hidden, 16-heads, 340M parameters

Bert Base available at https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip

Bert Large available at https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip

the tokenization.py file is from the google-research/bert repo https://github.com/google-research/bert

# GLUE tasks

execute <code>python ./utils/download_glue_data.py --data_dir ./data/glue_data</code> to download the tasks.

execute <code>python ./utils/preprocess_tasks.py --data_dir ./data/glue_data</code> to process the tsv files so pandas can read them.

# Keras template

the keras template is based on https://github.com/Ahmkel/Keras-Project-Template

# Keras implementaion of BERT

 The keras implementation of BERT is based on https://github.com/CyberZHG/keras-bert
