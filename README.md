# CNN on BERT Embeddings

Testing the performance of CNN and pretrained BERT embeddings on the GLUE Tasks

# BERT Model

The BERT model used is the BERT-Base, Uncased: 12-layer, 768-hidden, 12-heads, 110M parameters

it is available at https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1

replace the https://tfhub.dev with https://storage.googleapis.com/tfhub-modules and append a .tar.gz and place it in ./data/bert_module

the tokenization.py file is from the google-research/bert repo https://github.com/google-research/bert

# GLUE tasks

execute <code>python ./utils/download_glue_data.py --data_dir ./data/glue_data</code> to download the tasks.

execute <code>python ./utils/preprocess_tasks.py --data_dir ./data/glue_data</code> to process the tsv files so pandas can read them.

# Keras template

the keras template is based on https://github.com/Ahmkel/Keras-Project-Template
