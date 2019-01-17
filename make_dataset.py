from make_simple_bert_data import make_dataset
import pickle
data = make_dataset("/mnt/storage1/Data/BERT/uncased_L-12_H-768_A-12/bert_config.jsonl", "/mnt/storage1/Data/BERT/uncased_L-12_H-768_A-12/vocab.txt", "/mnt/storage1/Data/BERT/uncased_L-12_H-768_A-12/bert_model.ckpt", 64, 512)

with open ("simple_bert_dataset_bow.pkl" , "wb") as pkl:
    pickle.dump(data , pkl)