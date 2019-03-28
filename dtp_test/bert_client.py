from bert_serving.client import BertClient


bert_client = BertClient(port=5555, port_out=5556)

# start server
# /Library/Frameworks/Python.framework/Versions/3.7/bin/bert-serving-start -num_worker=1 -model_dir=/Users/zyzhao/Downloads/chinese_L-12_H-768_A-12/
