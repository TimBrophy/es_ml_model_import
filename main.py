from elasticsearch import Elasticsearch
from pathlib import Path
from eland.ml.pytorch import PyTorchModel
from eland.ml.pytorch.transformers import TransformerModel


def main():

    # I know this is bad:

    es_cloud_id = "YOUR CLOUD ID HERE"
    es_uname = "YOUR UNAME HERE"
    es_pword = "YOUR PASSWORD HERE"

    model_array = [
        {"model": "dslim/bert-base-NER", "type": "ner"},
        {"model": "typeform/distilbert-base-uncased-mnli", "type": "zero_shot_classification"},
        {"model": "distilbert-base-uncased-finetuned-sst-2-english", "type": "text_classification"},
        {"model": "sentence-transformers/msmarco-MiniLM-L-12-v3", "type": "text_embedding"},
        {"model": "bert-base-uncased", "type": "fill-mask"},
        {"model": "elastic/distilbert-base-cased-finetuned-conll03-english", "type": "ner"}
    ]

    for i in model_array:
        model_name = i["model"]
        model_type = i["type"]

        print("Processing: {}".format(model_name))

        try:
            tm = TransformerModel(model_name, model_type)
            tmp_path = "models"
            Path(tmp_path).mkdir(parents=True, exist_ok=True)
            model_path, config, vocab_path = tm.save(tmp_path)
        except:
            print("{} failed to download properly".format(model_name))

        try:
            es = Elasticsearch(
                cloud_id=es_cloud_id,
                basic_auth=(es_uname, es_pword)
            )
        except:
            print("Cannot create an ES connection")

        try:
            ptm = PyTorchModel(es, tm.elasticsearch_model_id())
            ptm.import_model(model_path=model_path, config_path=None, vocab_path=vocab_path, config=config)
        except:
            print("Could not import {}. Maybe it already exists.".format(model_name))


if __name__ == '__main__':
    main()
