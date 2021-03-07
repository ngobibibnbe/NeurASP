import argparse

import os
from spert.evaluator import Evaluator
from spert.entities import Dataset, Document, EntityType
from spert.input_reader import BaseInputReader, JsonInputReader
from typing import List, Tuple, Dict
from transformers import BertTokenizer
from spert import prediction

parser = argparse.ArgumentParser()
parser.add_argument("path", help="put the path containing NeurASP 1 and 2 and Spert predictions")
parser.add_argument("database", help="put the database spert predictions")

#parser.add_argument("threshold",type=int, help="threshold represent the threshold on which you want to test")
#parser.add_argument("no_neurASP_threshold",type=int, help="no_neurASP_threshold represent the no_neurASP_threshold on which you want to test")
args = parser.parse_args()
if args.path :
  path = args.path
  #pred_path = "prediction"
  #neur_path = 'neurASP_predictions_valid_epoch_16.json'
  print(args.path)

dataset_path = 'data/datasets/conll04/conll04_train.json'
model_path = 'bert-base-cased'
types_path = 'data/datasets/conll04/conll04_types.json'

if args.database =="scierc":    
  dataset_path = 'data/datasets/scierc/scierc_test.json'
  types_path = 'data/datasets/scierc/scierc_types.json'

max_span_size = 10
log_path = '.'
dataset_label = 'test'
examples_path = 'examples'
example_count = 4
rel_filter_threshold = 0.5


tokenizer = BertTokenizer.from_pretrained(model_path,
                                          do_lower_case=True,
                                          cache_dir='cache')

reader = JsonInputReader(types_path, tokenizer,
                            max_span_size=max_span_size, logger=True)
class CustomEvaluator(Evaluator):

    def __init__(self, pred_dataset: Dataset, dataset: Dataset):
        
        # super(CustomEvaluator, self).__init__(dataset, input_reader, text_encoder,
        #         rel_filter_threshold, no_overlapping,
        #         predictions_path, examples_path, example_count)
        self._gt_entities = []
        self._gt_relations = []

        self._pred_entities = []
        self._pred_relations = []
        self._pseudo_entity_type = EntityType('Entity', 1, 'Entity', 'Entity')
        self._convert_gt(dataset.documents)
        self._convert_gt(pred_dataset.documents, val=False)
        

    def _convert_gt(self, docs: List[Document], val=True):
        for doc in docs:
            relations = doc.relations
            entities = doc.entities

            # convert ground truth relations and entities for precision/recall/f1 evaluation
            sample_entities = [entity.as_tuple() for entity in entities]
            sample_relations = [rel.as_tuple() for rel in relations]

            # if self._no_overlapping:
            #     sample_entities, sample_relations = prediction.remove_overlapping(sample_entities,
            #                                                                             sample_relations)
            if val:
                self._gt_entities.append(sample_entities)
                self._gt_relations.append(sample_relations)
            else:
                self._pred_entities.append(sample_entities)
                self._pred_relations.append(sample_relations)
from glob import glob
test_dataset = reader.read(dataset_path, dataset_label)


dir = path+'/predictions_valid_epoch_*.json'


print("***********************************",dir)
for prediction in glob(str(dir)):
    print("***")
    #prediction = path+'/predictions_valid_epoch_20.json'
    neur_pred = path+'/neurASP_'+os.path.basename(prediction)
    neur2_pred = path+'/neurASP-FE_'+os.path.basename(prediction)
    neur3_pred = path+'/neurASP-FE-SUP_'+os.path.basename(prediction)
    neur4_pred = path+'/neurASP-SUP_'+os.path.basename(prediction)
    print("*****************************************", prediction)

    if os.path.isfile(neur4_pred):
        pred_dataset = reader.read(prediction, dataset_label)
        """neur_dataset = reader.read(neur_pred, dataset_label)
        neur2_dataset = reader.read(neur2_pred, dataset_label)
        neur3_dataset = reader.read(neur3_pred, dataset_label)"""

        print('#################', os.path.basename(prediction))
        evaluator = CustomEvaluator(pred_dataset, test_dataset)
        evaluator.compute_scores()

        """print('#################', os.path.basename(neur_pred))
        evaluator = CustomEvaluator(neur_dataset, test_dataset)
        evaluator.compute_scores()
        
        print('#################', os.path.basename(neur2_pred))
        evaluator = CustomEvaluator(neur2_dataset, test_dataset)
        evaluator.compute_scores()

        print('#################', os.path.basename(neur3_pred))
        evaluator = CustomEvaluator(neur3_dataset, test_dataset)
        evaluator.compute_scores()"""
        
        neur4_dataset = reader.read(neur4_pred, dataset_label)
        print('#################', os.path.basename(neur4_pred))
        evaluator = CustomEvaluator(neur4_dataset, test_dataset)
        evaluator.compute_scores()
    else:
        print(prediction, 'not have coresponding neurASP_'+prediction, 'files')
