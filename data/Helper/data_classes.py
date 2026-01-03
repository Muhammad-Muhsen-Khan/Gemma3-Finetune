from torch.utils.data import Dataset
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from datasets import load_dataset
import numpy as np
import pickle
import os
import json
from transformers import AutoTokenizer

import random
random.seed(42)

#Dataset
class PatientEntriesDataset(Dataset):
  def __init__(self, data_path, dataframe=None, include_symptoms_only=False, label_mapping=None, labels_to_idx=None):
    super().__init__()

    #Only load data if dataframe is not provided
    if dataframe is not None:
      df = dataframe
      self.labels_to_idx = labels_to_idx

    else:
      #load data
      df = pd.concat([pd.read_csv(dpath,header=0,dtype=str) for dpath in data_path], ignore_index=True)
      df = df[df['Entry'].notna()].reset_index(drop=True) #Remove any nan entries

      #Correct level_0 categories
      self.label_to_category = df[df['Level'] != '0'].set_index('Label')['Category'].to_dict() #Correct label categories
      df['Category'] = df.apply((lambda row: self.label_to_category[row['Label']]), axis=1)  #Corrects label categories for level_0

      #Remove other categories
      if include_symptoms_only:
        df = df[df['Category'] == 'Symptom']

      #Apply label_mapping
      # if label_mapping is not None:
      #   df['Label'] = df['Label'].map(label_mapping)
      df = df[df['Label'].notna()].reset_index(drop=True) #Remove any nan labels

      #Labels and indices
      if labels_to_idx is not None:
        self.labels_to_idx = labels_to_idx

      else:
        self.labels_to_idx = {label: idx for idx, label in enumerate(df['Label'].unique())}
      
      self.idx_to_labels = {idx: label for label, idx in self.labels_to_idx.items()}
      df['Label'] = df['Label'].map(self.labels_to_idx)  #Convert labels to indices

      #Number of classes
      self.n_classes = len(df['Label'].unique())

      #Add Label/Level (Allows stratification across label/level)
      df['Label/Level'] = df.apply(lambda row: f"{row['Label']}_{row['Level']}", axis=1)

      # #Remove level-0 (raw labels & synonyms) -> do not want to include them in test dataset
      self.L0 = df[df['Level'] == '0'].reset_index(drop=True)
      df = df[df['Level'] != '0'].reset_index(drop=True) # removing L0

    self.df = df
    self.xy = df.to_numpy()
    self.x = self.xy[:, 7] #(n_samples, )
    self.y = self.xy[:, 2] #(n_samples, )

    # self.encodings = tokenizer([self.x[idx] for idx in range(len(self.x))], truncation=True, padding=True, return_tensors="pt")
    # self.labels_list = [lbl for lbl in list(set(label_mapping.values())) if lbl not in [None, 'no symptom', 'more']]

    self.n_samples = self.xy.shape[0]

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
      return self.x[idx], self.y[idx]
  
class MD_sl_Dataset(Dataset):
  def __init__(self, data_path, dataframe=None, include_symptoms_only=False, label_mapping=None, labels_to_idx=None, sampling=False):
    super().__init__()

    #Only load data if dataframe is not provided
    if dataframe is not None:
      df = dataframe
      self.labels_to_idx = labels_to_idx

    else:
      #load data
      df = pd.concat([pd.read_csv(dpath,header=0,dtype=str) for dpath in data_path], ignore_index=True)
      df = df[df['Entry'].notna()].reset_index(drop=True) #Remove any nan entries

      #Correct level_0 categories
      self.label_to_category = df[df['Level'] != '0'].set_index('Label')['Category'].to_dict() #Correct label categories
      df['Category'] = df.apply((lambda row: self.label_to_category[row['Label']]), axis=1)  #Corrects label categories for level_0

      #Remove other categories
      if include_symptoms_only:
        df = df[df['Category'] == 'Symptom']

      #Apply label_mapping
      # if label_mapping is not None:
      #   df['Label'] = df['Label'].map(label_mapping)
      df = df[df['Label'].notna()].reset_index(drop=True) #Remove any nan labels

      #Labels and indices
      if labels_to_idx is not None:
        self.labels_to_idx = labels_to_idx

      else:
        self.labels_to_idx = {label: idx for idx, label in enumerate(df['Label'].unique())}
      
      self.idx_to_labels = {idx: label for label, idx in self.labels_to_idx.items()}
      df['Label'] = df['Label'].map(self.labels_to_idx)  #Convert labels to indices

      #Number of classes
      self.n_classes = len(df['Label'].unique())

      #Add Label/Level (Allows stratification across label/level)
      df['Label/Level'] = df.apply(lambda row: f"{row['Label']}_{row['Level']}", axis=1)

      # #Remove level-0 (raw labels & synonyms) -> do not want to include them in test dataset
      self.L0 = df[df['Level'] == '0'].reset_index(drop=True)
      df = df[df['Level'] != '0'].reset_index(drop=True) # removing L0

    self.df = df

    if sampling:
      # df is your existing dataframe (self.df)
      temp_df = df

      k = 10000
      frac = k / len(temp_df)

      _, sample_df = train_test_split(
          temp_df,
          test_size=frac,            # keep only 10k rows
          stratify=df["Label"],
          random_state=42
      )

        # assign back to the dataset
      self.df = sample_df.reset_index(drop=True)
      df = sample_df.reset_index(drop=True)

    self.xy = df.to_numpy()
    self.x = self.xy[:, 7] #(n_samples, )
    self.y = self.xy[:, 2] #(n_samples, )

    # self.encodings = tokenizer([self.x[idx] for idx in range(len(self.x))], truncation=True, padding=True, return_tensors="pt")
    # self.labels_list = [lbl for lbl in list(set(label_mapping.values())) if lbl not in [None, 'no symptom', 'more']]

    self.n_samples = self.xy.shape[0]

  def __len__(self):
    return self.n_samples

  def __getitem__(self, idx):
      labels = [self.y[idx]]
      labels_vector = torch.tensor([1 if lbl_idx in labels else 0 for lbl_idx in range(len(self.labels_to_idx))], dtype=torch.float32)

      return self.x[idx], labels_vector
  

class MD_ml_Dataset(Dataset):
  def __init__(self, data_path, label_mapping, labels_to_idx=None, data_split="train", train_size=6000, val_size=302, test_size=2000):
      #Load data
      df = pd.read_csv(data_path)

      #Apply label_mapping
      df['Label_A'] = df['Label_A'].map(label_mapping)
      df['Label_B'] = df['Label_B'].map(label_mapping)
      df['Label_C'] = df['Label_C'].map(label_mapping)

      #Drop NaN labels
      df = df[df['Label_A'].notna()].reset_index(drop=True)
      df = df[df['Label_B'].notna()].reset_index(drop=True)

      self.labels_to_idx = labels_to_idx

      # Shuffle and split the dataset
      df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle with a fixed random seed
      train_df = df.iloc[:train_size]
      val_df = df.iloc[train_size:train_size + val_size]
      test_df = df.iloc[train_size + val_size:]

      # Assign the correct subset
      if data_split == "train":
          self.df = train_df
      elif data_split == "val":
          self.df = val_df
      elif data_split == "test":
          self.df = test_df
      else:
          raise ValueError("data_split must be 'train', 'val', or 'test'")
      
      self.data = self.df.to_numpy().tolist()

  def __len__(self):
      return self.df.shape[0]

  def __getitem__(self, idx):
      labels = self.data[idx][1:3] if str(self.data[idx][3]) in ['nan', 'None'] else self.data[idx][1:4]
      labels_idxs = [self.labels_to_idx[lbl] for lbl in labels]

      labels_vector = torch.tensor([1 if lbl_idx in labels_idxs else 0 for lbl_idx in range(len(self.labels_to_idx))], dtype=torch.float32)

      return self.data[idx][-1], labels_vector

#MD_sl + MD_ml balanced dataset
class MD_balanced_Dataset(Dataset):
    def __init__(self, data_path, texts=None, labels=None, data_split="train", train_size=0.95):
        if texts is None or labels is None:
          #Load data
          with open(data_path, 'rb') as file:
            data = pickle.load(file)

          texts = data['data']
          labels = data['labels']

        # Shuffle and split the dataset
        indices = np.arange(len(texts))
        np.random.shuffle(indices)
        
        n_samples = int(train_size * len(texts))
        if data_split == "train":
          selected_indices = indices[:n_samples]

        else:
          selected_indices = indices[n_samples:]
        
        # Slice the data and labels
        self.entries = [texts[i] for i in selected_indices]
        self.entry_labels = labels[selected_indices].astype(np.float32)
      
        # self.encodings = tokenizer([self.entries[idx] for idx in range(len(self.entries))], truncation=True, padding=True, return_tensors="pt")
 
    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx], self.entry_labels[idx]

#Reuters Dataset
class reutersDataset(Dataset):
  def __init__(self, data_split, single_or_multi_labels=None, dataset=None, labels_to_idx=None):
    super().__init__()

    if dataset is not None:
        self.dataset = dataset
        self.labels_to_idx = labels_to_idx
        self.idx_to_labels = {idx: label for label, idx in self.labels_to_idx.items()}

        self.n_classes = len(self.labels_to_idx)

    else:
        #Load dataset
        reuters = load_dataset("ucirvine/reuters21578", "ModApte", trust_remote_code=True)

        #Get common labels across train (singles) and test + train(multi) splits
        singles_topics = []

        for split in ["train"]:
            for data in reuters[split]:
                # Filter out entries with empty or None text before processing
                if data.get("text", None) and str(data["text"]).strip() != "":
                    if len(data['topics']) == 1:
                        singles_topics += data['topics']

        # Filter topics that have less than 1% of entry occurrence in the train split

        # # 1. Count topic occurrences in train split (consider only single label entries)
        # topic_counts = {}
        # total_train = 0
        # for data in reuters["train"]:
        #     if len(data['topics']) == 1:
        #         total_train += 1
        #         topic = data['topics'][0]
        #         topic_counts[topic] = topic_counts.get(topic, 0) + 1

        # # 2. Minimum count threshold: 1% of train split (round up to ensure at least 1 if possible)
        # min_count = max(1, int(total_train * 0.01))

        # # 3. Filter singles_topics to only those above the threshold
        # singles_topics = [t for t in singles_topics if topic_counts.get(t, 0) >= min_count]

        singles_topics = sorted(set(singles_topics))
        
        #Number of classes
        self.n_classes = len(singles_topics)

        if labels_to_idx is None:
            self.labels_to_idx = {label: idx for idx, label in enumerate(singles_topics)}
        else:
            self.labels_to_idx = labels_to_idx

        self.idx_to_labels = {idx: label for label, idx in self.labels_to_idx.items()}

        #Split data into single and multi label
        if data_split == "single_train":
            self.dataset = [data for data in reuters['train'] if len(data['topics']) == 1]

        elif data_split == "single_val":
            self.dataset = [data for data in reuters['unused'] if len(data['topics']) == 1]

        elif data_split == "single_test":
            # Filter out entries with labels not in singles_topics (same as multi_test)
            self.dataset = []
            for data in reuters['test']:
                if len(data['topics']) == 1:
                    # Check if the single label is in singles_topics
                    if data['topics'][0] in singles_topics:
                        self.dataset.append(data)   

        elif data_split == "multi_test":
            self.dataset = []

            for split in ["test"]:
                for data in reuters[split]:
                    # if len(data['topics']) > 1:
                    check = True
                    for t in data['topics']:
                        if t not in singles_topics:
                            check = False
                            break

                    if check:
                        self.dataset.append(data)


        else:
            dataset = []
            for split in ["train"]:
                for data in reuters[split]:
                    # if len(data['topics']) > 1:
                    check = True
                    for t in data['topics']:
                        if t not in singles_topics:
                            check = False
                            break

                    if check:
                        dataset.append(data)
            
            random.shuffle(dataset)

            if data_split == "multi_train":
                # self.dataset = dataset
                # Deterministic selection of 700 validation indices
                rng = random.Random(42)
                val_indices = set(rng.sample(range(len(dataset)), 700))
                self.dataset = [data for idx, data in enumerate(dataset) if idx not in val_indices]

            elif data_split == "multi_val":
                # self.dataset = dataset
                # Deterministic selection of 700 validation indices
                rng = random.Random(42)
                val_indices = set(rng.sample(range(len(dataset)), 700))
                self.dataset = [data for idx, data in enumerate(dataset) if idx in val_indices]                    

    # After dataset is initialized, remove entries with empty text
    self.dataset = [data for data in self.dataset if data.get("text", None) and str(data["text"]).strip() != ""]

    self.x = [data['text'] for data in self.dataset]
    self.y = [data['topics'] for data in self.dataset]

    self.n_samples = len(self.dataset)

    self.single_or_multi_labels = single_or_multi_labels

  def __getitem__(self, index):
    labels = [self.labels_to_idx[lbl] for lbl in self.y[index]]

    if self.single_or_multi_labels == "multi":
        labels_vector = torch.tensor([1 if lbl_idx in labels else 0 for lbl_idx in range(len(self.labels_to_idx))], dtype=torch.float32)

    else:
        labels_vector = labels[0]

    return self.x[index], labels_vector

  def __len__(self):
    return self.n_samples

# #Reuters Dataset
# class reutersDataset(Dataset):
#   def __init__(self, data_split, single_or_multi_labels=None, dataset=None, labels_to_idx=None):
#     super().__init__()

#     if dataset is not None:
#         self.dataset = dataset
#         self.labels_to_idx = labels_to_idx
#         self.idx_to_labels = {idx: label for label, idx in self.labels_to_idx.items()}

#         self.n_classes = len(self.labels_to_idx)

#     else:
#         #Load dataset
#         reuters = load_dataset("ucirvine/reuters21578", "ModApte", trust_remote_code=True)

#         #Get common labels across train (singles) and test + train(multi) splits
#         singles_topics = []

#         for split in ["train", "test", "unused"]:
#             for data in reuters[split]:
#                 if len(data['topics']) == 1:
#                     singles_topics += data['topics']

#         singles_topics = list(set(singles_topics)) 
        
#         #Number of classes
#         self.n_classes = len(singles_topics)
#         # self.labels_to_idx = {label: idx for idx, label in enumerate(singles_topics)}
#         # self.idx_to_labels = {idx: label for label, idx in self.labels_to_idx.items()}
#         self.labels_to_idx = labels_to_idx
#         self.idx_to_labels = {idx: label for label, idx in self.labels_to_idx.items()}

#         #Split data into single and multi label
#         if data_split == "single_train":
#             self.dataset = [data for data in reuters['train'] if len(data['topics']) == 1]

#         elif data_split == "single_val":
#             self.dataset = [data for data in reuters['unused'] if len(data['topics']) == 1]

#         elif data_split == "single_test":
#             self.dataset = [data for data in reuters['test'] if len(data['topics']) == 1]   

#         elif data_split == "multi_test":
#             self.dataset = []

#             for split in ["test"]:
#                 for data in reuters[split]:
#                     # if len(data['topics']) > 1:
#                     check = True
#                     for t in data['topics']:
#                         if t not in singles_topics:
#                             check = False
#                             break

#                     if check:
#                         self.dataset.append(data)


#         else:
#             dataset = []
#             for split in ["train"]:
#                 for data in reuters[split]:
#                     # if len(data['topics']) > 1:
#                       check = True
#                       for t in data['topics']:
#                           if t not in singles_topics:
#                               check = False
#                               break

#                       if check:
#                           dataset.append(data)
            
#             random.shuffle(dataset)

#             if data_split == "multi_train":
#                 self.dataset = dataset[:7000]

#             elif data_split == "multi_val":
#                 self.dataset = dataset[7000:]                    

#     # Filter out any entries in self.dataset that are empty before further use
#     self.dataset = [data for data in self.dataset if data.get('text', '').strip() != '']

#     self.x = [data['text'] for data in self.dataset]
#     self.y = [data['topics'] for data in self.dataset]

#     self.n_samples = len(self.dataset)

#     self.single_or_multi_labels = single_or_multi_labels

#   def __getitem__(self, index):
#     labels = [self.labels_to_idx[lbl] for lbl in self.y[index]]

#     if self.single_or_multi_labels == "multi":
#         labels_vector = torch.tensor([1 if lbl_idx in labels else 0 for lbl_idx in range(len(self.labels_to_idx))], dtype=torch.float32)

#     else:
#         labels_vector = labels[0]

#     return self.x[index], labels_vector

#   def __len__(self):
#     return self.n_samples

class MIMICIVDataset(Dataset):
    def __init__(self, data_split, labels_to_idx=None):
        super().__init__()

        rawnotes_dir = "data/Datasets/mimiciv/rawnotes"
        note_files = os.listdir(rawnotes_dir)
        # Sort files to ensure consistent ordering across different machines/filesystems
        note_files = sorted(note_files)
        
        ground_truth_file = "data/Datasets/mimiciv/standardized_385_ground_truth.json"
        with open(ground_truth_file, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)

        codes = pd.read_csv('data/Datasets/mimiciv/all_icd10cm_codes.csv', encoding='utf-8')

        codes2desc = {row['code']: row['text'] for _, row in codes.iterrows()}

        # Shuffle note_files to randomize (with fixed seed for reproducibility)
        shuffled_files = note_files.copy()
        random.seed(42)  # Ensure deterministic shuffle
        random.shuffle(shuffled_files)

        test_size = max(1, int(0.1 * len(shuffled_files)))
        test_files = shuffled_files[:test_size]
        train_files = shuffled_files[test_size:]

        # Load test notes and ground truth
        test_texts = []
        test_ground_truths = []
        
        for filename in test_files:
            # Join rawnotes_dir with filename to get full path
            file_path = os.path.join(rawnotes_dir, filename)
            
            # Load text content from file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            test_texts.append(text_content)
            
            # Strip last 4 characters (".txt") to get note_id
            note_id = filename[:-4]
            
            # Load ground truth using note_id as key
            if note_id in ground_truth:
                test_ground_truths.append(list(set(list([i[:3] for i in ground_truth[note_id].keys() if i in codes2desc.keys()]))))
            else:
                print(f"No ground truth found for note {note_id}")
        
        # Load train notes and ground truth
        train_texts = []
        train_ground_truths = []
        
        for filename in train_files:
            # Join rawnotes_dir with filename to get full path
            file_path = os.path.join(rawnotes_dir, filename)
            
            # Load text content from file
            with open(file_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            train_texts.append(text_content)
            
            # Strip last 4 characters (".txt") to get note_id
            note_id = filename[:-4]
            
            # Load ground truth using note_id as key
            if note_id in ground_truth:
                train_ground_truths.append(list(set(list([i[:3] for i in ground_truth[note_id].keys() if i in codes2desc.keys()]))))
            else:
                print(f"No ground truth found for note {note_id}")
        
        # Create sets of codes present in train and test ground truths
        train_codes_set = set()
        for codes in train_ground_truths:
            train_codes_set.update(codes)

        test_codes_set = set()
        for codes in test_ground_truths:
            test_codes_set.update(codes)

        # Only keep codes that appear in both train and test
        shared_codes_set = train_codes_set.intersection(test_codes_set)

        # Filter each entry in test_ground_truths to only include codes present in shared_codes_set
        filtered_test_ground_truths = []
        for codes in test_ground_truths:
            filtered_codes = [code for code in codes if code in shared_codes_set]
            filtered_test_ground_truths.append(filtered_codes)
        test_ground_truths = filtered_test_ground_truths

        # Similarly filter each entry in train_ground_truths to only include codes present in shared_codes_set
        filtered_train_ground_truths = []
        for codes in train_ground_truths:
            filtered_codes = [code for code in codes if code in shared_codes_set]
            filtered_train_ground_truths.append(filtered_codes)
        train_ground_truths = filtered_train_ground_truths

        test_unique_labels = set(code for labels in test_ground_truths for code in labels)
        train_unique_labels = set(code for labels in train_ground_truths for code in labels)

        print(f"Number of unique labels in test_ground_truths: {len(test_unique_labels)}")
        print(f"Number of unique labels in train_ground_truths: {len(train_unique_labels)}")

        avg_labels_train = sum(len(labels) for labels in train_ground_truths) / len(train_ground_truths) if train_ground_truths else 0
        avg_labels_test = sum(len(labels) for labels in test_ground_truths) / len(test_ground_truths) if test_ground_truths else 0

        print(f"Average labels per entry (train): {avg_labels_train:.2f}")
        print(f"Average labels per entry (test): {avg_labels_test:.2f}")
        
        # Calculate overall average correctly: total labels / total entries
        total_labels = sum(len(labels) for labels in train_ground_truths) + sum(len(labels) for labels in test_ground_truths)
        total_entries = len(train_ground_truths) + len(test_ground_truths)
        avg_labels_total = total_labels / total_entries if total_entries > 0 else 0
        print(f"Average labels per entry (total): {avg_labels_total:.2f}")

        print(f"Number of train entries: {len(train_ground_truths)}")
        print(f"Number of test entries: {len(test_ground_truths)}")

        # Count tokens using a tokenizer
        tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
        train_tokens = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in train_texts)
        test_tokens = sum(len(tokenizer.encode(text, add_special_tokens=False)) for text in test_texts)
        total_tokens = train_tokens + test_tokens
        
        print(f"Total tokens in train notes: {train_tokens:,}")
        print(f"Total tokens in test notes: {test_tokens:,}")
        print(f"Total tokens across train and test notes: {total_tokens:,}")

        if labels_to_idx is None:
            labels_to_idx = {code: idx for idx, code in enumerate(sorted(list(train_unique_labels)))}
            self.labels_to_idx = labels_to_idx
            self.idx_to_labels = {idx: code for code, idx in labels_to_idx.items()}
        else:
            self.labels_to_idx = labels_to_idx
            self.idx_to_labels = {idx: code for code, idx in labels_to_idx.items()}

        self.test_texts = test_texts
        self.test_ground_truths = test_ground_truths
        self.train_texts = train_texts
        self.train_ground_truths = train_ground_truths

        if data_split == "test":
            self.texts = test_texts
            self.ground_truths = test_ground_truths
        else:
            self.texts = train_texts
            self.ground_truths = train_ground_truths

        self.n_classes = len(self.labels_to_idx)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        labels = [self.labels_to_idx[lbl] for lbl in self.ground_truths[index]]
        labels_vector = torch.tensor([1 if lbl_idx in labels else 0 for lbl_idx in range(len(self.labels_to_idx))], dtype=torch.float32)

        return self.texts[index], labels_vector