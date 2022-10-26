from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
from xml.dom import NoDataAllowedErr

import PIL, re, os, gc
from tqdm import tqdm
from einops import rearrange
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import CLIPProcessor, FlaxCLIPModel, CLIPTokenizer
import jax
import jax.numpy as jnp

from realestate_core.common.utils import load_from_pickle, save_to_pickle

### class abstraction for CLIP/HuggingFace model

class FlaxCLIP:
  def __init__(self, model_name: str = 'openai/clip-vit-base-patch32'):
    self.model_name = model_name

    self.model = None # lazy evaluation
    self.processor = None # lazy evaluation
    self.tokenizer = None # lazy evaluation

  def set_text_prompts_list(self, text_prompts_list: List[str]):
    '''
    text_prompts_list: text prompts with the following data structure:

    text_prompts_list = [
      {'item_name': 'p_granite_countertop','prompt_neutral': "a photo of a kitchen", 'prompt_positive': "a photo of a kitchen with beautiful granite counter top.", "prompt_type": "feature"},
      {'item_name': 'p_abundance_of_cabinet_storage', 'prompt_neutral': "a photo of a kitchen", 'prompt_positive': "a photo of a kitchen with abundance of cabinet storage.", "prompt_type": "feature"},
      {'item_name': 'p_impressive_custom_kitchen_cabinetry', 'prompt_neutral': "a photo of a kitchen", 'prompt_positive': "a photo of a kitchen with beautiful impressive custom kitchen cabinetry.", "prompt_type": "feature"},
      {'item_name': 'p_hardwood_flooring', 'prompt_neutral': "a photo of a kitchen", 'prompt_positive': "a photo of a kitchen with hardwood flooring.", "prompt_type": "feature"},
      
      {'item_name': 'p_excellent_kitchen', 
        'prompt_neutral': "a photo of a kitchen", 
        'prompt_positive': ["a photo of a beautiful gourmet kitchen.", 
                            "a photo of a dream kitchen."], 
        "prompt_type": "ensemble_quality"},

        {'item_name': 'p_room', 
        'prompt_neutral': ["a photo of a living room.", 
                            "a photo of a bathroom.", 
                            "a photo of a kitchen.", 
                            "a photo of a dining room.", 
                            "a photo of a garage.", 
                            "a photo of a laundry room."], 
        'prompt_positive': None, 
        "prompt_type": "multi_scene"},
        
        {'item_name': 'p_indoor', 'prompt_neutral': "An outdoor photo", 'prompt_positive': "An indoor photo", "prompt_type": "scene"}
    ]
    
    Set self.text_prompts_list and compute self.text_features
    '''

    self.text_prompts_list = text_prompts_list
    self.prompts_df = pd.DataFrame(self.text_prompts_list)

    # compute text embeddings
    try:
      if self.tokenizer is None: self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
    except:
      self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)

    if self.model is None: self.model = FlaxCLIPModel.from_pretrained(self.model_name)

    def get_text_features(text_prompt_pair):
      text_embeddings = self.model.get_text_features(self.tokenizer(text_prompt_pair, padding=True, return_tensors="np").input_ids)
      text_features = text_embeddings / jnp.linalg.norm(text_embeddings, axis=-1, keepdims=True)    
      return text_features

    binary_text_features_list = []
    for k, row in self.prompts_df.iterrows():
      if row.prompt_type == 'feature' or row.prompt_type == 'quality' or row.prompt_type == 'scene':
        text_features = get_text_features([row.prompt_neutral, row.prompt_positive])

      elif row.prompt_type == 'ensemble_quality':
        text_features = jnp.mean(jnp.stack([get_text_features([row.prompt_neutral, p]) for p in row.prompt_positive], axis=0), axis=0)

      else:
        continue

      binary_text_features_list.append(text_features)        

    self.binary_text_features = jnp.stack(binary_text_features_list, axis=0)   # stack (num_prompts, 2, 512)

    self.multi_text_features_list = []
    for k, row in self.prompts_df.iterrows():
      if row.prompt_type == 'multi_scene':
        text_features = get_text_features(row.prompt_neutral)      
      else:
        continue

      self.multi_text_features_list.append(text_features)
    

    del binary_text_features_list
    del self.tokenizer
    gc.collect()

    # assert self.text_features.shape == (len(self.text_prompts_list), 2, 512), 'wrong shape'

  def set_specific_feature_col_ids(self, specific_feature_col_ids: List[int]):
    self.specific_feature_col_ids = specific_feature_col_ids

  def set_general_quality_col_ids(self, general_quality_col_ids: List[int]):
    self.general_quality_col_ids = general_quality_col_ids

  def _preprocess_and_cache_image_npy(self, photos: List[Union[str, Path]], cache_file_prefix: str, batch_size=4096):
    '''
    This is done as a temporary measure to improve GPU utilization by 
    preprocessing the images and caching npy to disk by a 4-CPU instance, and
    later run the inference portion on a 1-GPU instance separately.

    TODO: find a better way to parallelize data preprocessing from CPU
    '''
    if self.processor is None: self.processor = CLIPProcessor.from_pretrained(self.model_name)

    photo_batches = [photos[i:i+batch_size] for i in range(0, len(photos), batch_size)]

    img_names_list = []
    for k, photo_batch in tqdm(enumerate(photo_batches)):
      img_names_list += [Path(img_name).name for img_name in photo_batch]
      
      imgs = [PIL.Image.open(img_name) for img_name in photo_batch]
      pixel_values = self.processor(images=imgs, return_tensors="np").pixel_values

      np.savez_compressed(f'{cache_file_prefix}_{k}', pixel_values=pixel_values)

    save_to_pickle(img_names_list, f'{cache_file_prefix}_img_names_list.pkl')

  def _predict_from_npy(self, cache_file_prefix: str, batch_size=64) -> pd.DataFrame:
    assert self.text_features is not None, 'text_features not set'

    npz_files = [str(f) for f in Path('.').lf(f'{cache_file_prefix}*.npz')]                
    npz_files = sorted(npz_files, key=lambda x: int(re.search(r'.*?_(\d+).npz', x).group(1)))

    probs_list = []
    for f in npz_files:
      print(f)
      pixel_values = np.load(f)['pixel_values']

      for i in tqdm(range(0, len(pixel_values), batch_size)):
        # print(i, i+batch_size)
        pixel_values_batch = pixel_values[i:i+batch_size]

        image_embeddings = self.model.get_image_features(pixel_values_batch)
        image_features = image_embeddings / jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True)  # normalize

        probs = jax.nn.softmax(100 * jnp.einsum('mc, ftc -> fmt', image_features, self.text_features), axis=-1)
        probs = rearrange(probs, 'f b c -> b f c')

        probs_list.append(np.array(probs))

    probs = np.concatenate(probs_list, axis=0)

    del probs_list 
    gc.collect()

    assert probs.shape[1:] == (len(self.text_prompts_list), 2)

    img_names_list = load_from_pickle(f'{cache_file_prefix}_img_names_list.pkl')

    df = pd.DataFrame(data={'img_name': img_names_list})
    for k, col in enumerate([self._prompt_to_colname(t[-1]) for t in self.text_prompts_list]):
        df[col] = probs[:, k, 1]
    df['data_src'] = cache_file_prefix

    # features score is the mean prob of specific features (excl. general qualify or room type classification)
    df['features_score'] = np.mean(df[[self._prompt_to_colname(t[-1]) for i, t in enumerate(self.text_prompts_list) if i in self.specific_feature_col_ids]].values, axis=-1)

    # filter out images that mostly likely not kitchen
    if 'prob_kitchen' in df.columns:
      df.drop(index=df.q_py("prob_kitchen < 0.5").index, inplace=True)
      df.defrag_index(inplace=True)
      df.drop(columns=['prob_kitchen'], inplace=True)

    return df

  def get_image_features(self, photos: List[Union[str, Path]], batch_size=64) -> Tuple[List, np.ndarray]:
    '''
    photos: list of image paths

    return:
      img_names_list: list of image names
      image_features: np.ndarray of shape (len(photos), 512)
    '''

    if self.processor is None: self.processor = CLIPProcessor.from_pretrained(self.model_name)
    if self.model is None: self.model = FlaxCLIPModel.from_pretrained(self.model_name)

    photo_batches = [photos[i:i+batch_size] for i in range(0, len(photos), batch_size)]

    img_names_list = []
    image_features_list = []
    for photo_batch in tqdm(photo_batches):
      imgs = [PIL.Image.open(img_name) for img_name in photo_batch]
      img_names_list += [Path(img_name).name for img_name in photo_batch]

      pixel_values = self.processor(images=imgs, return_tensors="np").pixel_values

      image_embeddings = self.model.get_image_features(pixel_values)
      image_features = image_embeddings / jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True)

      image_features_list.append(np.array(image_features))

    image_features = np.concatenate(image_features_list, axis=0)
    # np.savez_compressed(f'clip_image_features_{data_src_name}', files=img_names_list, image_features=image_features)
    
    return img_names_list, image_features

  def predict(self, photos: List[Union[str, Path]] = None, image_features: np.ndarray = None, image_names: Union[List, np.ndarray] = None, data_src_name: str = None, batch_size=64) -> pd.DataFrame:
    '''
    photos: list of image paths
    data_src_name: name of the data source, which is written to the output df in a column named 'data_src'
    '''
    assert self.text_prompts_list is not None, 'set_text_prompts_list not ran.'
    
    if photos is not None:

      if self.processor is None: self.processor = CLIPProcessor.from_pretrained(self.model_name)

      photo_batches = [photos[i:i+batch_size] for i in range(0, len(photos), batch_size)]

      img_names_list, probs_list = [], []
      for photo_batch in tqdm(photo_batches):
        imgs = [PIL.Image.open(img_name) for img_name in photo_batch]
        img_names_list += [Path(img_name).name for img_name in photo_batch]

        pixel_values = self.processor(images=imgs, return_tensors="np").pixel_values    
        
        image_embeddings = self.model.get_image_features(pixel_values)

        image_features = image_embeddings / jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True)  # normalize

        probs = jax.nn.softmax(100 * jnp.einsum('mc, ftc -> fmt', image_features, self.text_features), axis=-1)
        probs = rearrange(probs, 'f b c -> b f c')
        
        probs_list.append(np.array(probs))

      probs = np.concatenate(probs_list, axis=0)

      del probs_list 
      gc.collect()

      assert probs.shape[1:] == (len(self.text_prompts_list), 2)

    elif image_features is not None and image_names is not None:
      img_names_list = image_names
      binary_probs = jax.nn.softmax(100 * jnp.einsum('mc, ftc -> fmt', image_features, self.binary_text_features), axis=-1)
      binary_probs = rearrange(binary_probs, 'f b c -> b f c')
      binary_probs = np.array(binary_probs)

      multi_probs_list = []
      for multi_text_features in self.multi_text_features_list:
        multi_probs = jax.nn.softmax(100 * jnp.einsum('mc, tc -> mt', image_features, multi_text_features), axis=-1)
        multi_probs = np.array(multi_probs)

        multi_probs_list.append(multi_probs)

    else:
      raise ValueError('Either photos or image_features and image_names must be provided')

    df = pd.DataFrame(data={'img_name': img_names_list})
   
    for k, (idx, row) in enumerate(self.prompts_df.q_py("prompt_type == 'feature' or prompt_type == 'quality' or prompt_type == 'scene' or prompt_type == 'ensemble_quality'").iterrows()):
      df[row.item_name] = binary_probs[:, k, 1]

    for k, (idx, row) in enumerate(self.prompts_df.q_py("prompt_type == 'multi_scene'").iterrows()):
      for j, c in enumerate(row.prompt_neutral):
        df[c] = multi_probs_list[k][:, j]

    df['data_src'] = data_src_name

    # features score is the mean prob of specific features (excl. general qualify or room type classification)
    # df['features_score'] = np.mean(df[[self._prompt_to_colname(t[-1]) for i, t in enumerate(self.text_prompts_list) if i in self.specific_feature_col_ids]].values, axis=-1)

    feature_cols = list(self.prompts_df.q_py("prompt_type == 'feature'").item_name.values)
    df['features_score'] = np.mean(df[feature_cols].values, axis=-1)
    
    # filter out images that mostly likely not kitchen
    # if 'prob_kitchen' in df.columns:
    #   df.drop(index=df.q_py("prob_kitchen < 0.5").index, inplace=True)
    #   df.defrag_index(inplace=True)
    #   df.drop(columns=['prob_kitchen'], inplace=True)


    return df
    

    

  def reset_text_prompts(self):
    self.text_prompts_list = None
    self.binary_text_features = None
    self.multi_text_features_list = None
    gc.collect()


  def cleanup(self, photos: List[Union[str, Path]], cache_file_prefix: str):
    # clean all
    # for f in Path('.').lf('clip_*_df'): os.remove(f)
    # for f in Path('.').lf('*.npz'): os.remove(f)
    for f in photos: os.remove(f)
    
    # if Path(f'{cache_file_prefix}_img_names_list.pkl').exists():
    #   os.remove(f'{cache_file_prefix}_img_names_list.pkl')

  def save_text_prompts_to_prob_cols(self, dest_dir: Path):
    save_to_pickle(self.text_prompts_list, dest_dir/'kitchen_text_prompts_list.pkl')

  @staticmethod
  def prompt_to_colname(self, prompt):
    x = 'prob_' + prompt.replace('a photo of a kitchen with ', '').replace('a photo of a ', '').replace(' ', '_').replace('.', '')
    return x

if __name__ == '__main__':
  # test
  model_name = 'openai/clip-vit-base-patch32'
  model = FlaxCLIP(model_name)

  text_prompts_list = [
    ["a photo of a kitchen", "a photo of a kitchen with beautiful granite counter top."],
    ["a photo of a kitchen", "a photo of a kitchen with a large beautiful kitchen island."],  
    ["a photo of a kitchen", "a photo of a kitchen with beautiful full height cabinets."],
    ["a photo of a kitchen", "a photo of a kitchen with beautiful recessed lighting."],   
    ["a photo of a kitchen", "a photo of a beautiful gourmet kitchen."],   
    ["a photo of a kitchen", "a photo of a kitchen with large light fixture with unique finishes."],
    ["a photo of a kitchen", "a photo of a kitchen with abundance of cupboard storage."],
    ["a photo of a kitchen", "a photo of a kitchen with beautiful impressive custom kitchen cabinetry."],
    ["a photo of a kitchen", "a photo of a kitchen with beautiful wine fridge."],
    ["a photo of a room", "a photo of a kitchen."]
  #     ["a photo of a bathroom", "a photo of a kitchen."]
  ]

  specific_feature_col_ids = [0, 1, 2, 3, 5, 6, 7, 8]
  general_quality_col_ids = [4, 9]

  model.set_text_prompts_list(text_prompts_list)
  model.set_specific_feature_col_ids(specific_feature_col_ids)
  model.set_general_quality_col_ids(general_quality_col_ids)

  model.preprocess_and_cache_image_npy(['./test.jpg'], 'test')
  df = model.predict_from_npy('test')

  print(df)
  model.cleanup()    