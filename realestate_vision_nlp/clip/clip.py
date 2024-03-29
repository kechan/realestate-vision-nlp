from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
from xml.dom import NoDataAllowedErr

import tarfile, PIL, re, os, gc
from io import BytesIO
from tqdm import tqdm
from einops import rearrange
from pathlib import Path

import numpy as np
import tensorflow as tf
import pandas as pd
from transformers import CLIPProcessor, FlaxCLIPModel, CLIPTokenizer
import jax
import jax.numpy as jnp

from realestate_core.common.utils import load_from_pickle, save_to_pickle

### class abstraction for CLIP/HuggingFace model

class FlaxCLIP:
  def __init__(self, model_name: str = 'openai/clip-vit-base-patch32', from_pt=None):
    self.model_name = model_name

    self.model = None # lazy evaluation
    self.processor = None # lazy evaluation
    self.tokenizer = None # lazy evaluation

    self.from_pt = from_pt    # if only pytorch weight available

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
      try:
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
      except:
        self.tokenizer = CLIPProcessor.from_pretrained(self.model_name).tokenizer

    if self.model is None: 
      if self.from_pt is not None:
        self.model = FlaxCLIPModel.from_pretrained(self.model_name, from_pt=self.from_pt)
      else:
        self.model = FlaxCLIPModel.from_pretrained(self.model_name)

    def get_text_features(text_prompts):
      # return normalized vector representation of text prompts
      text_embeddings = self.model.get_text_features(self.tokenizer(text_prompts, padding=True, return_tensors="np").input_ids)
      text_features = text_embeddings / jnp.linalg.norm(text_embeddings, axis=-1, keepdims=True)    
      return text_features

    binary_text_features_list = []
    for k, row in self.prompts_df.iterrows():
      if row.prompt_type == '1:1':   # neutral vs positive pair
        text_features = get_text_features([row.prompt_neutral, row.prompt_positive])

      elif row.prompt_type == '1:M':   # neutral vs. an ensemble of positive prompts
        text_features = jnp.mean(jnp.stack([get_text_features([row.prompt_neutral, p]) for p in row.prompt_positive], axis=0), axis=0)

      else:
        continue

      binary_text_features_list.append(text_features)        

    self.binary_text_features = jnp.stack(binary_text_features_list, axis=0)   # stack (num_prompts, 2, 512)

    self.multi_text_features_list = []
    for k, row in self.prompts_df.iterrows():
      if row.prompt_type == 'M':     # M classes, prediction is argmax over M classes.
        text_features = get_text_features(row.prompt_neutral)      
      else:
        continue

      self.multi_text_features_list.append(text_features)
    

    del binary_text_features_list
    del self.tokenizer
    gc.collect()

    # assert self.text_features.shape == (len(self.text_prompts_list), 2, 512), 'wrong shape'

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

  def get_image_features(self, 
                         photos: List[Union[str, Path, Tuple[str, PIL.Image.Image]]] = None, 
                         ds: tf.data.Dataset = None, 
                         tarfile_path: Union[str, Path] = None,
                         batch_size=64) -> Tuple[List, np.ndarray]:
    '''
    photos: list of images. If str or Path, the full path to the image. 
            If (name, PIL.Image.Image) tuple, then (name of image, image object).

    ds: unbatched tf.data.Dataset of (image_byte, image_name) tuples, image size must be 224x224 and rescaled to [0, 1].

    tarfile_path: path to a tarfile containing images.

    return:
      img_names_list: list of image names
      image_features: np.ndarray of shape (len(photos), 512)
    '''

    if self.processor is None: self.processor = CLIPProcessor.from_pretrained(self.model_name)
    if self.model is None: 
      if self.from_pt is not None:
        self.model = FlaxCLIPModel.from_pretrained(self.model_name, from_pt=self.from_pt)
      else:
        self.model = FlaxCLIPModel.from_pretrained(self.model_name)

    if photos is not None:
      photo_batches = [photos[i:i+batch_size] for i in range(0, len(photos), batch_size)]

      # check if photos is List[Tuple[str, PIL.Image.Image]]
      is_photo_a_tuple = False
      if isinstance(photos[0], tuple) and isinstance(photos[0][0], str) and isinstance(photos[0][1], PIL.Image.Image):
        is_photo_a_tuple = True

      img_names_list = []
      image_features_list = []
      for photo_batch in tqdm(photo_batches):
        if not is_photo_a_tuple:
          imgs = [PIL.Image.open(img_name) for img_name in photo_batch]
          img_names_list += [Path(img_name).name for img_name in photo_batch]
        else:
          imgs = [img for _, img in photo_batch]
          img_names_list += [name for name, _ in photo_batch]

        pixel_values = self.processor(images=imgs, return_tensors="np").pixel_values

        image_features = self.model.get_image_features(pixel_values)
        # image_features = image_embeddings / jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True)

        image_features_list.append(np.array(image_features))

      image_features = np.concatenate(image_features_list, axis=0)
    elif ds is not None:
      # sanity dataset
      for img, name in ds.take(1).as_numpy_iterator():
        assert img.shape == (224, 224, 3)
        assert img.max() <= 1.0
        assert img.min() >= 0.0

      image_mean = np.array(self.processor.feature_extractor.image_mean).astype(np.float32)
      image_std = np.array(self.processor.feature_extractor.image_std).astype(np.float32)

      batch_img_ds = ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

      img_names_list = []
      image_features_list = []

      for imgs, names in tqdm(batch_img_ds.as_numpy_iterator()):
        img_names_list += [name.decode('utf-8') for name in names]

        # pixel_values = self.processor.feature_extractor.normalize(rearrange(imgs, 'b h w c -> b c h w'), mean=image_mean[:, None, None], std=image_std[:, None, None], rescale=True)
        pixel_values = self.processor.feature_extractor.normalize(rearrange(imgs, 'b h w c -> b c h w'), mean=image_mean[:, None, None], std=image_std[:, None, None])
        image_features = self.model.get_image_features(pixel_values)
        # image_features = image_embeddings / jnp.linalg.norm(image_embeddings, axis=-1, keepdims=True)
        image_features_list.append(np.array(image_features))

      image_features = np.concatenate(image_features_list, axis=0)
    elif tarfile_path is not None:
      img_names_list = []
      image_features_list = []

      with tarfile.open(tarfile_path, 'r') as tar:
        members = tar.getmembers()
        
        # batch the members
        member_batches = [members[i:i+batch_size] for i in range(0, len(members), batch_size)]

        for member_batch in tqdm(member_batches):          
          
          imgs = [PIL.Image.open(BytesIO(tar.extractfile(member).read())) for member in member_batch]
          img_names_list += [m.name for m in member_batch]

          pixel_values = self.processor(images=imgs, return_tensors="np").pixel_values

          image_features = self.model.get_image_features(pixel_values)          
          image_features_list.append(np.array(image_features))

      image_features = np.concatenate(image_features_list, axis=0)
    else:
      raise ValueError('photos and ds cannot be both None')
    
    return img_names_list, image_features

  def predict(self, 
              photos: List[Union[str, Path]] = None, 
              image_features: np.ndarray = None, 
              image_names: Union[List, np.ndarray] = None, 
              data_src_name: str = None, 
              batch_size=64) -> pd.DataFrame:
    '''
    photos: list of image paths
    data_src_name: name of the data source, which is written to the output df in a column named 'data_src'

    image_features: un-normalized image clip representation 
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
      image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)

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
   
    for k, (idx, row) in enumerate(self.prompts_df.q_py("prompt_type == '1:1' or prompt_type == '1:M'").iterrows()):
      df[row.item_name] = binary_probs[:, k, 1]

    for k, (idx, row) in enumerate(self.prompts_df.q_py("prompt_type == 'M'").iterrows()):
      for j, c in enumerate(row.prompt_neutral):
        df[c] = multi_probs_list[k][:, j]

    df['data_src'] = data_src_name
    
    # filter out images that mostly likely not kitchen
    # if 'prob_kitchen' in df.columns:
    #   df.drop(index=df.q_py("prob_kitchen < 0.5").index, inplace=True)
    #   df.defrag_index(inplace=True)
    #   df.drop(columns=['prob_kitchen'], inplace=True)

    return df
    

  def compute_feature_score(self, df: pd.DataFrame, scene_type: str):
    # A new column 'features_score' will be added to df

    # This method should be overriden by the subclass, depending on app context
    # features score is the mean prob of specific features (excl. general quality or room type classification)

    if scene_type == 'kitchen':
      # very similar group of features should be locally averaged, before averaging over all features 
    
      # 1) counter tops
      counter_top_scores = np.mean(df[['p_granite_countertop', 'p_marble_countertop', 'p_quartz_countertop']].values, axis=-1, keepdims=True)

      # 2) cabinet
      cabinet_score = np.mean(df[['p_full_height_cabinets', 'p_abundance_of_cabinet_storage', 'p_impressive_custom_kitchen_cabinetry']].values, axis=-1, keepdims=True)

      # 3) lighting
      lighting_score = np.mean(df[['p_recessed_lighting', 'p_large_light_fixture_with_unique_finishes']].values, axis=-1, keepdims=True)

      # 4) island
      island_score = df[['p_kitchen_island']].values

      # 5) stainless steel
      # ss_score = df[['p_stainless_steel']].values

      df['features_score'] = np.mean(np.concatenate([counter_top_scores, lighting_score, island_score], axis=-1), axis=-1)

    elif scene_type == 'bathroom':
      feature_cols = list(self.prompts_df.q_py("class_type == 'feature'").item_name.values)
      df['features_score'] = np.mean(df[feature_cols].values, axis=-1)

    elif scene_type == 'exterior':
      feature_cols = list(self.prompts_df.q_py("class_type == 'feature'").item_name.values)
      df['features_score'] = np.mean(df[feature_cols].values, axis=-1)

    else:
      raise ValueError(f'Unknown scene_type: {scene_type}')    

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

  def save_text_prompts(self, dest_dir: Path, scene_type: str):
    save_to_pickle(self.text_prompts_list, dest_dir/f'{scene_type}_text_prompts_list.pkl')


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