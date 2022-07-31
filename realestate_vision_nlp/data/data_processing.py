import shutil, re, os
from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
from pathlib import Path

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from datetime import datetime

from tfrecord_helper.tfrecord_helper import TFRecordHelper, TFRecordHelperWriter 
from realestate_nlp.common.run_config import home, bOnColab
from realestate_nlp.common.util import join_df

from realestate_vision.common.utils import get_listingId_from_image_name

from realestate_nlp.ner.data.data_loader import load_avm_prod_snapshot
try:
  from preproc_listing_qq import filter_for_listingType_SALE, fillna_with_empty_dict, isNone_or_NaN, flatten_dict_col, proc_quality, cleanup_livescoring, get_presented_best_estimate
  from preproc_listing_qq import fix_ON_provState, price_addr_filter
except:
  print(f'failed to import preproc_listing_qq')
  print(f"Please make sure home/'AVMDataAnalysis'/'monitoring' is in your python search path")

BATCH_SIZE = 100   # want this to be a perfect square 
SQRT_BATCH_SIZE = int(np.sqrt(BATCH_SIZE))

INOUT_DOOR_CLASS_NAMES = ['indoor', 'other', 'outdoor']
ROOM_CLASS_NAMES = ['basement', 'bathroom', 'bedroom', 'dining_room', 'garage', 'gym_room', 'kitchen', 'laundry_room', 'living_room', 'office', 'other', 'storage']

BOOLEAN_FEATURE_CLASS_NAMES = ['fireplace', 'agpool', 'body_of_water', 'igpool', 'balcony', 'deck_patio_veranda', 'ss_kitchen', 'double_sink', 'upg_kitchen']


def source_from_new_image_tagging(bigstack_tfrecord: str, image_destination_dir: str = None) -> Tuple[pd.DataFrame, List, List]:

  '''
  return a prediction (tagging by trained model) dataframe, and saved individual images to a destination directory
  if image_destination_dir is None, then no images are saved
  '''
  bigstack_tfrecord = str(bigstack_tfrecord)    # ensure a str, if this was a Path
  spec = TFRecordHelper.element_spec(tf.data.TFRecordDataset(bigstack_tfrecord), return_keys_only=True)
  print(spec)

  features = {
    'filenames': TFRecordHelper.DataType.VAR_STRING_ARRAY,
    'image_raw': TFRecordHelper.DataType.STRING,
    'orig_aspect_ratios': TFRecordHelper.DataType.VAR_FLOAT_ARRAY
  }

  parse_fn = TFRecordHelper.parse_fn(features)
  tile_img_ds = tf.data.TFRecordDataset(bigstack_tfrecord).map(parse_fn).map(lambda x: (x['filenames'], tf.image.decode_jpeg(x['image_raw'], channels=3), x['orig_aspect_ratios'])) 
    
  img_ds = tile_img_ds.map(unstack).unbatch()

  # rearrange data tuple position
  img_ds = img_ds.map(lambda fname, img, r: (img, fname, r))

  batch_size = 32
  batch_img_ds = img_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

  filenames = []
  for x in batch_img_ds.as_numpy_iterator():
    filenames += list(x[1])

  # in/outdoor and room type predictions
  
  model = load_model(home/'ListingImageClassification'/'training'/'hydra_all'/'resnet50_hydra_all.acc.0.9322.h5', compile=False)

  yhats = model.predict(batch_img_ds)
  iodoor_yhats = yhats[0]
  room_yhats = yhats[1]
  '''
  fireplace_yhats = yhats[2]
  agpool_yhats = yhats[3]
  body_of_water_yhats = yhats[4]
  igpool_yhats = yhats[5]
  balcony_yhats = yhats[6]
  deck_patio_veranda_yhats = yhats[7]
  ss_kitchen_yhats = yhats[8]
  double_sink_yhats = yhats[9]
  upg_kitchen_yhats = yhats[10]
  # room_top_3 = tf.math.top_k(room_yhats, k=3)
  '''

  inoutdoor = [INOUT_DOOR_CLASS_NAMES[int(y)] for y in np.squeeze(np.argmax(iodoor_yhats, axis=-1))]

  predictions_df = pd.DataFrame(data={
      'img': filenames,
      'inoutdoor': inoutdoor,
      'p_iodoor': np.round(np.max(iodoor_yhats, axis=-1).astype('float'), 4),
    
      'room': [ROOM_CLASS_NAMES[int(y)] for y in np.squeeze(np.argmax(room_yhats, axis=-1))],
      'p_room': np.round(np.max(room_yhats, axis=-1).astype('float'), 4),
    })

  # exterior type predictions
  exteriors_model_files = [
                           'resnet50_distill_exteriors.acc.0.9106.h5',
                           'resnet50_distill_exteriors.acc.0.9114.h5',
                           'resnet50_distill_exteriors.acc.0.9131.h5',
                           'resnet50_distill_exteriors.acc.0.9116.h5',
                           'resnet50_distill_exteriors.acc.0.9072.h5'
                           ]
  exteriors_labels = ['facade', 'backyard', 'view', 'exterior']

  @tf.autograph.experimental.do_not_convert
  def resize_img(img, filename, r):
    return tf.image.resize(img, (img_height, img_width), method=tf.image.ResizeMethod.BILINEAR)

  y_preds = []
  for m in exteriors_model_files:
    model = load_model(home/'ListingImageClassification'/'training'/'exteriors'/m)
    img_height, img_width = model.input.shape[1:3]

    # resized_batch_img_ds = batch_img_ds.map(lambda img, filename, r: tf.image.resize(img, (img_height, img_width), method=tf.image.ResizeMethod.BILINEAR), num_parallel_calls=AUTO)
    resized_batch_img_ds = batch_img_ds.map(resize_img, num_parallel_calls=tf.data.AUTOTUNE)

    y_pred = model.predict(resized_batch_img_ds)
    y_pred = tf.nn.sigmoid(y_pred).numpy()     # distill model return logits
    y_preds.append(y_pred)

  y_preds = np.stack(y_preds)
  y_pred = np.mean(y_preds, axis=0)

  exterior_predictions_df = pd.DataFrame(data={
        'img': filenames,
        'p_facade': np.round(y_pred[:, 1].astype('float'), 4),    # first element is an indicator for hard vs. soft labels during training, should ignore during inference.
        'p_backyard': np.round(y_pred[:, 2].astype('float'), 4),
        'p_view': np.round(y_pred[:, 3].astype('float'), 4),
        'p_exterior': np.round(y_pred[:, 4].astype('float'), 4)
      })

  # cleanup
  predictions_df.img = predictions_df.img.apply(lambda x: x.decode('utf-8'))
  exterior_predictions_df.img = exterior_predictions_df.img.apply(lambda x: x.decode('utf-8'))

  predictions_df.drop(index=predictions_df.q_py("img == ''").index, inplace=True)   # drop img with no names
  predictions_df.defrag_index(inplace=True)
  exterior_predictions_df.drop(index=exterior_predictions_df.q_py("img == ''").index, inplace=True)   # drop img with no names
  exterior_predictions_df.defrag_index(inplace=True)

  # merge the 2 set of predictions
  predictions_df = join_df(predictions_df, exterior_predictions_df, left_on='img', how='inner')

  # remove irrelevant exterior predictions from non outdoor 
  idx = predictions_df.q_py("inoutdoor.isin(['indoor', 'other'])").index
  predictions_df.loc[idx, 'p_facade'] = np.NaN
  predictions_df.loc[idx, 'p_backyard'] = np.NaN
  predictions_df.loc[idx, 'p_view'] = np.NaN
  predictions_df.loc[idx, 'p_exterior'] = np.NaN

  predictions_df['listingId'] = predictions_df.img.apply(get_listingId_from_image_name)

  # extract/save individual images
  overwritten_listingIds = []
  if image_destination_dir is not None:
    image_destination_dir = Path(image_destination_dir)

    # for listing whose images may have been updated in the latest tagging, 
    # just remove and repopulate
    for listingId in set([get_listingId_from_image_name(f) for f in filenames]):
      if (image_destination_dir/listingId).exists():
        shutil.rmtree(image_destination_dir/listingId)
        overwritten_listingIds.append(listingId)

    for img, fname, r in img_ds:
      filename = fname.numpy().decode('utf-8')
      if filename != '':
        listingId = get_listingId_from_image_name(filename)
        
        r = tf.cast(float(r), tf.float32)

        shape = img.shape
        height = shape[0]
        width = tf.cast(r * shape[1], tf.int32)

        img = tf.image.resize(img, size=(height, width), method=tf.image.ResizeMethod.BICUBIC, antialias=True)

        (image_destination_dir/listingId).mkdir(exist_ok=True)

        tf.io.write_file(str(image_destination_dir/listingId/filename), tf.io.encode_jpeg(tf.cast(img, tf.uint8), quality=100))

  return predictions_df, filenames, overwritten_listingIds


def download_and_process_new_avm_snapshot(snapshot_date=datetime(2022, 7, 27), download_from_gs=True) -> pd.DataFrame:
  avm_snapshot_listing_df = load_avm_prod_snapshot(snapshot_date=snapshot_date, download_from_gs=download_from_gs)
  avm_snapshot_listing_df['photo_uris'] = avm_snapshot_listing_df.photos.apply(lambda x: [re.sub(r'//rlp.jumplisting.com/photos', '', i['fileName']) for i in  x['photos']])
  avm_snapshot_listing_df.quickQuote = avm_snapshot_listing_df.quickQuote.apply(fillna_with_empty_dict)
  avm_snapshot_listing_df = filter_for_listingType_SALE(avm_snapshot_listing_df)

  listing_quickquote_df = flatten_dict_col(avm_snapshot_listing_df, cols=['quickQuote'], return_only_flattened=True)[0]
  proc_quality(listing_quickquote_df)

  listing_quickquote_df.exactMatch = listing_quickquote_df.exactMatch.apply(fillna_with_empty_dict)
  listing_quickquote_df.lmaMatch = listing_quickquote_df.lmaMatch.apply(fillna_with_empty_dict)
  listing_quickquote_df.liveScoring = listing_quickquote_df.liveScoring.apply(fillna_with_empty_dict)

  listing_quickquote_exactmatch_df = flatten_dict_col(listing_quickquote_df, cols=['exactMatch'], return_only_flattened=True, use_prefix=True)[0]
  listing_quickquote_lmamatch_df = flatten_dict_col(listing_quickquote_df, cols=['lmaMatch'], return_only_flattened=True, use_prefix=True)[0]
  listing_quickquote_livescoring_df = flatten_dict_col(listing_quickquote_df, cols=['liveScoring'], return_only_flattened=True, use_prefix=True)[0]

  # clean listing_quickquote_livescoring_df
  cleanup_livescoring(listing_quickquote_livescoring_df)

  # join with avm_snapshot_listing_df
  columns = list(avm_snapshot_listing_df.columns) + ['quality'] + list(listing_quickquote_exactmatch_df.columns) + list(listing_quickquote_lmamatch_df.columns) + list(listing_quickquote_livescoring_df.columns)

  avm_snapshot_listing_df = pd.concat([avm_snapshot_listing_df, 
                          listing_quickquote_df[['quality']], 
                          listing_quickquote_exactmatch_df,
                          listing_quickquote_lmamatch_df,
                          listing_quickquote_livescoring_df,
                        ], axis=1, ignore_index=True)

  avm_snapshot_listing_df.columns = columns

  avm_snapshot_listing_df.parsedAddress = avm_snapshot_listing_df.parsedAddress.apply(fillna_with_empty_dict)
  parsedAddress_df = flatten_dict_col(avm_snapshot_listing_df, cols=['parsedAddress'], return_only_flattened=True, use_prefix=True)[0]

  columns = list(avm_snapshot_listing_df.columns) + list(parsedAddress_df.columns)
  avm_snapshot_listing_df = pd.concat([avm_snapshot_listing_df, parsedAddress_df], axis=1, ignore_index=True)
  avm_snapshot_listing_df.columns = columns

  del listing_quickquote_df, listing_quickquote_exactmatch_df, listing_quickquote_lmamatch_df, listing_quickquote_livescoring_df, parsedAddress_df
  gc.collect();

  avm_snapshot_listing_df.drop(columns=['parsedAddress', 'quickQuote'], inplace=True)

  avm_snapshot_listing_df[['presented', 'presented_qq_lower', 'presented_qq_upper']] = avm_snapshot_listing_df.apply(get_presented_best_estimate, axis=1)

  avm_snapshot_listing_df.drop(index=avm_snapshot_listing_df.q_py("lastUpdate <= '2021-12-01'").index, inplace=True)
  fix_ON_provState(avm_snapshot_listing_df)

  price_threshold = 50000
  USE_HIGH_PRICE_THRESHOLD = False
  high_price_threshold = 1e7

  avm_snapshot_listing_df = price_addr_filter(avm_snapshot_listing_df, price_threshold, use_high_price_threshold=USE_HIGH_PRICE_THRESHOLD, high_price_threshold=high_price_threshold)

  avm_snapshot_listing_df.defrag_index(inplace=True)

  return avm_snapshot_listing_df
  

def gen_samples_for_downloading(avm_snapshot_listing_df, image_dir: Path, n_sample=1000) -> pd.DataFrame:
  '''
  The output df should be uploaded to jumptools VM to download the images.
  '''
  # excl. jumpId already in image_dir
  excl_jumpIds = [f.name for f in image_dir.ls()]
  print(f'len(excl_jumpIds): {len(excl_jumpIds)}')

  # avm_snapshot_listing_df.drop(index=avm_snapshot_listing_df.q_py("jumpId.isin(@excl_jumpIds)").index, inplace=True)
  # avm_snapshot_listing_df.defrag_index(inplace=True)
  # sample_avm_snapshot_listing_df = avm_snapshot_listing_df.sample(n=n_sample, random_state=42)

  sample_avm_snapshot_listing_df = avm_snapshot_listing_df.q_py("~jumpId.isin(@excl_jumpIds)").sample(n=n_sample, random_state=42)[['jumpId', 'photo_uris']].copy()
  sample_avm_snapshot_listing_df.defrag_index(inplace=True)

  assert len(set(sample_avm_snapshot_listing_df.jumpId.values).intersection(set(excl_jumpIds))) == 0, 'sample_avm_snapshot_listing_df and excl_jumpIds overlap'

  return sample_avm_snapshot_listing_df


def return_clean_sample_images(test_height=224, test_width=224, image_dir: Path = None) -> List:
  '''
  return list of image paths that are clean, and use it for next step in the pipeline
  '''
  if image_dir is None and bOnColab:
    image_dir = Path('/content/photos')

  assert image_dir is not None, 'image_dir is None, this must be provided.'
  
  imgs = [str(f) for f in image_dir.rlf('*.jpg')]

  removed_imgs = []
  for f in imgs:
    fobj = open(f, 'rb')
    if tf.compat.as_bytes('JFIF') not in fobj.peek(10):
      print(f'{f} is corrupted, removed')
      os.remove(f)
      removed_imgs.append(f)

  for f in removed_imgs:
    imgs.remove(f)

  def integrity_check(filename):
    image_string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, (test_height, test_width), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
    
    return filename

  # ds = tf.data.Dataset.from_tensor_slices(imgs)
  # for filename in tqdm(ds.map(integrity_check, num_parallel_calls=tf.data.AUTOTUNE)):
  #   continue
  removed_imgs = []
    
  for filename in imgs:
    try:
      _ = integrity_check(filename)
    except:
      print(f'{filename} failed integrity check, removed')
      os.remove(filename)
      removed_imgs.append(filename)

  for f in removed_imgs:
    imgs.remove(f)
  
  return imgs

def create_tfrecord_from_image_list(imgs: List, out_tfrecord_filepath: str = 'image.tfrecords', height=416, width=416):
  '''
  imgs: list of image paths
  out_tfrecord_filepath: full path name of output tfrecord file

  return nothing. File written to
  '''

  out_tfrecord_filepath = str(out_tfrecord_filepath)

  file_ds = tf.data.Dataset.from_tensor_slices(
      {
        'filename': [Path(f).name for f in imgs],
        'filepath': imgs
      }
  )

  def proc_img(x):   # mainly just resizing
    img_str = tf.io.read_file(x['filepath'])
    image = tf.image.decode_jpeg(img_str, channels=3)
    image = tf.image.resize(image, (height, width), method=tf.image.ResizeMethod.BICUBIC, antialias=True)
    image = tf.cast(image, tf.uint8)
    resized_img_str = tf.io.encode_jpeg(image, quality=100)
    return resized_img_str

  data_ds = file_ds.map(lambda x: {'filename': x['filename'], 'image_raw': proc_img(x)}, num_parallel_calls=tf.data.AUTOTUNE)

  features = {
    'filename': TFRecordHelper.DataType.STRING,
    'image_raw': TFRecordHelper.DataType.STRING,   # bytes for the encoded jpeg, png, etc.
  }

  parse_fn = TFRecordHelper.parse_fn(features)

  with TFRecordHelperWriter(out_tfrecord_filepath, features = features) as f:
    f.write(data_ds)

def gen_predictions_for(tfrecord_filepath: str, imgs: List) -> pd.DataFrame:
  features = {
    'filename': TFRecordHelper.DataType.STRING,
    'image_raw': TFRecordHelper.DataType.STRING,   # bytes for the encoded jpeg, png, etc.
  }

  parse_fn = TFRecordHelper.parse_fn(features)

  # predictions for in/outdoor and room type
  img_ds = tf.data.TFRecordDataset(tfrecord_filepath).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE).map(lambda x: tf.image.decode_jpeg(x['image_raw'], channels=3), num_parallel_calls=tf.data.AUTOTUNE)

  batch_size = 32

  batch_img_ds = img_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

  model = load_model(home/'ListingImageClassification'/'training'/'hydra_all'/'resnet50_hydra_all.acc.0.9322.h5', compile=False)

  yhats = model.predict(batch_img_ds)

  iodoor_yhats = yhats[0]
  room_yhats = yhats[1]
  '''
  fireplace_yhats = yhats[2]
  agpool_yhats = yhats[3]
  body_of_water_yhats = yhats[4]
  igpool_yhats = yhats[5]
  balcony_yhats = yhats[6]
  deck_patio_veranda_yhats = yhats[7]
  ss_kitchen_yhats = yhats[8]
  double_sink_yhats = yhats[9]
  upg_kitchen_yhats = yhats[10]
  '''
  
  inoutdoor = [INOUT_DOOR_CLASS_NAMES[int(y)] for y in np.squeeze(np.argmax(iodoor_yhats, axis=-1))]

  predictions_df = pd.DataFrame(data={
    'img': [Path(img).name for img in imgs],
    'inoutdoor': inoutdoor,
    'p_iodoor': np.round(np.max(iodoor_yhats, axis=-1).astype('float'), 4),

    'room': [ROOM_CLASS_NAMES[int(y)] for y in np.squeeze(np.argmax(room_yhats, axis=-1))],
    'p_room': np.round(np.max(room_yhats, axis=-1).astype('float'), 4),
  })

  # predictions for exterior types, use ensemble of models
  exteriors_model_files = [
                           'resnet50_distill_exteriors.acc.0.9106.h5',
                           'resnet50_distill_exteriors.acc.0.9114.h5',
                           'resnet50_distill_exteriors.acc.0.9131.h5',
                           'resnet50_distill_exteriors.acc.0.9116.h5',
                           'resnet50_distill_exteriors.acc.0.9072.h5'
                           ]

  exteriors_labels = ['facade', 'backyard', 'view', 'exterior']

  img_height, img_width = 224, 224    # exterior classification model take in 224x224 images

  def read_decode_resize_jpg(x):
    # image_string = tf.io.read_file(filename)
    image_string = x['image_raw']
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, (img_height, img_width), method=tf.image.ResizeMethod.BILINEAR)
    
    return image

  ds = tf.data.TFRecordDataset(tfrecord_filepath).map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
  img_ds = ds.map(read_decode_resize_jpg, num_parallel_calls=tf.data.AUTOTUNE)
  batch_img_ds = img_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

  y_preds = []
  for m in exteriors_model_files:
    print(m)
    model = load_model(home/'ListingImageClassification'/'training'/'exteriors'/m)
    input_img_height, input_img_width = model.input.shape[1:3]

    assert input_img_height == img_height and input_img_width == img_width, 'model input shape is not 224x224'
    
    y_pred = model.predict(batch_img_ds)
    y_pred = tf.nn.sigmoid(y_pred).numpy()     # distill model return logits
    y_preds.append(y_pred)

  y_preds = np.stack(y_preds)
  y_pred = np.mean(y_preds, axis=0)
  
  exterior_predictions_df = pd.DataFrame(data={
        'img': [Path(img).name for img in imgs],
        'p_facade': np.round(y_pred[:, 1].astype('float'), 4),    # first element was an indicator for hard vs. soft labels during training, should ignore during inference.
        'p_backyard': np.round(y_pred[:, 2].astype('float'), 4),
        'p_view': np.round(y_pred[:, 3].astype('float'), 4),
        'p_exterior': np.round(y_pred[:, 4].astype('float'), 4)
      })


  predictions_df = join_df(predictions_df, exterior_predictions_df, left_on='img', how='inner')

  # remove irrelevant exterior predictions from non outdoor 
  idx = predictions_df.q_py("inoutdoor.isin(['indoor', 'other'])").index
  predictions_df.loc[idx, 'p_facade'] = np.NaN
  predictions_df.loc[idx, 'p_backyard'] = np.NaN
  predictions_df.loc[idx, 'p_view'] = np.NaN
  predictions_df.loc[idx, 'p_exterior'] = np.NaN

  predictions_df['listingId'] = predictions_df.img.apply(get_listingId_from_image_name)

  return predictions_df




def unstack(filenames, bigImg, orig_aspect_ratios):    # unstack grid of NxN images
  batch_size = tf.shape(filenames)[0]

  img = tf.transpose(tf.reshape(tf.transpose(tf.reshape(bigImg, (SQRT_BATCH_SIZE, 416, 416*SQRT_BATCH_SIZE, 3)), (0, 2, 1, 3)), (-1, 416, 416, 3)), (0, 2, 1, 3))

  if batch_size != BATCH_SIZE:   # pad filenames, orig_aspect_ratios
    pad_size = BATCH_SIZE - batch_size
    out_filenames =  tf.concat([tf.sparse.to_dense(filenames), tf.zeros((pad_size,), dtype=tf.string)], axis=0)
    out_orig_aspect_ratios = tf.concat([tf.sparse.to_dense(orig_aspect_ratios), -1*tf.ones((pad_size,), dtype=tf.float32)], axis=0)
  else:
    out_filenames = tf.sparse.to_dense(filenames)
    out_orig_aspect_ratios = tf.sparse.to_dense(orig_aspect_ratios)

  return out_filenames, img, out_orig_aspect_ratios