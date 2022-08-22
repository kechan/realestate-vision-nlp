import shutil
from time import sleep
from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
from pathlib import Path
import pandas as pd
try:
  from doccano_api_client import DoccanoClient
except:
  print('doccano_api_client not installed. Doccano related functions will not work.')

from realestate_nlp.common.run_config import home, bOnColab
from realestate_vision.common.utils import load_from_pickle, save_to_pickle, get_listingId_from_image_name, get_listing_folder_from_image_name

from .data_processing import apply_avm_processing_to_listing_df

def load_listing_json_and_prediction(data_dir) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, List]:
  '''
  Return listing es info from listing_es_df (image tagging) and avm_snapshot_listing_df (AVM Monitoring)
  Return predicted features for listing images (predictions_df). 
  Return jumpIds that have high res images
  '''

  predictions_df = pd.read_feather(data_dir/'predictions_df')
  print('Loaded predictions_df')
  print(f'total #: {predictions_df.shape[0]}, # of unique listing: {predictions_df.listingId.nunique()}')

  listing_es_df = pd.read_pickle(data_dir/'673_listing_es_df.pickle.gz', compression='gzip') 
  listing_es_df.listingType = listing_es_df.listingType.apply(lambda a: a[-1])  # e.g. [listingType, RES] => 'RES'
  listing_es_df.drop_duplicates(subset='jumpId', keep='last', inplace=True)
  listing_es_df.defrag_index(inplace=True)
  print('Loaded listing_es_df')
  print(f'total #: {listing_es_df.shape[0]}, # of unique listing: {listing_es_df.jumpId.nunique()}')

  avm_snapshot_listing_df = pd.read_feather(data_dir/'avm_snapshot_listing_df')
  print('Loaded avm_snapshot_listing_df')
  print(f'total #: {avm_snapshot_listing_df.shape[0]}, # of unique listing: {avm_snapshot_listing_df.jumpId.nunique()}')

  jumpIds_w_high_res_images = load_from_pickle(data_dir/'jumpIds_w_high_res_images.pkl')
  print(f'{len(jumpIds_w_high_res_images)} listings with high res images')

  # keep only listings that have predictions 

  listingIds = list(predictions_df.listingId.unique())
  listing_es_df.drop(index=listing_es_df.q_py("~jumpId.isin(@listingIds)").index, inplace=True)
  listing_es_df.defrag_index(inplace=True)

  avm_snapshot_listing_df.drop(index=avm_snapshot_listing_df.q_py("~jumpId.isin(@listingIds)").index, inplace=True)
  avm_snapshot_listing_df.defrag_index(inplace=True)

  # drop some unncessary cols from avm_snapshot_listing_df
  avm_snapshot_listing_df.drop(columns=['PHOTO_HOST', 'avm', 'photo_uris', 'pt', 'score', 't'], inplace=True)

  # apply AVM processing, quick quote stuff.
  listing_es_df = apply_avm_processing_to_listing_df(listing_es_df)

  return listing_es_df, avm_snapshot_listing_df, predictions_df, jumpIds_w_high_res_images

def copy_images_to_dir(df: pd.DataFrame, src_image_dir: str, dest_image_dir: str = None) -> None:
  '''
  Copy images in df.img from src_image_dir to dest_image_dir

  df: dataframe (with a column named 'img')
  src_image_dir: root directory where the source images are located
  dest_image_dir: root directory where images will be downloaded
  '''

  src_image_dir = Path(src_image_dir)

  for k, row in df.iterrows():
    listingId = get_listingId_from_image_name(row.img)
    folder = get_listing_folder_from_image_name(row.img)

    if (src_image_dir/listingId/row.img).exists():
      shutil.copy(src_image_dir/listingId/row.img, dest_image_dir)
    elif (src_image_dir/folder/row.img).exists():
      shutil.copy(src_image_dir/folder/row.img, dest_image_dir)
    else:
      print(f'Image not found: {row.img}')

def upload_images_to_doccano_project(
  doccano_host_port: str ='127.0.0.1:8001', 
  doccano_login='admin', 
  doccano_passwd='password', 
  image_dir: str = '.', 
  project_name: str = None, 
  ) -> None:
  '''
  Upload images to doccano project

  image_dir: root directory where images are located
  project_name: name of project on doccano
  '''

  client = DoccanoClient(f'http://{doccano_host_port}', doccano_login, doccano_passwd)

  projects = client.get('/v1/projects', params={'limit':1000, 'offset':0})['results']
  project_name_to_id = {p['name']: p['id'] for p in projects if 'fixme' not in p['name']}
  project_id_to_name = {p['id']: p['name'] for p in projects if 'fixme' not in p['name']}

  if project_name is None:
    print('project_name is None, exiting')
    return

  project_id = project_name_to_id[project_name]

  image_dir = Path(image_dir)

  for f in image_dir.rlf('*.jpg'):
    img_fp = open(f, 'rb')
    resp = client.post_doc_upload_binary(project_id=project_id, files=[img_fp], format='ImageFile')
    img_fp.close()

    sleep(0.1)

def get_all_listing_df(keep_cols=['jumpId', 'remarks'], dedup=False):
  # on My Mac Backup
  data_dir = Path('/Volumes/My Mac Backup/RLP/ListingImageClassification/data/bigstack_rlp_listing_images_tfrecords')
  assert data_dir.exists(), f'{data_dir} does not exist'

  print("Loading from image taggings system")
  dfs = []
  for f in data_dir.lf('*_df.pickle.gz'):
    df = pd.read_pickle(f, compression='gzip')
    dfs.append(df)

  df = pd.concat(dfs, axis=0, ignore_index=True)
  df.defrag_index(inplace=True)

  # from NLP footage project
  print("Loading from NLP footage project")
  listing_es_df = pd.read_feather(home/'NLP'/'data'/'listing_es_df_pickles'/'listing_es_df')   # no photos

  print("Loading from recent imaging tagging, listings with AVM quick quote")
  listing_es_w_qq_df = pd.read_pickle(home/'ConditionSentiment'/'data'/'673_listing_es_df.pickle.gz', compression='gzip')   # with photos
  listing_es_w_qq_df.listingType = listing_es_w_qq_df.listingType.apply(lambda a: a[-1])  # e.g. [listingType, RES] => 'RES'
  listing_es_w_qq_df.drop_duplicates(subset='jumpId', keep='last', inplace=True)
  listing_es_w_qq_df.defrag_index(inplace=True)
  print("\tApplying AVM processing to listing_es_w_qq_df")
  listing_es_w_qq_df = apply_avm_processing_to_listing_df(listing_es_w_qq_df)

  print("Loading from AVM snapshots")
  avm_snapshot_listing_df = pd.read_feather(home/'ConditionSentiment'/'data'/'avm_snapshot_listing_df')

  # Combine all listing sources
  df = pd.concat([df, listing_es_df, listing_es_w_qq_df, avm_snapshot_listing_df], axis=0, ignore_index=True)

  def to_datetime(x):
    try:
      return pd.to_datetime(x, format="%y-%m-%d:%H:%M:%S")
    except:
      return pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S")

  df.lastUpdate = df.lastUpdate.apply(to_datetime)

  # dedup and keep record with latest lastUpdate
  if dedup:
    df.sort_values('lastUpdate', ascending=False, inplace=True)
    df.drop_duplicates(subset='jumpId', keep='first', inplace=True)
    df.defrag_index(inplace=True)

   # Load older listings
   # TODO: Re-org the dataset 
  data_listings_listing_df_csv_df = pd.read_csv(home/'ListingImageClassification'/'data'/'listings'/'listing_df.csv', dtype={'listingId': str})
  data_listings_listing_df_csv_df.rename(columns={'listingId': 'jumpId'}, inplace=True)

  data_listings_full_listing_df_csv_df = pd.read_csv(home/'ListingImageClassification'/'data'/'listings'/'full_listing_df.csv', dtype={'listingId': str})
  data_listings_full_listing_df_csv_df.rename(columns={'listingId': 'jumpId'}, inplace=True)

  # skip these for now, it has no "remarks".
  '''
  listing_df = pd.read_feather(home/'ListingImageClassification'/'data'/'listing_df')
  listing_df.rename(columns={'listingId': 'jumpId'}, inplace=True)

  # from BiqQuery for recommendations engine development, but they have no remarks
  bq_listing_dfs = []
  for f in (home/'ListingRecommendation'/'data'/'bq_listings').lf('bq_master_listing_*_df'):
    bq_listing_df = pd.read_feather(f)
    bq_listing_dfs.append(bq_listing_df)
  bq_listing_df = pd.concat(bq_listing_dfs, axis=0, ignore_index=True)
  bq_listing_df.rename(columns={'listingId': 'jumpId'}, inplace=True)
  # '''

  df = pd.concat([data_listings_listing_df_csv_df, 
                  data_listings_full_listing_df_csv_df, 
                  df], axis=0, ignore_index=True)
  df.drop_duplicates(subset='jumpId', keep='last', inplace=True)
  df.defrag_index(inplace=True)

  df = df[keep_cols]
  return df

    
