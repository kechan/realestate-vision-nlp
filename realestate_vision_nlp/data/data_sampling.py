from typing import List, Set, Dict, Tuple, Any, Optional, Iterator, Union
from pathlib import Path

import pandas as pd

from realestate_nlp.common.run_config import home, bOnColab

neg_terms_in_remark = ['mini home', 'as-is', 
                       #'where is',    # do not use this for now, need to double check context
                       'affordable starter home',
                       'fully rented', 'elbow grease', 'handyman special', 'build equity', 'sweat equity',
                       'down to the studs', 'renovation project'
                       ]

pos_terms_in_remark = ['craftsmanship', 'finest material', 'granite counter', 'marble counter', 'stone counter', 
                       'quartz counter', 'gourmet kitchen', 'chef kitchen', 'custom kitchen', 'hardwood floor', 'engineered hardwood',
                       'attention to detail', 'wine cellar', 'wine Room', 'wine fridge', 'high end finishes', 
                       'fully renovated', 's/s appliances', 'stainless steel appliances',
                       'boutique building', 'boutique loft', 'boutique condo',
                       'superior amenities', 'custom window coverings', 'luxury condo',
                       'custom build', 'coffered ceiling', 'ensuite', 'open concept', 
                       'luxury home', 'private elevator', 'custom elevator'
                       ]
                       
def sample_listings_with_neg_remarks(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
  '''
  Sample listings with negative remarks

  Input: a dataframe with remarks

  Return: dataframe of sampled listings, list of sampled listingIds
  '''

  assert 'remarks' in df.columns, 'remarks column not found'

  or_neg_terms = '|'.join(neg_terms_in_remark)
  neg_kw_df = df.q_py("remarks.notnull() and remarks.str.contains(@or_neg_terms, case=False)").copy()

  neg_kw_jumpIds = list(neg_kw_df.jumpId.values)
  
  return neg_kw_df, neg_kw_jumpIds

def sample_listings_with_pos_remarks(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
  '''
  Sample listings with positive remarks

  Input: a dataframe with remarks

  Return: dataframe of sampled listings, list of sampled listingIds
  '''

  assert 'remarks' in df.columns, 'remarks column not found'

  or_pos_terms = '|'.join(pos_terms_in_remark)
  pos_kw_df = df.q_py("remarks.notnull() and remarks.str.contains(@or_pos_terms, case=False)==False").copy()

  pos_kw_jumpIds = list(pos_kw_df.jumpId.values)
  
  return pos_kw_df, pos_kw_jumpIds


def sample_listings_with_price_less_than_qq(df: pd.DataFrame, factor: float) -> Tuple[pd.DataFrame, List]:
  '''
  Sample listings with price less than 'factor' of lower bound of quick quote range.

  Input: a dataframe with price and presented_qq_lower (lower bound of quick quote range)

  Return: dataframe of sampled listings, list of sampled listingIds
  '''

  assert 'price' in df.columns, 'price column not found'
  assert 'presented_qq_lower' in df.columns, 'presented_qq_lower column not found'

  lower_than_qq_df = df.q_py(f"price < presented_qq_lower*{factor}").copy()

  lower_than_qq_jumpIds = list(lower_than_qq_df.jumpId.values)
  
  return lower_than_qq_df, lower_than_qq_jumpIds

def sample_carriage_trade(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
  '''
  Sample carriage trade listings

  Input: a dataframe with remarks

  Return: dataframe of sampled listings, list of sampled listingIds
  '''

  assert 'remarks' in df.columns, 'remarks column not found'

  carriage_trade_df = df.q_py("carriageTrade").copy()
  carriage_trade_df.defrag_index(inplace=True)
  
  carriage_trade_jumpIds = list(carriage_trade_df.jumpId.values)
  
  return carriage_trade_df, carriage_trade_jumpIds

def sample_kitchen_features(df: pd.DataFrame, p_threshold: float = 0.9) -> Tuple[pd.DataFrame]:
  '''
  Sample listings with kitchen features with probability greater than p_threshold

  Input: a dataframe with predictions from all_hydra

  Return: dataframe of sampled listings, list of sampled listingIds
  '''

  assert 'room' in df.columns, 'room column not found'
  assert 'p_ss_kitchen' in df.columns, 'p_ss_kitchen column not found'
  assert 'p_double_sink' in df.columns, 'p_double_sink column not found'
  assert 'p_upg_kitchen' in df.columns, 'p_upg_kitchen column not found'

  good_quality_kitchen_df = df.q_py("room == 'kitchen' and (p_ss_kitchen > 0.9 and p_double_sink > 0.9 and p_upg_kitchen > 0.9)").copy()
  good_quality_kitchen_df.defrag_index(inplace=True)

  return good_quality_kitchen_df

def sample_listings_with_price_more_than_qq(df: pd.DataFrame) -> Tuple[pd.DataFrame, List]:
  '''
  Sample listings with price more than upper bound of quick quote range.

  Input: a dataframe with price and presented_qq_upper (upper bound of quick quote range)

  Return: dataframe of sampled listings, list of sampled listingIds
  '''

  assert 'price' in df.columns, 'price column not found'
  assert 'presented_qq_upper' in df.columns, 'presented_qq_upper column not found'

  higher_than_qq_df = df.q_py(f"price > presented_qq_upper").copy()

  higher_than_qq_jumpIds = list(higher_than_qq_df.jumpId.values)
  
  return higher_than_qq_df, higher_than_qq_jumpIds

# def 