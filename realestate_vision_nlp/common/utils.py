from realestate_vision.common.utils import get_listingId_from_image_name, get_listing_folder_from_image_name
import realestate_vision

def get_image_full_path(img_name: str, src_image_dir: str) -> str:
  '''
  Get full path of image
  '''
  listingId = get_listingId_from_image_name(img_name)
  folder = get_listing_folder_from_image_name(img_name)

  if (src_image_dir/listingId/img_name).exists():
    return src_image_dir/listingId/img_name
  elif (src_image_dir/folder/img_name).exists():
    return src_image_dir/folder/img_name
  else:
    print(f'Image not found: {img_name}')
    return None

def get_listing_subfolder_from_listing_id(listing_id: str, src_image_dir: str) -> str:
  '''
  Get listing subfolder from listing_id
  '''
  hier_subfolder = realestate_vision.common.utils.get_listing_subfolder_from_listing_id(listing_id)

  if (src_image_dir/listing_id).exists():
    return src_image_dir/listing_id
  elif (src_image_dir/hier_subfolder).exists():
    return src_image_dir/hier_subfolder
  else:
    print(f'listing folder not found: {listing_id}')
    return None
  