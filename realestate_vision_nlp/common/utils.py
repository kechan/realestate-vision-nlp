from realestate_vision.common.utils import get_listingId_from_image_name, get_listing_folder_from_image_name


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