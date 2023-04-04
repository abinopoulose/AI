from bing_image_downloader import downloader 
downloader.download("wrinkle face", limit=1000,  output_dir='dataset', adult_filter_off=True, force_replace=False, timeout=600000, verbose=True)