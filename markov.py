import kagglehub

# Download latest version
path = kagglehub.dataset_download("nisargchodavadiya/daily-gold-price-20152021-time-series")

print("Path to dataset files:", path)