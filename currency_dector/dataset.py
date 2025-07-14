import kagglehub



# Download latest version
path = kagglehub.dataset_download("gauravsahani/indian-currency-notes-classifier")

print("Path to dataset files:", path)