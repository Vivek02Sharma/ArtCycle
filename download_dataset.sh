# Define the destination directory
destination_dir="./datasets"

# Create the directory if it does not exist
mkdir -p "$destination_dir"

# Download the dataset using Kaggle API
kaggle datasets download -d balraj98/monet2photo -p "$destination_dir"

# Unzip the dataset in the destination folder
unzip "$destination_dir/monet2photo.zip" -d "$destination_dir"

# Optional: Remove the zip file after unzipping to save space
rm "$destination_dir/monet2photo.zip"

echo "Dataset downloaded and unzipped successfully!"
