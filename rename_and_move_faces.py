import os

# Path to the directory containing the images to be renamed
directory_path = r'C:\Users\ADMIN\Downloads\4tos me'

# List all files in the directory
files = os.listdir(directory_path)

# Initialize a counter for the new filenames
counter = 1

# Loop through all files and rename them orderly
for file_name in files:
    # Construct full file path
    source_file_path = os.path.join(directory_path, file_name)
    
    # Ensure it's a file (not a directory)
    if os.path.isfile(source_file_path):
        # New filename in the format "image_1.jpg", "image_2.jpg", etc.
        new_file_name = f"image_{counter}.jpg"
        
        # Construct the new file path
        destination_file_path = os.path.join(directory_path, new_file_name)
        
        # Rename the file
        os.rename(source_file_path, destination_file_path)
        
        # Increment the counter
        counter += 1

print("Renaming completed.")
