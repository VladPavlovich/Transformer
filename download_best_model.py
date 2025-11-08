import gdown

def download_from_google_drive(file_id, output_path):
    """
    Download a file from Google Drive using its file ID.
    
    Args:
        file_id (str): The Google Drive file ID (from the shareable link)
        output_path (str): The local path where the file should be saved
    
    Returns:
        str: Path to the downloaded file
    """
    gdown.download(id=file_id, output=output_path, quiet=False)
    return output_path


if __name__ == "__main__":
    # Example usage:
    # Replace 'YOUR_FILE_ID' with your actual Google Drive file ID
    # The file ID is the part after '/d/' and before '/view' in a Google Drive shareable link
    # Example link: https://drive.google.com/file/d/1SYtUvGhqFYce0cr23V3TdBDmM63X7z3-/view?usp=sharing
    # File ID would be: 1SYtUvGhqFYce0cr23V3TdBDmM63X7z3-
    
    file_id = "YOUR_FILE_ID"  # Replace with your file ID
    output_path = "best_model.pt"
    
    download_from_google_drive(file_id, output_path)
    print(f"Downloaded to: {output_path}")
