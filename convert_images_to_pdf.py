import os
import img2pdf
from pathlib import Path
from PIL import Image

def convert_images_to_pdf(data_dir):
    print(f"Scanning {data_dir} for images to convert to PDF...")
    
    image_extensions = {'.png', '.jpg', '.jpeg'}
    image_files = []
    for root, dirs, files in os.walk(data_dir):
        # Skip directories already processed or output directories
        if any(x in root for x in ['train', 'test', 'duplicates']):
            continue
            
        for file in files:
            ext = Path(file).suffix.lower()
            if ext in image_extensions:
                image_files.append(Path(root) / file)
    
    total = len(image_files)
    print(f"Found {total} image files to convert.")
    
    converted_count = 0
    error_count = 0
    
    for i, img_path in enumerate(image_files, 1):
        pdf_path = img_path.with_suffix('.pdf')
        print(f"[{i}/{total}] Converting {img_path.name} to {pdf_path.name}...")
        
        try:
            # Open image to verify/auto-rotate if needed (though img2pdf handles most)
            # and write to PDF
            with open(pdf_path, "wb") as f:
                f.write(img2pdf.convert(str(img_path)))
            
            # Remove original image
            os.remove(img_path)
            converted_count += 1
            print(f"  ✅ Success.")
        except Exception as e:
            print(f"  ❌ Error: {e}")
            error_count += 1
            
    print("\n" + "="*60)
    print(f"CONVERSION COMPLETE")
    print("="*60)
    print(f"Successfully converted: {converted_count}")
    print(f"Errors: {error_count}")
    print("="*60)

if __name__ == "__main__":
    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"
    convert_images_to_pdf(data_dir)
