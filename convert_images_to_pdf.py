import os
import img2pdf
from pathlib import Path

def convert_png_to_pdf(data_dir):
    print(f"Scanning {data_dir} for .png files...")
    
    png_files = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.lower().endswith('.png'):
                png_files.append(Path(root) / file)
    
    total = len(png_files)
    print(f"Found {total} .png files to convert.")
    
    converted_count = 0
    error_count = 0
    
    for i, png_path in enumerate(png_files, 1):
        pdf_path = png_path.with_suffix('.pdf')
        print(f"[{i}/{total}] Converting {png_path.name}...")
        
        try:
            # Convert PNG to PDF bytes
            with open(pdf_path, "wb") as f:
                f.write(img2pdf.convert(str(png_path)))
            
            # If successful, remove the original PNG
            os.remove(png_path)
            converted_count += 1
            print(f"  ✅ Converted and removed original.")
        except Exception as e:
            print(f"  ❌ Error converting {png_path.name}: {e}")
            error_count += 1
            
    print("\n" + "="*60)
    print(f"CONVERSION COMPLETE")
    print("="*60)
    print(f"Successfully converted: {converted_count}")
    print(f"Errors: {error_count}")
    print("="*60)

if __name__ == "__main__":
    project_root = Path(__file__).parent
    data_dir = project_root / "data"
    convert_png_to_pdf(data_dir)
