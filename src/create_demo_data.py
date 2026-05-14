import cv2
import numpy as np
import os

def create_synthetic_pages(output_dir, num_pages=10):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Creating {num_pages} synthetic pages in {output_dir}...")
    
    # Standard Mushaf page size (approximate aspect ratio)
    width, height = 800, 1200
    
    for i in range(1, num_pages + 1):
        # Create a white page
        page = np.full((height, width, 3), 255, dtype=np.uint8)
        
        # Add some "text" lines (rectangles)
        num_lines = 15
        for line in range(num_lines):
            y_start = 100 + line * 60
            y_end = y_start + 30
            # Randomly vary line length and position to simulate different pages
            x_start = np.random.randint(50, 150)
            x_end = np.random.randint(width - 150, width - 50)
            
            # Use different patterns for different pages to make them distinguishable
            color = (0, 0, 0)
            cv2.rectangle(page, (x_start, y_start), (x_end, y_end), color, -1)
            
            # Add some unique features for each page
            cv2.putText(page, f"PAGE {i}", (width//2 - 50, height - 50), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
            
        filename = f"page_{i:03d}.png"
        cv2.imwrite(os.path.join(output_dir, filename), page)

if __name__ == "__main__":
    create_synthetic_pages("wasil_project/dataset/pages")
