import json
import uuid
import numpy as np
from PIL import Image
from pathlib import Path
from playwright.sync_api import sync_playwright

from src.configs import rendering_config as cfg
from .html_styler import generate_html_with_style
from .noise_injector import apply_noise


class TableRenderer:
    def __init__(self, source_table_id: str, lang_code: str, table_data: dict):
        self.source_table_id = source_table_id
        self.lang_code = lang_code
        self.table_data = table_data

    def _extract_bounding_boxes_with_playwright(self, page):
        """Extract accurate bounding boxes using Playwright."""
        bboxes = {
            'header': [None] * len(self.table_data.get('columns', [])),
            'cells': [
                [None] * len(self.table_data.get('columns', []))
                for _ in self.table_data.get('data', [])
            ],
        }

        # Extract header cells
        header_cells = page.query_selector_all('[id^="cell-h-"]')
        for cell in header_cells:
            cell_id = cell.get_attribute('id')
            idx = int(cell_id.split('-')[-1])
            box = cell.bounding_box()
            if box:
                bboxes['header'][idx] = [
                    box['x'],
                    box['y'],
                    box['x'] + box['width'],
                    box['y'] + box['height']
                ]

        # Extract data cells
        data_cells = page.query_selector_all('[id^="cell-r"]')
        for cell in data_cells:
            cell_id = cell.get_attribute('id')
            parts = cell_id.replace('cell-r', '').split('-c')
            r, c = int(parts[0]), int(parts[1])
            box = cell.bounding_box()
            if box:
                bboxes['cells'][r][c] = [
                    box['x'],
                    box['y'],
                    box['x'] + box['width'],
                    box['y'] + box['height']
                ]

        return bboxes

    def render_and_save(self, is_clean=False):
        """Render the table as an image with optional noise and bounding boxes metadata.
        
        Args:
            is_clean: If True, saves a clean version without noise
        """
        try:
            # 1. Generate styled HTML
            html_content, style_name = generate_html_with_style(
                self.table_data, cfg.BASE_FONT_SIZE
            )

            # 2. Use Playwright to render and capture
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
                # Set content and wait for load
                page.set_content(html_content)
                page.wait_for_load_state('networkidle')
                
                # Get table element to calculate optimal viewport
                table = page.query_selector('table')
                if not table:
                    raise ValueError("Table element not found in HTML")
                
                table_box = table.bounding_box()
                
                # Set viewport to fit table with padding
                viewport_width = int(table_box['width'] + 40)
                viewport_height = int(table_box['height'] + 40)
                page.set_viewport_size({
                    'width': viewport_width,
                    'height': viewport_height
                })
                
                # Extract bounding boxes before screenshot
                bboxes = self._extract_bounding_boxes_with_playwright(page)
                
                # Take screenshot
                screenshot_bytes = page.screenshot(type='png', full_page=True)
                browser.close()

            # 3. Load screenshot as PIL Image
            from io import BytesIO
            pil_image = Image.open(BytesIO(screenshot_bytes))
            image_array = np.array(pil_image.convert("RGB"))

            # 4. Apply noise only if not clean version
            if is_clean:
                final_image = Image.fromarray(image_array)
                noise_profile = {"clean": True}
                jpeg_quality = 95  # High quality for clean images
            else:
                noisy_image_array, noise_profile = apply_noise(image_array)
                final_image = Image.fromarray(noisy_image_array)
                jpeg_quality = int(noise_profile.get("jpeg_quality", 90))

            # 5. Create subfolder structure: visual/{source_table_id}/
            table_subfolder = cfg.VISUAL_IMAGES_DIR / self.source_table_id
            table_subfolder.mkdir(parents=True, exist_ok=True)
            
            # Also create metadata subfolder
            metadata_subfolder = cfg.VISUAL_METADATA_DIR / self.source_table_id
            metadata_subfolder.mkdir(parents=True, exist_ok=True)

            # 6. Prepare filenames with lang_style_clean/uuid format
            if is_clean:
                image_filename = f"{self.lang_code}_{style_name}_clean.jpg"
                meta_filename = f"{self.lang_code}_{style_name}_clean.json"
                image_id = f"{self.lang_code}_{style_name}_clean"
            else:
                image_uuid = str(uuid.uuid4())[:8]
                image_filename = f"{self.lang_code}_{style_name}_{image_uuid}.jpg"
                meta_filename = f"{self.lang_code}_{style_name}_{image_uuid}.json"
                image_id = f"{self.lang_code}_{style_name}_{image_uuid}"

            metadata = {
                "image_id": image_id,
                "source_table_id": self.source_table_id,
                "language": self.lang_code,
                "style_profile": style_name,
                "noise_profile": noise_profile,
                "is_clean": is_clean,
                "bbox_coordinates": bboxes,
                "dimensions": {
                    "width": final_image.width,
                    "height": final_image.height
                },
            }

            # 7. Save image and metadata in subfolders
            final_image.save(table_subfolder / image_filename, quality=jpeg_quality)
            with open(metadata_subfolder / meta_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            return True

        except Exception as e:
            import traceback
            print(f"  [ERROR] Failed to render {self.source_table_id} in {self.lang_code}. Reason: {e}")
            traceback.print_exc()
            return False