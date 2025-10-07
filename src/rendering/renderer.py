import json
import uuid
import tempfile
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
        bboxes = {
            'header': [None] * len(self.table_data.get('columns', [])),
            'cells': [
                [None] * len(self.table_data.get('columns', []))
                for _ in self.table_data.get('data', [])
            ],
        }

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

    def render_and_save(self):
        try:
            html_content, style_name = generate_html_with_style(
                self.table_data, cfg.BASE_FONT_SIZE
            )

           
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                
               
                page.set_content(html_content)
                page.wait_for_load_state('networkidle')
                
               
                table = page.query_selector('table')
                if not table:
                    raise ValueError("Table element not found in HTML")
                
                table_box = table.bounding_box()
                
               
                viewport_width = int(table_box['width'] + 40)
                viewport_height = int(table_box['height'] + 40)
                page.set_viewport_size({
                    'width': viewport_width,
                    'height': viewport_height
                })
                
              
                bboxes = self._extract_bounding_boxes_with_playwright(page)
                
  
                screenshot_bytes = page.screenshot(type='png', full_page=True)
                browser.close()


            from io import BytesIO
            pil_image = Image.open(BytesIO(screenshot_bytes))
            image_array = np.array(pil_image.convert("RGB"))


            noisy_image_array, noise_profile = apply_noise(image_array)
            noisy_image = Image.fromarray(noisy_image_array)


            table_subfolder = cfg.VISUAL_IMAGES_DIR / self.source_table_id
            table_subfolder.mkdir(parents=True, exist_ok=True)
            

            metadata_subfolder = cfg.VISUAL_METADATA_DIR / self.source_table_id
            metadata_subfolder.mkdir(parents=True, exist_ok=True)


            image_uuid = str(uuid.uuid4())[:8]
            image_filename = f"{self.lang_code}_{style_name}_{image_uuid}.jpg"
            meta_filename = f"{self.lang_code}_{style_name}_{image_uuid}.json"

            metadata = {
                "image_id": f"{self.lang_code}_{style_name}_{image_uuid}",
                "source_table_id": self.source_table_id,
                "language": self.lang_code,
                "style_profile": style_name,
                "noise_profile": noise_profile,
                "bbox_coordinates": bboxes,
                "dimensions": {
                    "width": noisy_image.width,
                    "height": noisy_image.height
                },
            }

            jpeg_quality = int(noise_profile.get("jpeg_quality", 90))
            noisy_image.save(table_subfolder / image_filename, quality=jpeg_quality)
            with open(metadata_subfolder / meta_filename, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)

            return True

        except Exception as e:
            import traceback
            print(f"  [ERROR] Failed to render {self.source_table_id} in {self.lang_code}. Reason: {e}")
            traceback.print_exc()
            return False