import json
import numpy as np
import cv2
from pathlib import Path
from shapely.geometry import shape, box
from shapely.strtree import STRtree

json_path = "D:\\files\\HE\\train\\raw\\1\\a-202500006-2.geojson"

with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        for feature in data.get('features', []):
            geom = shape(feature.get('geometry', {}))

print(data)
# print(geom)