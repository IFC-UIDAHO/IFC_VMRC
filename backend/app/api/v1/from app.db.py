from app.db.session import SessionLocal
from app.models.raster_layer import RasterLayer

import os
from pathlib import Path

db = SessionLocal()

RASTER_ROOT = Path(os.getenv("RASTER_ROOT", "/data/Mortality"))

layer = RasterLayer(
    name="DF Dry April demo",
    description="Demo raster for clipping",
    storage_path=str(
        RASTER_ROOT / "Mortality-DEC30" / "Douglas_Fir" / "h" / "M_DF_D04_h.tif"
    ),
)
db.add(layer)
db.commit()
db.refresh(layer)
print(layer.id)

db.close()
