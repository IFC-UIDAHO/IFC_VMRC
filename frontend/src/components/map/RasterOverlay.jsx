import PropTypes from "prop-types";
import { ImageOverlay } from "react-leaflet";
import { backendUrl } from "../../lib/rasterApi";

export default function RasterOverlay({ overlayUrl, bounds }) {
  if (!overlayUrl || !bounds) {
    console.warn("RasterOverlay: Missing overlayUrl or bounds, not rendering");
    return null;
  }

  if (
    typeof bounds.west !== "number" ||
    typeof bounds.south !== "number" ||
    typeof bounds.east !== "number" ||
    typeof bounds.north !== "number"
  ) {
    console.error("RasterOverlay: Invalid bounds structure", bounds);
    return null;
  }

  const fullUrl = backendUrl(overlayUrl);

  const leafletBounds = [
    [bounds.south, bounds.west],
    [bounds.north, bounds.east],
  ];

  console.log("RasterOverlay: Rendering with bounds:", {
    west: bounds.west,
    south: bounds.south,
    east: bounds.east,
    north: bounds.north,
    leafletBounds,
    fullUrl,
  });

  return (
    <ImageOverlay
      url={fullUrl}
      bounds={leafletBounds}
      opacity={1.0}
      interactive={false}
      zIndex={200}
      className="vmrc-raster-overlay raster-overlay-pixelated"
    />
  );
}

RasterOverlay.propTypes = {
  overlayUrl: PropTypes.string,
  bounds: PropTypes.shape({
    north: PropTypes.number,
    south: PropTypes.number,
    east: PropTypes.number,
    west: PropTypes.number,
  }),
};