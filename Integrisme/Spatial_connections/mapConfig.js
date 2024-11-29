const MapConfig = {
    width: 960,
    height: 600,
    projection: {
        scale: 300,
        center: [-4, 12],  // Center on West Africa
        translate: [480, 350]  // Adjusted for better centering
    },
    colors: {
        background: "#f0f0f0",
        land: "#f0f0f0",
        stroke: "#fff",
        circles: "red"
    },
    zoom: {
        min: 1,
        max: 8
    },
    circle: {
        minRadius: 5,
        maxRadius: 25,
        minOpacity: 0.5,
        maxOpacity: 0.8
    }
};

export default MapConfig; 