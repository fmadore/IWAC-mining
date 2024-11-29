const MapConfig = {
    width: 960,
    height: 600,
    projection: {
        scale: 150,
        center: [0, 20],
        translate: [480, 300]
    },
    colors: {
        background: "#f0f0f0",
        land: "#d3d3d3",
        stroke: "#999",
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