// Add these configurations at the top
const config = {
    margin: { top: 80, right: 20, bottom: 10, left: 100 },
    cellSize: 10,
    cellPadding: 1,
    maxOpacity: 0.8,
    minOpacity: 0.2
};

// Update the data loading URL to use the GitHub raw file
const dataUrl = 'https://github.com/fmadore/Mining_IWAC/raw/refs/heads/main/Integrisme/Cooccurrence/data/cooccurrence.json';

// Load data and initialize visualization
d3.json(dataUrl).then(data => {
    // Initial setup
    const windowType = document.getElementById('window-type').value;
    createMatrix(data, windowType);

    // Add event listeners
    document.getElementById('window-type').addEventListener('change', (event) => {
        createMatrix(data, event.target.value);
    });

    document.getElementById('order').addEventListener('change', (event) => {
        createMatrix(data, document.getElementById('window-type').value);
    });
}).catch(error => {
    console.error('Error loading the data:', error);
    d3.select("#matrix")
        .append("p")
        .attr("class", "error")
        .text("Error loading visualization data. Please check the console for details.");
});

function createMatrix(data, windowType) {
    // Clear previous visualization
    d3.select("#matrix").selectAll("*").remove();

    const nodes = data[windowType].nodes;
    const links = data[windowType].links;

    // Create an adjacency matrix from links
    const matrix = Array(nodes.length).fill().map(() => Array(nodes.length).fill(0));
    links.forEach(link => {
        matrix[link.source][link.target] = link.value;
    });

    // Find max value for scaling
    const maxValue = d3.max(links, d => d.value);

    // Create color scale
    const colorScale = d3.scaleSequential()
        .domain([0, maxValue])
        .interpolator(d3.interpolateBlues);

    // Calculate size based on number of nodes
    const size = nodes.length * (config.cellSize + config.cellPadding);
    const width = size + config.margin.left + config.margin.right;
    const height = size + config.margin.top + config.margin.bottom;

    // Create SVG
    const svg = d3.select("#matrix")
        .append("svg")
        .attr("width", width)
        .attr("height", height)
        .append("g")
        .attr("transform", `translate(${config.margin.left},${config.margin.top})`);

    // Add rows
    const rows = svg.selectAll(".row")
        .data(nodes)
        .enter()
        .append("g")
        .attr("class", "row")
        .attr("transform", (d, i) => `translate(0,${i * (config.cellSize + config.cellPadding)})`);

    // Add cells
    rows.selectAll(".cell")
        .data((d, i) => matrix[i].map((value, j) => ({value, i, j})))
        .enter()
        .append("rect")
        .attr("class", "cell")
        .attr("x", (d, i) => i * (config.cellSize + config.cellPadding))
        .attr("width", config.cellSize)
        .attr("height", config.cellSize)
        .style("fill", d => d.value > 0 ? colorScale(d.value) : "#f8f9fa")
        .style("opacity", d => {
            if (d.value === 0) return 0.1;
            const normalizedValue = d.value / maxValue;
            return config.minOpacity + normalizedValue * (config.maxOpacity - config.minOpacity);
        })
        .on("mouseover", showTooltip)
        .on("mouseout", hideTooltip);

    // Add row labels
    rows.append("text")
        .attr("class", "label")
        .attr("x", -5)
        .attr("y", config.cellSize / 2)
        .attr("text-anchor", "end")
        .attr("alignment-baseline", "middle")
        .text(d => d.name)
        .style("font-size", "10px");

    // Add column labels
    svg.selectAll(".column-label")
        .data(nodes)
        .enter()
        .append("text")
        .attr("class", "label")
        .attr("x", (d, i) => i * (config.cellSize + config.cellPadding) + config.cellSize / 2)
        .attr("y", -5)
        .attr("transform", (d, i) => {
            const x = i * (config.cellSize + config.cellPadding) + config.cellSize / 2;
            return `rotate(-45,${x},-5)`;
        })
        .attr("text-anchor", "end")
        .text(d => d.name)
        .style("font-size", "10px");

    // Add tooltip div if it doesn't exist
    if (!d3.select("#tooltip").size()) {
        d3.select("body").append("div")
            .attr("id", "tooltip")
            .attr("class", "tooltip")
            .style("display", "none");
    }
}

function showTooltip(event, d) {
    if (d.value > 0) {
        const tooltip = d3.select("#tooltip");
        tooltip.style("display", "block")
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px")
            .html(`Co-occurrence: ${d.value}`);
    }
}

function hideTooltip() {
    d3.select("#tooltip").style("display", "none");
} 