// Add these configurations at the top
const config = {
    margin: { top: 160, right: 20, bottom: 10, left: 120 },
    cellSize: 16,
    cellPadding: 3,
    maxOpacity: 1.0,
    minOpacity: 0.3,
    colors: {
        empty: "#f8f9fa",
        filled: d3.interpolateBlues
    }
};

// Add error handling for matrix container
function showError(message) {
    const matrixDiv = d3.select("#matrix");
    matrixDiv.selectAll("*").remove();
    matrixDiv
        .append("div")
        .attr("class", "error-message")
        .attr("role", "alert")
        .text(message);
}

// Update the data loading URL to use the correct path
const dataUrl = './data/cooccurrence.json';

// Load data and initialize visualization
d3.json(dataUrl).then(data => {
    if (!data) {
        throw new Error('No data received');
    }
    
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
    showError("Error loading visualization data. Please ensure the data file exists and is properly formatted.");
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

    // Get all non-zero values to better understand the distribution
    const nonZeroValues = links.map(d => d.value).filter(v => v > 0);
    
    // Calculate quartiles for better distribution understanding
    const quartiles = {
        q1: d3.quantile(nonZeroValues, 0.25),
        q2: d3.quantile(nonZeroValues, 0.5),
        q3: d3.quantile(nonZeroValues, 0.75)
    };

    // Create a more nuanced color scale using quantile breaks
    const colorScale = d3.scaleQuantile()
        .domain([0, quartiles.q1, quartiles.q2, quartiles.q3, d3.max(nonZeroValues)])
        .range([
            d3.interpolateBlues(0.1),  // Very light blue for lowest values
            d3.interpolateBlues(0.3),  // Light blue
            d3.interpolateBlues(0.5),  // Medium blue
            d3.interpolateBlues(0.7),  // Darker blue
            d3.interpolateBlues(0.9)   // Darkest blue
        ]);

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
        .data((d, i) => matrix[i].map((value, j) => ({value, i, j, nodes})))
        .enter()
        .append("rect")
        .attr("class", "cell")
        .attr("x", (d, i) => i * (config.cellSize + config.cellPadding))
        .attr("width", config.cellSize)
        .attr("height", config.cellSize)
        .style("fill", d => {
            if (d.value === 0) return config.colors.empty;
            return colorScale(d.value);
        })
        .style("opacity", d => {
            if (d.value === 0) return 0.05;
            // Use a more pronounced opacity scale
            const normalizedValue = d.value / maxValue;
            return config.minOpacity + Math.pow(normalizedValue, 0.5) * 
                   (config.maxOpacity - config.minOpacity);
        })
        .style("stroke", "#ddd")
        .style("stroke-width", 0.5)
        .attr("data-row", d => d.i)
        .attr("data-col", d => d.j)
        .on("mouseover", (event, d) => {
            showTooltip(event, d);
            highlightCell(d.i, d.j);
        })
        .on("mouseout", (event, d) => {
            hideTooltip();
            unhighlightCell();
        });

    // Add row labels with unique class
    rows.append("text")
        .attr("class", d => `label row-label-${nodes.indexOf(d)}`)
        .attr("x", -12)
        .attr("y", config.cellSize / 2)
        .attr("text-anchor", "end")
        .attr("alignment-baseline", "middle")
        .text(d => d.name)
        .style("font-size", "12px")
        .style("font-weight", "500");

    // Add column labels with unique class
    svg.selectAll(".column-label")
        .data(nodes)
        .enter()
        .append("text")
        .attr("class", (d, i) => `label col-label-${i}`)
        .attr("x", (d, i) => i * (config.cellSize + config.cellPadding) + config.cellSize / 2)
        .attr("y", -30)
        .attr("transform", (d, i) => {
            const x = i * (config.cellSize + config.cellPadding) + config.cellSize / 2;
            const y = -30;
            return `rotate(-65,${x},${y})`;
        })
        .attr("text-anchor", "end")
        .attr("dy", ".2em")
        .text(d => d.name)
        .style("font-size", "12px")
        .style("font-weight", "500");

    // Add tooltip div if it doesn't exist
    if (!d3.select("#tooltip").size()) {
        d3.select("body").append("div")
            .attr("id", "tooltip")
            .attr("class", "tooltip")
            .style("display", "none");
    }

    // Add these new functions within createMatrix
    function highlightCell(row, col) {
        d3.selectAll(".cell")
            .style("opacity", function() {
                const cellRow = +this.getAttribute("data-row");
                const cellCol = +this.getAttribute("data-col");
                if (cellRow === row && cellCol === col) {
                    return 1;
                }
                return 0.15; // Reduced opacity for non-highlighted cells
            });

        // Highlight the labels
        d3.selectAll(".label")
            .style("fill", "#999");
        
        d3.select(`.row-label-${row}`)
            .style("fill", "#2171b5")
            .style("font-weight", "bold");
        
        d3.select(`.col-label-${col}`)
            .style("fill", "#2171b5")
            .style("font-weight", "bold");
    }

    function unhighlightCell() {
        // Reset cell opacity
        d3.selectAll(".cell")
            .style("opacity", d => {
                if (d.value === 0) return 0.05;
                const normalizedValue = d.value / maxValue;
                return config.minOpacity + normalizedValue * (config.maxOpacity - config.minOpacity);
            });

        // Reset label styles
        d3.selectAll(".label")
            .style("fill", "#333")
            .style("font-weight", "500");
    }
}

function showTooltip(event, d) {
    if (d.value > 0) {
        const tooltip = d3.select("#tooltip");
        const source = d.nodes[d.i].name;
        const target = d.nodes[d.j].name;
        tooltip.style("display", "block")
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 10) + "px")
            .html(`${source} ↔ ${target}<br>Co-occurrences: ${d.value}`);
    }
}

function hideTooltip() {
    d3.select("#tooltip").style("display", "none");
} 