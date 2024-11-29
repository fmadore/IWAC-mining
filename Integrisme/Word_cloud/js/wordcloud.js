function initializeVisualization(words) {
    console.log('Initializing visualization with words:', words); // Debug log

    // Clear previous content
    d3.select("#word-cloud").html("");

    // Set up the word cloud layout
    const layout = d3.layout.cloud()
        .size([CONFIG.width, CONFIG.height])
        .words(words)
        .padding(CONFIG.padding)
        .rotate(() => 0)
        .fontSize(d => Math.sqrt(d.size) * 10)
        .on("end", draw);

    // Start the layout
    layout.start();

    // Function to draw the word cloud
    function draw(words) {
        const svg = d3.select("#word-cloud")
            .append("svg")
            .attr("width", CONFIG.width)
            .attr("height", CONFIG.height)
            .append("g")
            .attr("transform", `translate(${CONFIG.width/2},${CONFIG.height/2})`);

        // Add words to the cloud
        svg.selectAll("text")
            .data(words)
            .enter()
            .append("text")
            .style("font-size", d => `${d.size}px`)
            .style("font-family", "Impact")
            .style("fill", () => d3.schemeCategory10[Math.floor(Math.random() * 10)])
            .attr("text-anchor", "middle")
            .attr("transform", d => `translate(${d.x},${d.y})rotate(${d.rotate})`)
            .text(d => d.text);
    }
} 