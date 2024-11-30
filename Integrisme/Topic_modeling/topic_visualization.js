// Configuration
const config = {
    width: 1100,
    height: 800,
    topicRadius: 20,
    documentRadius: 5,
    linkStrengthScale: d3.scaleLinear().range([0, 1]),
    minZoom: 0.1,
    maxZoom: 4
};

// Create SVG with zoom container
const svg = d3.select('#visualization')
    .append('svg')
    .attr('width', config.width)
    .attr('height', config.height);

// Add zoom container
const g = svg.append('g');

// Create zoom behavior
const zoom = d3.zoom()
    .scaleExtent([config.minZoom, config.maxZoom])
    .on('zoom', (event) => {
        g.attr('transform', event.transform);
    });

// Enable zoom
svg.call(zoom);

// Add zoom controls to HTML
const zoomControls = d3.select('.controls')
    .append('div')
    .attr('class', 'zoom-controls')
    .style('margin-top', '10px');

zoomControls.append('button')
    .text('Zoom In')
    .on('click', () => {
        svg.transition()
            .duration(750)
            .call(zoom.scaleBy, 1.3);
    });

zoomControls.append('button')
    .text('Zoom Out')
    .on('click', () => {
        svg.transition()
            .duration(750)
            .call(zoom.scaleBy, 1 / 1.3);
    });

zoomControls.append('button')
    .text('Reset')
    .on('click', () => {
        svg.transition()
            .duration(750)
            .call(zoom.transform, d3.zoomIdentity);
    });

// Create tooltip
const tooltip = d3.select('body')
    .append('div')
    .attr('class', 'tooltip')
    .style('opacity', 0);

// Load and process data
async function loadData() {
    const data = await d3.json('./topic_modeling_results.json');
    return processData(data);
}

function processData(data) {
    // Create nodes with consistent IDs
    const nodes = [
        // Topics with their original IDs
        ...data.topics.map(t => ({
            ...t,
            type: 'topic',
            nodeId: `t${t.id}` // Prefix topic IDs with 't'
        })),
        // Documents with prefixed IDs
        ...data.documents.map(d => ({
            ...d,
            type: 'document',
            nodeId: `d${d.id}` // Prefix document IDs with 'd'
        }))
    ];
    
    // Create links with the new ID format
    const links = [];
    data.documents.forEach(doc => {
        doc.topic_weights.forEach((weight, topicIdx) => {
            if (weight > 0.2) { // Initial threshold
                links.push({
                    source: `t${topicIdx}`, // Reference topic ID
                    target: `d${doc.id}`,   // Reference document ID
                    weight: weight
                });
            }
        });
    });
    
    return { nodes, links };
}

function updateVisualization(data, threshold = 0.2) {
    // Filter links based on threshold
    const filteredLinks = data.links.filter(l => l.weight > threshold);
    
    // Create force simulation
    const simulation = d3.forceSimulation(data.nodes)
        .force('link', d3.forceLink(filteredLinks)
            .id(d => d.nodeId)
            .strength(d => d.weight))
        .force('charge', d3.forceManyBody().strength(-100))
        .force('center', d3.forceCenter(config.width / 2, config.height / 2))
        .force('collision', d3.forceCollide().radius(d => 
            d.type === 'topic' ? config.topicRadius : config.documentRadius));

    // Draw links
    const link = g.selectAll('.link')  // Use g instead of svg
        .data(filteredLinks)
        .join('line')
        .attr('class', 'link')
        .attr('stroke-width', d => d.weight * 2);

    // Draw nodes
    const node = g.selectAll('.node')  // Use g instead of svg
        .data(data.nodes)
        .join('circle')
        .attr('class', d => `node node-${d.type}`)
        .attr('r', d => d.type === 'topic' ? config.topicRadius : config.documentRadius)
        .call(drag(simulation));

    // Add tooltips
    node.on('mouseover', (event, d) => {
        let content;
        if (d.type === 'topic') {
            content = `
                <div class="tooltip-topic">
                    <h4>${d.label}</h4>
                    <div class="tooltip-prevalence">
                        Corpus coverage: ${(d.prevalence * 100).toFixed(1)}%
                    </div>
                    <div class="tooltip-words">
                        ${d.word_weights.map(([word, weight]) => 
                            `<span class="tooltip-word" style="opacity: ${weight}">${word}</span>`
                        ).join(' ')}
                    </div>
                </div>
            `;
        } else {
            content = `
                <div class="tooltip-document">
                    <h4>${d.title}</h4>
                    <div class="tooltip-meta">
                        Published: ${d.date}<br>
                        Source: ${d.publisher}
                    </div>
                </div>
            `;
        }
        
        tooltip.transition()
            .duration(200)
            .style('opacity', .9);
        tooltip.html(content)
            .style('left', (event.pageX + 10) + 'px')
            .style('top', (event.pageY - 10) + 'px');
    })
    .on('mouseout', () => {
        tooltip.transition()
            .duration(500)
            .style('opacity', 0);
    });

    // Update positions
    simulation.on('tick', () => {
        link
            .attr('x1', d => d.source.x)
            .attr('y1', d => d.source.y)
            .attr('x2', d => d.target.x)
            .attr('y2', d => d.target.y);

        node
            .attr('cx', d => d.x)
            .attr('cy', d => d.y);
    });
}

// Drag behavior
function drag(simulation) {
    function dragstarted(event) {
        if (!event.active) simulation.alphaTarget(0.3).restart();
        event.subject.fx = event.subject.x;
        event.subject.fy = event.subject.y;
    }
    
    function dragged(event) {
        event.subject.fx = event.x;
        event.subject.fy = event.y;
    }
    
    function dragended(event) {
        if (!event.active) simulation.alphaTarget(0);
        event.subject.fx = null;
        event.subject.fy = null;
    }
    
    return d3.drag()
        .on('start', dragstarted)
        .on('drag', dragged)
        .on('end', dragended);
}

// Initialize visualization
loadData().then(data => {
    updateVisualization(data);
    
    // Add threshold control
    d3.select('#threshold').on('input', function() {
        const value = this.value / 100;
        d3.select('#threshold-value').text(value.toFixed(2));
        updateVisualization(data, value);
    });
}); 