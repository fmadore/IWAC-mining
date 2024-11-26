// Matrix visualization
class MatrixVisualization {
    constructor() {
        this.margin = {top: 80, right: 0, bottom: 10, left: 80};
        this.width = 720;
        this.height = 720;
        this.data = null;
    }

    async initialize() {
        // Load the data
        this.data = await d3.json('data/cooccurrence.json');
        
        // Create the SVG container
        this.svg = d3.select('#matrix')
            .append('svg')
            .attr('width', this.width + this.margin.left + this.margin.right)
            .attr('height', this.height + this.margin.top + this.margin.bottom)
            .append('g')
            .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

        // Create scales
        this.x = d3.scaleBand().range([0, this.width]);
        this.y = d3.scaleBand().range([0, this.height]);
        this.color = d3.scaleLinear()
            .domain([0, d3.max(this.data.links, d => d.value)])
            .range(['#f7fbff', '#08306b']);

        // Initialize the visualization
        this.updateVisualization('name');

        // Add event listener for ordering
        d3.select('#order').on('change', (event) => {
            this.updateVisualization(event.target.value);
        });
    }

    updateVisualization(orderBy) {
        // Order nodes
        const nodes = this.data.nodes;
        switch(orderBy) {
            case 'count':
                nodes.sort((a, b) => {
                    const aCount = this.data.links.filter(l => 
                        l.source === nodes.indexOf(a) || l.target === nodes.indexOf(a)
                    ).reduce((sum, l) => sum + l.value, 0);
                    const bCount = this.data.links.filter(l => 
                        l.source === nodes.indexOf(b) || l.target === nodes.indexOf(b)
                    ).reduce((sum, l) => sum + l.value, 0);
                    return bCount - aCount;
                });
                break;
            case 'cluster':
                // Simple clustering based on connection strength
                // More sophisticated clustering could be implemented
                nodes.sort((a, b) => {
                    const aConnections = this.data.links.filter(l => 
                        l.source === nodes.indexOf(a) || l.target === nodes.indexOf(a)
                    );
                    const bConnections = this.data.links.filter(l => 
                        l.source === nodes.indexOf(b) || l.target === nodes.indexOf(b)
                    );
                    return bConnections.length - aConnections.length;
                });
                break;
            default: // 'name'
                nodes.sort((a, b) => a.name.localeCompare(b.name));
        }

        // Update scales
        this.x.domain(nodes.map(d => d.id));
        this.y.domain(nodes.map(d => d.id));

        // Create the rows
        const rows = this.svg.selectAll('.row')
            .data(nodes)
            .join('g')
            .attr('class', 'row')
            .attr('transform', d => `translate(0,${this.y(d.id)})`);

        // Create row labels
        rows.selectAll('text')
            .data(d => [d])
            .join('text')
            .attr('x', -6)
            .attr('y', this.y.bandwidth() / 2)
            .attr('dy', '.32em')
            .text(d => d.name);

        // Create the columns
        const columns = this.svg.selectAll('.column')
            .data(nodes)
            .join('g')
            .attr('class', 'column')
            .attr('transform', d => `translate(${this.x(d.id)},-6)`);

        // Create column labels
        columns.selectAll('text')
            .data(d => [d])
            .join('text')
            .attr('x', 6)
            .attr('y', 0)
            .attr('dy', '.32em')
            .text(d => d.name);

        // Create cells
        const cells = rows.selectAll('.cell')
            .data(d => nodes.map(x => {
                const link = this.data.links.find(l => 
                    (l.source === nodes.indexOf(d) && l.target === nodes.indexOf(x)) ||
                    (l.source === nodes.indexOf(x) && l.target === nodes.indexOf(d))
                );
                return {
                    x: x.id,
                    y: d.id,
                    value: link ? link.value : 0
                };
            }))
            .join('rect')
            .attr('class', 'cell')
            .attr('x', d => this.x(d.x))
            .attr('width', this.x.bandwidth())
            .attr('height', this.y.bandwidth())
            .style('fill', d => this.color(d.value));

        // Add tooltip
        const tooltip = d3.select('body')
            .append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);

        cells
            .on('mouseover', (event, d) => {
                tooltip.transition()
                    .duration(200)
                    .style('opacity', .9);
                tooltip.html(`${d.x} - ${d.y}<br/>Cooccurrences: ${d.value}`)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mouseout', () => {
                tooltip.transition()
                    .duration(500)
                    .style('opacity', 0);
            });
    }
}

// Initialize the visualization when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const matrix = new MatrixVisualization();
    matrix.initialize();
}); 