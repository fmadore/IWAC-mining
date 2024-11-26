// Matrix visualization
class MatrixVisualization {
    constructor() {
        this.margin = {top: 100, right: 100, bottom: 10, left: 100};
        this.width = 800;
        this.height = 800;
        this.data = null;
        this.transitionDuration = 750;
    }

    async initialize() {
        try {
            // Load the data
            this.data = await d3.json('data/cooccurrence.json');
            
            // Create the SVG container
            this.svg = d3.select('#matrix')
                .append('svg')
                .attr('width', this.width + this.margin.left + this.margin.right)
                .attr('height', this.height + this.margin.top + this.margin.bottom)
                .append('g')
                .attr('transform', `translate(${this.margin.left},${this.margin.top})`);

            // Add tooltip div if it doesn't exist
            if (!d3.select('body').select('.tooltip').size()) {
                d3.select('body').append('div')
                    .attr('class', 'tooltip')
                    .style('opacity', 0);
            }

            // Create scales
            this.x = d3.scaleBand().range([0, this.width]).padding(0.05);
            this.y = d3.scaleBand().range([0, this.height]).padding(0.05);
            this.color = d3.scaleSequential()
                .interpolator(d3.interpolateBlues)
                .domain([0, d3.max(this.data.links, d => d.value)]);

            // Initialize the visualization
            this.updateVisualization('name');

            // Add event listeners for ordering
            d3.select('#order').on('change', (event) => {
                this.updateVisualization(event.target.value);
            });
        } catch (error) {
            console.error('Error initializing visualization:', error);
        }
    }

    computeNodeDegrees() {
        const degrees = new Map();
        this.data.nodes.forEach(node => {
            const links = this.data.links.filter(l => 
                l.source === node.id || l.target === node.id
            );
            degrees.set(node.id, d3.sum(links, l => l.value));
        });
        return degrees;
    }

    updateVisualization(orderBy) {
        try {
            // Order nodes
            const nodes = [...this.data.nodes];
            const degrees = this.computeNodeDegrees();

            switch(orderBy) {
                case 'frequency':
                    nodes.sort((a, b) => degrees.get(b.id) - degrees.get(a.id));
                    break;
                case 'cluster':
                    nodes.sort((a, b) => {
                        const aStrength = this.data.links.filter(l => l.source === a.id || l.target === a.id)
                            .reduce((sum, l) => sum + l.value, 0);
                        const bStrength = this.data.links.filter(l => l.source === b.id || l.target === b.id)
                            .reduce((sum, l) => sum + l.value, 0);
                        return bStrength - aStrength;
                    });
                    break;
                default: // 'name'
                    nodes.sort((a, b) => a.name.localeCompare(b.name));
            }

            // Update scales
            this.x.domain(nodes.map(d => d.id));
            this.y.domain(nodes.map(d => d.id));

            // Update rows with transition
            const rows = this.svg.selectAll('.row')
                .data(nodes, d => d.id);

            // Exit old rows
            rows.exit().remove();

            // Enter new rows
            const rowsEnter = rows.enter()
                .append('g')
                .attr('class', 'row');

            // Update all rows with transition
            rows.merge(rowsEnter)
                .transition()
                .duration(this.transitionDuration)
                .attr('transform', d => `translate(0,${this.y(d.id)})`);

            // Update row labels
            const rowLabels = rows.merge(rowsEnter).selectAll('text')
                .data(d => [d]);

            rowLabels.enter()
                .append('text')
                .merge(rowLabels)
                .attr('x', -6)
                .attr('y', this.y.bandwidth() / 2)
                .attr('dy', '.32em')
                .attr('class', 'row-label')
                .attr('alignment-baseline', 'middle')
                .text(d => d.name);

            // Update columns with transition
            const columns = this.svg.selectAll('.column')
                .data(nodes, d => d.id);

            // Exit old columns
            columns.exit().remove();

            // Enter new columns
            const columnsEnter = columns.enter()
                .append('g')
                .attr('class', 'column');

            // Update all columns with transition
            columns.merge(columnsEnter)
                .transition()
                .duration(this.transitionDuration)
                .attr('transform', d => `translate(${this.x(d.id)},-6)`);

            // Update column labels
            const columnLabels = columns.merge(columnsEnter).selectAll('text')
                .data(d => [d]);

            columnLabels.enter()
                .append('text')
                .merge(columnLabels)
                .attr('x', 5)
                .attr('y', -5)
                .attr('class', 'column-label')
                .attr('alignment-baseline', 'hanging')
                .attr('transform', 'rotate(-45)')
                .text(d => d.name);

            // Create cell data with new ordering
            const cellData = [];
            nodes.forEach((source, i) => {
                nodes.forEach((target, j) => {
                    const link = this.data.links.find(l => 
                        (l.source === source.id && l.target === target.id) ||
                        (l.source === target.id && l.target === source.id)
                    );
                    cellData.push({
                        source: source.id,
                        target: target.id,
                        value: link ? link.value : 0,
                        x: i,
                        y: j
                    });
                });
            });

            // Update cells with transition
            const cells = this.svg.selectAll('.cell')
                .data(cellData, d => `${d.source}-${d.target}`);

            // Exit old cells
            cells.exit().remove();

            // Enter new cells
            const cellsEnter = cells.enter()
                .append('rect')
                .attr('class', 'cell')
                .style('fill', d => d.value > 0 ? this.color(d.value) : '#f8f9fa')
                .style('opacity', d => d.value > 0 ? 1 : 0.1);

            // Update all cells with transition
            cells.merge(cellsEnter)
                .transition()
                .duration(this.transitionDuration)
                .attr('x', d => this.x(d.source))
                .attr('y', d => this.y(d.target))
                .attr('width', this.x.bandwidth())
                .attr('height', this.y.bandwidth())
                .style('fill', d => d.value > 0 ? this.color(d.value) : '#f8f9fa')
                .style('opacity', d => d.value > 0 ? 1 : 0.1);

            // Add tooltips (needs to be outside transition)
            this.svg.selectAll('.cell')
                .on('mouseover', (event, d) => {
                    if (d.value > 0) {
                        const sourceName = nodes.find(n => n.id === d.source).name;
                        const targetName = nodes.find(n => n.id === d.target).name;
                        
                        d3.select('.tooltip')
                            .transition()
                            .duration(200)
                            .style('opacity', .9);
                        d3.select('.tooltip')
                            .html(`${sourceName} - ${targetName}<br/>Cooccurrences: ${d.value}`)
                            .style('left', (event.pageX + 10) + 'px')
                            .style('top', (event.pageY - 28) + 'px');
                    }
                })
                .on('mouseout', () => {
                    d3.select('.tooltip')
                        .transition()
                        .duration(500)
                        .style('opacity', 0);
                });

        } catch (error) {
            console.error('Error updating visualization:', error);
        }
    }
}

// Initialize the visualization when the page loads
document.addEventListener('DOMContentLoaded', () => {
    const matrix = new MatrixVisualization();
    matrix.initialize();
}); 