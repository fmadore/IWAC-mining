// Set up dimensions and margins
const margin = {top: 40, right: 80, bottom: 60, left: 50};
const width = 960 - margin.left - margin.right;
const height = 500 - margin.top - margin.bottom;

// Create SVG container
const svg = d3.select('#visualization')
  .append('svg')
    .attr('width', width + margin.left + margin.right)
    .attr('height', height + margin.top + margin.bottom)
    .attr('role', 'img')  // Accessibility
    .attr('aria-label', 'Sentiment Analysis Over Time');

// Add clipPath definition
svg.append('defs')
    .append('clipPath')
    .attr('id', 'clip')
    .append('rect')
    .attr('width', width)
    .attr('height', height);

// Create a group for zoom
const g = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

// Create a clipped group for the dots
const dotsGroup = g.append('g')
    .attr('clip-path', 'url(#clip)');

// Create scales
const x = d3.scaleTime()
  .range([0, width]);

const y = d3.scaleLinear()
  .range([height, 0]);

// Add axes
const xAxis = d3.axisBottom(x);
const yAxis = d3.axisLeft(y);

// Initialize current sentiment type and data store
let currentSentiment = 'compound';
let sentimentData = [];

// Create zoom behavior
const zoom = d3.zoom()
    .scaleExtent([0.5, 20])  // Set zoom limits
    .extent([[0, 0], [width, height]])
    .on('zoom', zoomed);

// Add zoom functionality to SVG
svg.call(zoom);

// Zoom function
function zoomed(event) {
    // Update x and y scales according to zoom
    const newX = event.transform.rescaleX(x);
    const newY = event.transform.rescaleY(y);
    
    // Update axes
    g.select('.x-axis').call(xAxis.scale(newX));
    g.select('.y-axis').call(yAxis.scale(newY));
    
    dotsGroup.selectAll('.dot')
        .attr('cx', d => newX(d.date))
        .attr('cy', d => newY(d[currentSentiment]));
}

// Add zoom controls
const zoomControls = d3.select('#visualization')
    .append('div')
    .attr('class', 'zoom-controls')
    .style('position', 'absolute')
    .style('top', '10px')
    .style('right', '10px');

zoomControls.append('button')
    .attr('class', 'zoom-button')
    .text('+')
    .on('click', () => zoomBy(1.5));

zoomControls.append('button')
    .attr('class', 'zoom-button')
    .text('-')
    .on('click', () => zoomBy(0.67));

zoomControls.append('button')
    .attr('class', 'zoom-button')
    .text('Reset')
    .on('click', resetZoom);

function zoomBy(factor) {
    svg.transition()
        .duration(750)
        .call(zoom.scaleBy, factor);
}

function resetZoom() {
    svg.transition()
        .duration(750)
        .call(zoom.transform, d3.zoomIdentity);
}

// Function to update the visualization
function updateVisualization() {
    currentSentiment = d3.select('#sentimentType').property('value');
    
    // Reset zoom first
    resetZoom();
    
    // Update y-axis domain for the selected sentiment
    y.domain(d3.extent(sentimentData, d => d[currentSentiment]));
    
    // Update y-axis
    g.select('.y-axis')
        .transition()
        .duration(750)
        .call(yAxis);
    
    dotsGroup.selectAll('.dot')
        .transition()
        .duration(750)
        .attr('cy', d => y(d[currentSentiment]));
}

// Load and process data
async function visualizeSentiment() {
    try {
        const data = await d3.json('integrisme_data_with_sentiment.json');
        
        // Process sentiment data
        sentimentData = data
            .filter(article => article.date && article.sentiment_analysis)
            .map(article => ({
                date: new Date(article.date.split('/')[0]),
                compound: article.sentiment_analysis.compound,
                positive: article.sentiment_analysis.positive,
                negative: article.sentiment_analysis.negative,
                neutral: article.sentiment_analysis.neutral,
                title: article['o:title'] || 'Sans titre',
                newspaper: article['dcterms:publisher']?.[0]?.['display_title'] || 'Unknown newspaper'
            }))
            .sort((a, b) => a.date - b.date);

        // Update scales
        x.domain(d3.extent(sentimentData, d => d.date));
        y.domain(d3.extent(sentimentData, d => d[currentSentiment]));

        // Add axes to SVG
        g.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis);

        g.append('g')
            .attr('class', 'y-axis')
            .call(yAxis);

        // Add axis labels
        g.append('text')
            .attr('class', 'x-label')
            .attr('text-anchor', 'middle')
            .attr('x', width / 2)
            .attr('y', height + 40)
            .text('Date');

        g.append('text')
            .attr('class', 'y-label')
            .attr('text-anchor', 'middle')
            .attr('transform', 'rotate(-90)')
            .attr('y', -40)
            .attr('x', -height / 2)
            .text('Sentiment');

        // Add interactive tooltip
        const tooltip = d3.select('body')
            .append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);

        // Add interactive dots
        dotsGroup.selectAll('.dot')
            .data(sentimentData)
            .enter()
            .append('circle')
            .attr('class', 'dot')
            .attr('cx', d => x(d.date))
            .attr('cy', d => y(d[currentSentiment]))
            .attr('r', 6)
            .on('mouseover', function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', 9);
                
                tooltip.transition()
                    .duration(200)
                    .style('opacity', .9);
                
                tooltip.html(`
                    <strong>${d.title}</strong><br/>
                    <em>${d.newspaper}</em><br/>
                    Date: ${d3.timeFormat('%Y-%m-%d')(d.date)}<br/>
                    Compound: ${d.compound.toFixed(2)}<br/>
                    Positive: ${d.positive.toFixed(2)}<br/>
                    Negative: ${d.negative.toFixed(2)}<br/>
                    Neutral: ${d.neutral.toFixed(2)}
                `)
                    .style('left', (event.pageX + 10) + 'px')
                    .style('top', (event.pageY - 28) + 'px');
            })
            .on('mouseout', function() {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', 6);
                
                tooltip.transition()
                    .duration(500)
                    .style('opacity', 0);
            });

        // Add event listener for sentiment type selector
        d3.select('#sentimentType').on('change', updateVisualization);
        
    } catch (error) {
        console.error('Error loading or processing data:', error);
    }
}

// Initialize visualization when DOM is loaded
document.addEventListener('DOMContentLoaded', visualizeSentiment);