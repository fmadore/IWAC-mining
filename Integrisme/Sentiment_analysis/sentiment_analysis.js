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
    .attr('aria-label', 'Sentiment Analysis Over Time')
  .append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`);

// Create scales
const x = d3.scaleTime()
  .range([0, width]);

const y = d3.scaleLinear()
  .range([height, 0]);

// Create line generator
const line = d3.line()
  .x(d => x(d.date))
  .y(d => y(d[currentSentiment]));

// Add axes
const xAxis = d3.axisBottom(x);
const yAxis = d3.axisLeft(y);

// Initialize current sentiment type and data store
let currentSentiment = 'compound';
let sentimentData = [];

// Function to update the visualization
function updateVisualization() {
    currentSentiment = d3.select('#sentimentType').property('value');
    
    // Update y-axis domain for the selected sentiment
    y.domain(d3.extent(sentimentData, d => d[currentSentiment]));
    
    // Update y-axis
    svg.select('.y-axis')
        .transition()
        .duration(750)
        .call(yAxis);
    
    // Update line
    svg.select('.line')
        .transition()
        .duration(750)
        .attr('d', line(sentimentData));
    
    // Update dots
    svg.selectAll('.dot')
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
                date: new Date(article.date),
                compound: article.sentiment_analysis.compound,
                positive: article.sentiment_analysis.positive,
                negative: article.sentiment_analysis.negative,
                neutral: article.sentiment_analysis.neutral,
                title: article['o:title'] || 'Sans titre'
            }))
            .sort((a, b) => a.date - b.date);

        // Update scales
        x.domain(d3.extent(sentimentData, d => d.date));
        y.domain(d3.extent(sentimentData, d => d[currentSentiment]));

        // Add axes to SVG
        svg.append('g')
            .attr('class', 'x-axis')
            .attr('transform', `translate(0,${height})`)
            .call(xAxis);

        svg.append('g')
            .attr('class', 'y-axis')
            .call(yAxis);

        // Add axis labels
        svg.append('text')
            .attr('class', 'x-label')
            .attr('text-anchor', 'middle')
            .attr('x', width / 2)
            .attr('y', height + 40)
            .text('Date');

        svg.append('text')
            .attr('class', 'y-label')
            .attr('text-anchor', 'middle')
            .attr('transform', 'rotate(-90)')
            .attr('y', -40)
            .attr('x', -height / 2)
            .text('Sentiment');

        // Add the line path
        svg.append('path')
            .datum(sentimentData)
            .attr('class', 'line')
            .attr('d', line)
            .attr('fill', 'none')
            .attr('stroke', 'steelblue')
            .attr('stroke-width', 2);

        // Add interactive tooltip
        const tooltip = d3.select('body')
            .append('div')
            .attr('class', 'tooltip')
            .style('opacity', 0);

        // Add interactive dots
        svg.selectAll('.dot')
            .data(sentimentData)
            .enter()
            .append('circle')
            .attr('class', 'dot')
            .attr('cx', d => x(d.date))
            .attr('cy', d => y(d[currentSentiment]))
            .attr('r', 5)
            .on('mouseover', function(event, d) {
                d3.select(this)
                    .transition()
                    .duration(200)
                    .attr('r', 8);
                
                tooltip.transition()
                    .duration(200)
                    .style('opacity', .9);
                
                tooltip.html(`
                    <strong>${d.title}</strong><br/>
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
                    .attr('r', 5);
                
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