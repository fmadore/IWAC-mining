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
    .attr('aria-label', 'Sentiment Analysis Timeline')
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
  .y(d => y(d.sentiment));

// Add axes
const xAxis = d3.axisBottom(x);
const yAxis = d3.axisLeft(y);

// Load and process data
async function visualizeSentiment() {
  try {
    const data = await d3.json('integrisme_data.json');
    
    // Process sentiment data
    const sentimentData = data.map(article => {
      return {
        date: new Date(article.date),
        compound: article.sentiment_analysis.compound,
        positive: article.sentiment_analysis.positive,
        negative: article.sentiment_analysis.negative,
        neutral: article.sentiment_analysis.neutral
      };
    }).sort((a, b) => a.date - b.date);

    // Update scales
    x.domain(d3.extent(sentimentData, d => d.date));
    y.domain(d3.extent(sentimentData, d => d.sentiment));

    // Add axes to SVG
    svg.append('g')
      .attr('class', 'x-axis')
      .attr('transform', `translate(0,${height})`)
      .call(xAxis);

    svg.append('g')
      .attr('class', 'y-axis')
      .call(yAxis);

    // Add the line path
    const path = svg.append('path')
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
      .attr('cy', d => y(d.sentiment))
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

    // Add responsive behavior
    function resize() {
      // Update width and height based on container
      const container = d3.select('#visualization').node().getBoundingClientRect();
      const newWidth = container.width - margin.left - margin.right;
      const newHeight = container.height - margin.top - margin.bottom;

      // Update SVG dimensions
      svg.attr('width', newWidth + margin.left + margin.right)
         .attr('height', newHeight + margin.top + margin.bottom);

      // Update scales
      x.range([0, newWidth]);
      y.range([newHeight, 0]);

      // Update axes and line
      svg.select('.x-axis')
        .attr('transform', `translate(0,${newHeight})`)
        .call(xAxis);

      svg.select('.y-axis')
        .call(yAxis);

      svg.select('.line')
        .attr('d', line);

      // Update dots
      svg.selectAll('.dot')
        .attr('cx', d => x(d.date))
        .attr('cy', d => y(d.sentiment));
    }

    // Add window resize listener
    window.addEventListener('resize', resize);
    
  } catch (error) {
    console.error('Error loading or processing data:', error);
  }
} 