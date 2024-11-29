import MapConfig from './mapConfig.js';

export default class MapViz {
    constructor(container) {
        this.svg = d3.select(container);
        this.tooltip = d3.select(".tooltip");
        this.width = MapConfig.width;
        this.height = MapConfig.height;
        this.g = this.svg.append("g");
        this.setupMap();
    }

    setupMap() {
        // Add background
        this.svg.append("rect")
            .attr("width", this.width)
            .attr("height", this.height)
            .attr("fill", MapConfig.colors.background);

        // Create projection
        this.projection = d3.geoMercator()
            .scale(MapConfig.projection.scale)
            .center(MapConfig.projection.center)
            .translate([this.width / 2, this.height / 2]);

        this.path = d3.geoPath().projection(this.projection);
        this.setupZoom();
    }

    setupZoom() {
        const zoom = d3.zoom()
            .scaleExtent([MapConfig.zoom.min, MapConfig.zoom.max])
            .on("zoom", (event) => this.handleZoom(event));

        this.svg.call(zoom);
    }

    handleZoom(event) {
        this.g.attr("transform", event.transform);
        this.g.selectAll("circle")
            .attr("r", d => this.radiusScale(d.properties.mentions) / event.transform.k);
        this.g.selectAll("path")
            .style("stroke-width", 0.5 / event.transform.k);
    }

    createScales(maxMentions) {
        this.radiusScale = d3.scaleSqrt()
            .domain([1, maxMentions])
            .range([MapConfig.circle.minRadius, MapConfig.circle.maxRadius]);

        this.opacityScale = d3.scaleLinear()
            .domain([1, maxMentions])
            .range([MapConfig.circle.minOpacity, MapConfig.circle.maxOpacity]);
    }

    drawMap(topoData) {
        // Clear any existing paths
        this.g.selectAll("path").remove();

        this.g.append("g")
            .selectAll("path")
            .data(topoData.features)
            .join("path")
            .attr("d", this.path)
            .attr("fill", MapConfig.colors.land)
            .style("stroke", MapConfig.colors.stroke)
            .style("stroke-width", 0.5);
    }

    drawCircles(data) {
        // Clear any existing circles
        this.g.selectAll("circle").remove();

        const maxMentions = d3.max(data.features, d => d.properties.mentions);
        this.createScales(maxMentions);

        this.g.selectAll("circle")
            .data(data.features)
            .join("circle")
            .attr("class", "circle-marker")
            .attr("cx", d => this.projection(d.geometry.coordinates)[0])
            .attr("cy", d => this.projection(d.geometry.coordinates)[1])
            .attr("r", d => this.radiusScale(d.properties.mentions))
            .style("fill", MapConfig.colors.circles)
            .style("fill-opacity", d => this.opacityScale(d.properties.mentions))
            .on("mouseover", (event, d) => this.handleMouseOver(event, d))
            .on("mouseout", () => this.handleMouseOut());
    }

    handleMouseOut() {
        // Fix the event target reference
        d3.select(this.lastHoveredElement)
            .style("stroke", "#fff")
            .style("stroke-width", "1px");
        
        this.tooltip.transition()
            .duration(500)
            .style("opacity", 0);
    }

    handleMouseOver(event, d) {
        // Store the last hovered element
        this.lastHoveredElement = event.target;

        d3.select(event.target)
            .style("stroke", "#000")
            .style("stroke-width", "2px");
        
        this.tooltip.transition()
            .duration(200)
            .style("opacity", .9);
        
        this.tooltip.html(`<strong>${d.properties.name}</strong><br/>${d.properties.mentions} mentions`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
    }
} 