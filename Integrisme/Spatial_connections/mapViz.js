import MapConfig from './mapConfig.js';

export default class MapViz {
    constructor(container) {
        console.log("Initializing MapViz with container:", container);
        this.svg = d3.select(container);
        if (this.svg.empty()) {
            throw new Error(`Could not find element: ${container}`);
        }
        this.svg.selectAll("*").remove();
        
        // Add background rect first
        this.svg.append("rect")
            .attr("width", "100%")
            .attr("height", "100%")
            .attr("fill", MapConfig.colors.background);
        
        // Then create the group for map elements
        this.g = this.svg.append("g");
        
        this.tooltip = d3.select(".tooltip");
        this.width = MapConfig.width;
        this.height = MapConfig.height;
        this.setupMap();
    }

    setupMap() {
        this.projection = d3.geoMercator()
            .scale(MapConfig.projection.scale)
            .center(MapConfig.projection.center)
            .translate(MapConfig.projection.translate);

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
            .style("stroke-width", `${0.5 / event.transform.k}px`);
        this.svg.select(".legend")
            .attr("transform", `translate(${(this.width - 220) / event.transform.k}, ${(this.height - 70) / event.transform.k})`);
    }

    createScales(maxMentions) {
        this.radiusScale = d3.scaleSqrt()
            .domain([1, maxMentions])
            .range([MapConfig.circle.minRadius, MapConfig.circle.maxRadius]);

        this.opacityScale = d3.scaleLinear()
            .domain([1, maxMentions])
            .range([MapConfig.circle.minOpacity, MapConfig.circle.maxOpacity]);
    }

    createChoroplethScale(data, topoData) {
        // Create map of country mentions
        const countryMentions = new Map();
        
        // Function to find which country contains a point
        const findCountry = (coords) => {
            const point = this.projection(coords);
            let containingCountry = null;
            
            topoData.features.forEach(feature => {
                if (d3.geoContains(feature, coords)) {
                    containingCountry = feature.properties.name;
                }
            });
            
            return containingCountry;
        };

        // Count mentions for each location
        data.features.forEach(feature => {
            const coords = feature.geometry.coordinates;
            const country = findCountry(coords);
            if (country) {
                const mentions = feature.properties.mentions;
                countryMentions.set(country, (countryMentions.get(country) || 0) + mentions);
            }
        });

        // Create color scale
        this.choroplethScale = d3.scaleQuantile()
            .domain([0, ...countryMentions.values()])
            .range(MapConfig.colors.choropleth.scale);

        return countryMentions;
    }

    drawMap(topoData, locationData) {
        console.log("Drawing map with data:", topoData);
        
        const countryMentions = this.createChoroplethScale(locationData, topoData);
        
        this.g.selectAll("path").remove();
        
        this.g.selectAll("path")
            .data(topoData.features)
            .join("path")
            .attr("d", this.path)
            .attr("fill", d => {
                const mentions = countryMentions.get(d.properties.name);
                return mentions ? this.choroplethScale(mentions) : MapConfig.colors.choropleth.noData;
            })
            .attr("stroke", MapConfig.colors.stroke)
            .attr("stroke-width", "0.5px")
            .attr("vector-effect", "non-scaling-stroke")
            .on("mouseover", (event, d) => {
                const mentions = countryMentions.get(d.properties.name) || 0;
                this.tooltip.transition()
                    .duration(200)
                    .style("opacity", .9);
                this.tooltip.html(`<strong>${d.properties.name}</strong><br/>${mentions} mentions`)
                    .style("left", (event.pageX + 10) + "px")
                    .style("top", (event.pageY - 28) + "px");
            })
            .on("mouseout", () => this.handleMouseOut());

        this.drawLegend(countryMentions);
    }

    drawCircles(data) {
        console.log("Drawing circles with data:", data);
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
            .on("mouseover", (event, d) => this.handleCircleMouseOver(event, d))
            .on("mouseout", () => this.handleMouseOut());
    }

    handleMouseOver(event, d, countryMentions) {
        const mentions = countryMentions.get(d.properties.name) || 0;
        this.tooltip.transition()
            .duration(200)
            .style("opacity", .9);
        this.tooltip.html(`<strong>${d.properties.name}</strong><br/>${mentions} mentions`)
            .style("left", (event.pageX + 10) + "px")
            .style("top", (event.pageY - 28) + "px");
    }

    handleCircleMouseOver(event, d) {
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

    handleMouseOut() {
        d3.select(event.target)
            .style("stroke", "#fff")
            .style("stroke-width", "1px");
        
        this.tooltip.transition()
            .duration(500)
            .style("opacity", 0);
    }

    drawLegend(countryMentions) {
        const legendWidth = 200;
        const legendHeight = 50;
        
        const legend = this.svg.append("g")
            .attr("class", "legend")
            .attr("transform", `translate(${this.width - legendWidth - 20}, ${this.height - legendHeight - 20})`);

        const extent = d3.extent([...countryMentions.values()]);
        const legendScale = d3.scaleLinear()
            .domain(extent)
            .range([0, legendWidth]);

        const legendAxis = d3.axisBottom(legendScale)
            .ticks(5);

        const gradientData = MapConfig.colors.choropleth.scale;
        
        const gradient = legend.append("defs")
            .append("linearGradient")
            .attr("id", "legend-gradient")
            .attr("x1", "0%")
            .attr("x2", "100%")
            .attr("y1", "0%")
            .attr("y2", "0%");

        gradient.selectAll("stop")
            .data(gradientData)
            .enter()
            .append("stop")
            .attr("offset", (d, i) => `${(i * 100) / (gradientData.length - 1)}%`)
            .attr("stop-color", d => d);

        legend.append("rect")
            .attr("width", legendWidth)
            .attr("height", 10)
            .style("fill", "url(#legend-gradient)");

        legend.append("g")
            .attr("transform", `translate(0, 10)`)
            .call(legendAxis);
    }
} 