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
        // Create map of country mentions using ISO codes
        const countryMentions = new Map();
        
        // Function to find which country contains a point
        const findCountry = (coords) => {
            let containingCountry = null;
            
            topoData.features.forEach(feature => {
                if (d3.geoContains(feature, coords)) {
                    // Use ISO code instead of name
                    containingCountry = feature.id || feature.properties.iso_a3;
                }
            });
            
            return containingCountry;
        };

        // Count mentions for each location
        data.features.forEach(feature => {
            const coords = feature.geometry.coordinates;
            const countryCode = findCountry(coords);
            if (countryCode) {
                const mentions = feature.properties.mentions;
                countryMentions.set(countryCode, (countryMentions.get(countryCode) || 0) + mentions);
            }
        });

        // Create color scale using threshold scale
        const values = [...countryMentions.values()];
        const max = d3.max(values);
        
        this.choroplethScale = d3.scaleThreshold()
            .domain([1, 5, 10, 25, 50, 100, 150]) // Custom breakpoints for better distribution
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
                const mentions = countryMentions.get(d.id || d.properties.iso_a3);
                return mentions ? this.choroplethScale(mentions) : MapConfig.colors.choropleth.noData;
            })
            .attr("stroke", MapConfig.colors.stroke)
            .attr("stroke-width", "0.5px")
            .attr("vector-effect", "non-scaling-stroke")
            .on("mouseover", (event, d) => {
                const mentions = countryMentions.get(d.id || d.properties.iso_a3) || 0;
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

        // Filter out country-level locations
        const cityData = data.features.filter(feature => {
            // List of location types that should be considered as cities
            const countryNames = [
                "Burkina Faso", "Mali", "Côte d'Ivoire", "Bénin", "Togo", "Ghana", 
                "Nigéria", "Niger", "Tchad", "Libye", "Tunisie", "Maroc", "Mauritanie",
                "Sénégal", "Gambie", "Guinée", "Sierra Leone", "Liberia", "Cameroun",
                "Soudan", "Égypte", "Arabie saoudite", "Iran", "Irak", "Syrie", "Turquie",
                "Liban", "Israël", "Palestine", "Koweït", "Qatar", "Émirats arabes unis",
                "Afghanistan", "Pakistan", "Inde", "Bangladesh", "Chine", "États-Unis",
                "Canada", "Royaume-Uni", "Angleterre", "Irlande du Nord", "Allemagne",
                "Belgique", "Suisse", "Espagne", "Suède", "Somalie", "Rwanda",
                "Burundi", "Île Maurice", "Afrique du Sud", "France", "Algérie",
                "République centrafricaine", "République du Congo", "Bosnie-Herzégovine"
            ];
            
            const isCityLevel = !feature.properties.name.includes("République") && 
                !countryNames.includes(feature.properties.name);
            return isCityLevel;
        });

        const maxMentions = d3.max(cityData, d => d.properties.mentions);
        this.createScales(maxMentions);

        this.g.selectAll("circle")
            .data(cityData)
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
        const legendWidth = 300;
        const legendHeight = 40;
        const padding = 10;
        const boxWidth = 35;
        
        this.svg.selectAll(".legend").remove();
        
        const legend = this.svg.append("g")
            .attr("class", "legend")
            .attr("transform", `translate(${this.width - legendWidth - padding}, ${this.height - legendHeight - padding - 20})`);

        // Define the legend data with ranges
        const legendData = [
            { range: "0", color: MapConfig.colors.choropleth.noData },
            { range: "1-4", color: MapConfig.colors.choropleth.scale[0] },
            { range: "5-9", color: MapConfig.colors.choropleth.scale[1] },
            { range: "10-24", color: MapConfig.colors.choropleth.scale[2] },
            { range: "25-49", color: MapConfig.colors.choropleth.scale[3] },
            { range: "50-99", color: MapConfig.colors.choropleth.scale[4] },
            { range: "100-149", color: MapConfig.colors.choropleth.scale[5] },
            { range: "150+", color: MapConfig.colors.choropleth.scale[6] }
        ];

        // Add white background
        legend.append("rect")
            .attr("x", -padding)
            .attr("y", -padding)
            .attr("width", legendWidth + (padding * 2))
            .attr("height", legendHeight + (padding * 2))
            .attr("fill", "white")
            .attr("opacity", 0.9)
            .attr("rx", 5)  // Rounded corners
            .attr("ry", 5);

        // Add title
        legend.append("text")
            .attr("x", 0)
            .attr("y", 15)
            .style("font-size", "12px")
            .style("font-weight", "bold")
            .text("Number of mentions");

        // Create legend items
        const legendItems = legend.selectAll(".legend-item")
            .data(legendData)
            .enter()
            .append("g")
            .attr("class", "legend-item")
            .attr("transform", (d, i) => `translate(${i * boxWidth}, 25)`);

        // Add color boxes
        legendItems.append("rect")
            .attr("width", boxWidth - 2)
            .attr("height", 10)
            .attr("fill", d => d.color)
            .attr("stroke", "#999")
            .attr("stroke-width", 0.5);

        // Add labels
        legendItems.append("text")
            .attr("x", boxWidth/2 - 1)
            .attr("y", 25)
            .attr("text-anchor", "middle")
            .style("font-size", "10px")
            .text(d => d.range);
    }
} 