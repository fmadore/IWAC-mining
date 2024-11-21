// Tooltip management
class Tooltip {
    constructor() {
        this.element = d3.select("body")
            .append("div")
            .attr("class", "tooltip")
            .style("opacity", 0);
    }

    show(event, data) {
        const bbox = event.target.getBoundingClientRect();
        const centerX = bbox.left + (bbox.width / 2);
        const topY = bbox.top - 10;
        
        this.element
            .style("visibility", "visible")
            .style("opacity", 1)
            .html(`
                <strong>Word:</strong> ${data.text}<br/>
                <strong>Frequency:</strong> ${Utils.formatNumber(data.frequency)}<br/>
                <strong>Percentage:</strong> ${data.percentage}%<br/>
                <strong>Rank:</strong> ${data.rank}/${data.total}
            `)
            .style("left", `${centerX}px`)
            .style("top", `${topY}px`)
            .style("transform", "translate(-50%, -100%)");
    }

    hide() {
        this.element
            .style("opacity", 0)
            .style("visibility", "hidden");
    }
} 