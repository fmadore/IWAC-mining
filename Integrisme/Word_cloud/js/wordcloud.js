class WordCloud {
    constructor() {
        this.svg = null;
        this.layout = null;
        this.tooltip = new Tooltip();
        this.sidebar = new Sidebar();
        this.allWords = [];
        this.currentWords = [];
        this.initialize();
    }

    initialize() {
        // Clear existing content
        d3.select("#word-cloud").html("");
        
        // Create SVG
        this.svg = d3.select("#word-cloud")
            .append("svg")
            .attr("width", CONFIG.wordCloud.width)
            .attr("height", CONFIG.wordCloud.height)
            .append("g")
            .attr("transform", `translate(${CONFIG.wordCloud.width/2},${CONFIG.wordCloud.height/2})`);

        // Initialize cloud layout
        this.layout = d3.layout.cloud()
            .size([CONFIG.wordCloud.width, CONFIG.wordCloud.height])
            .padding(CONFIG.wordCloud.padding)
            .rotate(() => CONFIG.wordCloud.rotationAngles[Math.floor(Math.random() * CONFIG.wordCloud.rotationAngles.length)])
            .font(CONFIG.wordCloud.fontFamily)
            .fontSize(d => this.calculateFontSize(d.size));
    }

    calculateFontSize(frequency) {
        const [min, max] = CONFIG.wordCloud.fontSizeScale;
        const scale = d3.scaleLog()
            .domain([1, this.maxFrequency])
            .range([min, max]);
        return scale(frequency);
    }

    update(numWords) {
        if (!this.allWords.length) return;

        this.currentWords = this.allWords
            .slice(0, numWords)
            .map((word, index) => ({
                text: word[0],
                size: word[1],
                frequency: word[1],
                rank: index + 1,
                total: numWords,
                percentage: Utils.calculatePercentage(word[1], this.allWords[0][1])
            }));

        this.maxFrequency = this.currentWords[0].frequency;

        this.layout
            .words(this.currentWords)
            .on("end", words => this.draw(words))
            .start();

        // Update sidebar
        this.sidebar.update(this.currentWords);
    }

    draw(words) {
        // Remove existing words
        this.svg.selectAll("text").remove();

        // Add new words
        const texts = this.svg.selectAll("text")
            .data(words)
            .enter().append("text")
            .style("font-size", d => `${d.size}px`)
            .style("font-family", CONFIG.wordCloud.fontFamily)
            .style("fill", d => Utils.getWordColor(d.text))
            .attr("text-anchor", "middle")
            .attr("transform", d => `translate(${d.x},${d.y}) rotate(${d.rotate})`)
            .text(d => d.text);

        // Add interactions
        texts
            .on("mouseover", (event, d) => this.tooltip.show(event, d))
            .on("mouseout", () => this.tooltip.hide());
    }
} 