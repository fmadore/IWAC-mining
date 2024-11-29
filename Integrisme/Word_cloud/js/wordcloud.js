// Word cloud visualization
class WordCloud {
    constructor() {
        this.tooltip = new Tooltip();
        this.allWords = [];
        this.currentWords = [];
    }

    update(numWords) {
        d3.select('#word-cloud').selectAll("*").remove();
        
        if (!this.allWords.length) {
            Utils.showError("No words data available");
            return;
        }
        
        try {
            const maxFreq = Math.max(...this.allWords.map(([_, freq]) => freq));
            const minFreq = Math.min(...this.allWords.map(([_, freq]) => freq));
            
            const sizeScale = d3.scaleLinear()
                .domain([minFreq, maxFreq])
                .range([CONFIG.minFontSize, CONFIG.maxFontSize]);

            this.currentWords = this.allWords
                .slice(0, numWords)
                .map(([text, size]) => ({
                    text,
                    size: sizeScale(size),
                    frequency: size,
                    color: Utils.getWordColor(text)
                }));

            const totalFrequency = this.allWords.reduce((sum, [_, freq]) => sum + freq, 0);
            this.currentWords.forEach(word => {
                word.percentage = Utils.calculatePercentage(word.frequency, totalFrequency);
            });

            this.layout = d3.layout.cloud()
                .size([CONFIG.width, CONFIG.height])
                .words(this.currentWords)
                .padding(5)
                .rotate(() => Math.random() < 0.5 ? 0 : 90)
                .font('Arial')
                .fontSize(d => d.size)
                .spiral('rectangular')
                .on('end', words => this.draw(words));

            this.layout.start();
        } catch (error) {
            console.error('Error updating word cloud:', error);
            Utils.showError("Failed to update word cloud");
        }
    }

    draw(words) {
        const svg = d3.select('#word-cloud')
            .append('svg')
            .attr('width', CONFIG.width)
            .attr('height', CONFIG.height)
            .attr('viewBox', `0 0 ${CONFIG.width} ${CONFIG.height}`)
            .attr('preserveAspectRatio', 'xMidYMid meet')
            .append('g')
            .attr('transform', `translate(${CONFIG.width/2},${CONFIG.height/2})`);

        // Add ARIA labels for accessibility
        svg.attr('role', 'img')
           .attr('aria-label', 'Word cloud visualization');

        const wordElements = svg.selectAll('text')
            .data(words)
            .enter()
            .append('text')
            .attr('class', 'word')
            .style('font-size', '0px') // Start with size 0 for animation
            .style('font-family', 'Arial')
            .style('fill', d => d.color)
            .style('font-weight', 'bold')
            .attr('text-anchor', 'middle')
            .text(d => d.text)
            .attr('transform', d => `translate(${d.x},${d.y}) rotate(${d.rotate})`);

        // Add entrance animation
        wordElements.transition()
            .duration(600)
            .ease(d3.easeBackOut)
            .style('font-size', d => `${d.size}px`);

        // Add interactivity
        wordElements
            .on('mouseover', (event, d) => {
                const bbox = event.target.getBoundingClientRect();
                d.rank = this.currentWords.findIndex(w => w.text === d.text) + 1;
                d.total = this.currentWords.length;
                this.tooltip.show(event, d);
                
                // Highlight the word
                d3.select(event.target)
                    .transition()
                    .duration(200)
                    .style('opacity', 0.7)
                    .style('font-size', `${d.size * 1.1}px`);
            })
            .on('mouseout', (event) => {
                this.tooltip.hide();
                
                // Reset word style
                d3.select(event.target)
                    .transition()
                    .duration(200)
                    .style('opacity', 1)
                    .style('font-size', d => `${d.size}px`);
            });

        // Update sidebar
        const sidebar = new Sidebar();
        sidebar.update(this.currentWords);
    }
} 