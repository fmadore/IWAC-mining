// Main application logic
class WordCloudApp {
    constructor() {
        this.wordCloud = new WordCloud();
        this.sidebar = new Sidebar();
        this.initialize();
    }

    initialize() {
        this.setupEventListeners();
        this.loadData();
    }

    setupEventListeners() {
        const slider = document.getElementById('word-slider');
        const wordCount = document.getElementById('word-count');
        
        slider.addEventListener('input', () => {
            const numWords = parseInt(slider.value);
            wordCount.textContent = numWords;
            this.wordCloud.update(numWords);
        });
    }

    loadData() {
        const container = document.getElementById('word-cloud');
        container.innerHTML = '<div class="loading">Loading word cloud data...</div>';

        d3.json('word_frequencies.json')
            .then(data => {
                if (!data || Object.keys(data).length === 0) {
                    throw new Error('No data available');
                }
                this.wordCloud.allWords = Object.entries(data);
                this.wordCloud.update(parseInt(document.getElementById('word-slider').value));
            })
            .catch(error => {
                console.error('Error loading the JSON file:', error);
                Utils.showError("Failed to load word cloud data. Please try again later.");
            });
    }
}

// Initialize the application when the document is ready
document.addEventListener('DOMContentLoaded', () => {
    new WordCloudApp();
}); 