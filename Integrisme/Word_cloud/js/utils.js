// Utility functions
const Utils = {
    formatNumber(num) {
        return num.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
    },

    calculatePercentage(count, total) {
        return ((count / total) * 100).toFixed(2);
    },

    showError(message) {
        const container = document.getElementById('word-cloud');
        container.innerHTML = `
            <div class="error-message">
                <h3>Error</h3>
                <p>${message}</p>
            </div>
        `;
    },

    getWordColor(word) {
        for (const [category, words] of Object.entries(WORD_CATEGORIES)) {
            if (words.includes(word.toLowerCase())) {
                const intensity = Math.random() * 0.4 + 0.3;
                return CATEGORY_COLORS[category](intensity);
            }
        }
        return CATEGORY_COLORS.default(Math.random() * 0.4 + 0.3);
    }
}; 